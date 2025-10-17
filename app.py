import os
import re
import json
import uuid
import sqlite3
import logging
import tempfile
import requests
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor

import torch
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from nltk.tokenize import sent_tokenize
from PIL import Image, ImageOps

from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import faiss

# 🔧 BATASI THREAD (HEMAT RAM)
torch.set_num_threads(1)

# ========== CONFIG ==========
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

INDEX_FILE = "pdf_index.faiss"
SQLITE_FILE = "metadata.db"

if os.name == "nt":  # Windows
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    POPPLER_PATH = r"C:\Program Files\poppler-25.07.0\Library\bin"
else:  # Linux (Render)
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
    POPPLER_PATH = "/usr/bin"

# 🧠 MODEL RINGAN UNTUK RENDER
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
RERANKER_MODEL = "cross-encoder/qnli-distilroberta-base"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY belum diatur di environment variable!")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

OCR_WORKERS = 2  # kurangi worker agar hemat RAM
EMBED_BATCH = 16
TOP_K = 10
RERANK_TOP_K = 4
CONTEXT_CHAR_LIMIT = 9000
MAX_UPLOAD_SIZE = 500 * 1024 * 1024

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ========== FASTAPI ==========
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ========== MODELS ==========
logger.info("Memuat model embedding dan reranker ringan (optimasi untuk Render)...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL)
logger.info("Model siap digunakan.")

# ========== SQLITE HELPERS ==========
def init_db():
    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        uuid TEXT UNIQUE,
        text TEXT,
        filename TEXT,
        page INTEGER
    );
    """)
    conn.commit()
    return conn

db_conn = init_db()

# ========== FAISS HELPERS ==========
def load_or_create_faiss(dim: int):
    if os.path.exists(INDEX_FILE):
        try:
            index = faiss.read_index(INDEX_FILE)
            if not isinstance(index, faiss.IndexIDMap):
                index = faiss.IndexIDMap(index)
            logger.info("Index FAISS dimuat dari disk.")
            return index
        except Exception as e:
            logger.warning(f"Gagal memuat index lama: {e}. Membuat index baru.")
    quant = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap(quant)

faiss_index = None
if os.path.exists(INDEX_FILE):
    try:
        faiss_index = faiss.read_index(INDEX_FILE)
        logger.info("Index FAISS dimuat dari file repository.")
    except Exception as e:
        logger.warning(f"Gagal memuat FAISS index: {e}")

if not os.path.exists(SQLITE_FILE):
    db_conn = init_db()
else:
    db_conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    logger.info("Database metadata dimuat dari file repository.")

# ========== UTILITIES ==========
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()

def chunk_text_by_sentence(text: str, chunk_size=700, overlap=100) -> List[str]:
    sentences = sent_tokenize(text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) < chunk_size:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = current[-overlap:] + " " + sent if overlap and len(current) > overlap else sent
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c]

def generate_int64_id():
    return np.int64(uuid.uuid4().int & ((1 << 63) - 1))

# ========== OCR & PDF EXTRACTION ==========
def preprocess_image(img: Image.Image) -> Image.Image:
    try:
        osd = pytesseract.image_to_osd(img)
        rot = re.search(r'Rotate: (\d+)', osd)
        if rot:
            angle = int(rot.group(1))
            if angle != 0:
                img = img.rotate(-angle, expand=True)
    except Exception:
        pass
    return ImageOps.grayscale(img)

def ocr_page_worker(file_path: str, page_number: int) -> str:
    try:
        images = convert_from_path(file_path, first_page=page_number, last_page=page_number, poppler_path=POPPLER_PATH)
        text = ""
        for img in images:
            img = preprocess_image(img)
            text += pytesseract.image_to_string(img, lang="ind") + "\n"
        return text
    except Exception as e:
        logger.error(f"OCR gagal pada {file_path} halaman {page_number}: {e}")
        return ""

def extract_text_from_pdf(file_path: str) -> str:
    all_text, ocr_pages = [], []
    with fitz.open(file_path) as doc:
        for i, page in enumerate(doc):
            raw = page.get_text("text") or ""
            cleaned = clean_text(raw)
            if len(cleaned) < 100:
                ocr_pages.append(i + 1)
                all_text.append("")
            else:
                all_text.append(cleaned)
    if ocr_pages:
        with ThreadPoolExecutor(max_workers=OCR_WORKERS) as ex:
            results = list(ex.map(lambda p: ocr_page_worker(file_path, p), ocr_pages))
        for i, r in zip(ocr_pages, results):
            all_text[i - 1] = clean_text(r)
    return "\n".join([t for t in all_text if t])

# ========== STORAGE ==========
def add_chunks_to_store(filename: str, chunks: List[str]):
    global faiss_index
    if not chunks:
        return []
    embeddings = np.vstack([
        embed_model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        for batch in [chunks[i:i + EMBED_BATCH] for i in range(0, len(chunks), EMBED_BATCH)]
    ])
    dim = embeddings.shape[1]
    if faiss_index is None:
        faiss_index = load_or_create_faiss(dim)
    cur = db_conn.cursor()
    ids = []
    for chunk in chunks:
        int_id = int(generate_int64_id())
        u = str(uuid.uuid4())
        cur.execute("INSERT INTO chunks (id, uuid, text, filename, page) VALUES (?, ?, ?, ?, ?)",
                    (int_id, u, chunk, filename, -1))
        ids.append(int_id)
    db_conn.commit()
    faiss_index.add_with_ids(embeddings, np.array(ids, dtype=np.int64))
    faiss.write_index(faiss_index, INDEX_FILE)
    logger.info(f"Menambahkan {len(chunks)} potongan ke index dan metadata.")
    return ids

# ========== RETRIEVAL ==========
def retrieve_relevant_snippets(question: str, top_k=TOP_K):
    global faiss_index
    if faiss_index is None and os.path.exists(INDEX_FILE):
        faiss_index = faiss.read_index(INDEX_FILE)
    if faiss_index is None:
        return []
    q_emb = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    D, I = faiss_index.search(q_emb, k=max(top_k * 2, 16))
    cur = db_conn.cursor()
    snippets = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        cur.execute("SELECT text FROM chunks WHERE id = ?", (int(idx),))
        row = cur.fetchone()
        if row and row[0].strip():
            snippets.append((row[0], float(score)))
    snippets = [(s, sc) for s, sc in snippets if len(s) > 50 and sc > 0.3]
    seen, filtered = set(), []
    for s, sc in sorted(snippets, key=lambda x: x[1], reverse=True):
        key = s[:150].lower()
        if key not in seen:
            seen.add(key)
            filtered.append(s)
        if len(filtered) >= top_k:
            break
    return [s for s, _ in filtered]

def rerank_snippets(question: str, snippets: List[str], top_n=RERANK_TOP_K):
    if not snippets:
        return []
    pairs = [[question, s] for s in snippets]
    scores = reranker.predict(pairs)
    combined = list(zip(snippets, scores))
    combined.sort(key=lambda x: x[1], reverse=True)
    return [s for s, sc in combined if sc > 0.3][:top_n]

# ========== LLM ==========
def call_llm_with_context(question: str, context_snippets: List[str]) -> str:
    context = "\n\n".join(context_snippets)[:CONTEXT_CHAR_LIMIT] if context_snippets else ""
    system_instructions = (
        "Kamu adalah asisten AI yang menjawab pertanyaan dalam Bahasa Indonesia. "
        "Gunakan konteks yang diberikan jika relevan. Jika konteks tidak cukup, balas dengan: Belum tersedia informasi."
    )
    prompt = f"{system_instructions}\n\nKonteks:\n{context}\n\nPertanyaan:\n{question}"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Hybrid PDF Chatbot"
    }
    payload = {"model": OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.25, "max_tokens": 800}
    try:
        res = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Error memanggil LLM: {e}")
        return "Belum tersedia informasi"

# ========== ENDPOINTS ==========
@app.post("/upload_pdf")
async def upload_pdf(files: List[UploadFile]):
    total_chunks = 0
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            return JSONResponse(status_code=400, content={"message": "Hanya file PDF yang diperbolehkan."})
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_SIZE:
            return JSONResponse(status_code=400, content={"message": "Ukuran file melebihi batas."})
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(path, "wb") as f:
            f.write(contents)
        text = extract_text_from_pdf(path)
        if not text:
            continue
        chunks = chunk_text_by_sentence(text)
        total_chunks += len(add_chunks_to_store(file.filename, chunks))
    return {"message": f"Berhasil memproses {len(files)} file, total potongan baru: {total_chunks}"}

@app.post("/ask")
async def ask(question: str = Form(...)):
    try:
        cands = retrieve_relevant_snippets(question, top_k=TOP_K)
        if not cands:
            return {"answer": "Belum tersedia informasi", "context_used": 0}
        top_snips = rerank_snippets(question, cands, top_n=RERANK_TOP_K)
        ans = call_llm_with_context(question, top_snips)
        return {"answer": ans, "context_used": len(top_snips)}
    except Exception as e:
        logger.exception("Kesalahan di endpoint /ask")
        return JSONResponse(status_code=500, content={"answer": "Belum tersedia informasi", "context_used": 0})

@app.get("/health")
def health():
    return {"status": "ok"}

# ========== RUN ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # agar cocok dengan Render
    uvicorn.run("app:app", host="0.0.0.0", port=port)

