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


EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY belum diatur di environment variable!")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

OCR_WORKERS = 4
EMBED_BATCH = 32
TOP_K = 12          # dari 8 → 12
RERANK_TOP_K = 5    # dari 3 → 5
CONTEXT_CHAR_LIMIT = 9000  # dari 7000 → 9000
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
logger.info("Memuat model embedding dan reranker (ini bisa memakan waktu)...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL)
logger.info("Model siap.")

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
                logger.warning("Index lama bukan IndexIDMap, membungkus ulang agar add_with_ids bisa dipakai.")
                index = faiss.IndexIDMap(index)
            logger.info("Index FAISS dimuat dari disk.")
            return index
        except Exception as e:
            logger.warning(f"Gagal memuat index lama: {e}. Membuat index baru.")
    quant = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(quant)
    return index

faiss_index = None

# ========== UTILITIES ==========
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\x00", " ")
    return text.strip()

def chunk_text_by_sentence(text: str, chunk_size=700, overlap=100) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < chunk_size:
            current += " " + sent
        else:
            chunks.append(current.strip())
            if overlap and len(current) > overlap:
                current = current[-overlap:] + " " + sent
            else:
                current = sent
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c]

def generate_int64_id():
    uid = uuid.uuid4().int & ((1 << 63) - 1)
    return np.int64(uid)

# ========== PDF EXTRACTION & OCR (SEMUA HALAMAN + PREPROCESSING + DETEKSI ORIENTASI) ==========
def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Convert ke grayscale, rotate sesuai orientasi, dan tingkatkan kontras.
    """
    # deteksi orientasi dan rotasi otomatis
    try:
        osd = pytesseract.image_to_osd(img)
        rot = re.search(r'Rotate: (\d+)', osd)
        if rot:
            angle = int(rot.group(1))
            if angle != 0:
                img = img.rotate(-angle, expand=True)
    except Exception as e:
        logger.warning(f"Gagal deteksi orientasi: {e}")

    # grayscale
    img = ImageOps.grayscale(img)
    # optional: bisa ditambahkan binarization atau kontras
    return img

def ocr_page_worker(file_path: str, page_number: int) -> str:
    """
    OCR halaman tertentu, dengan preprocessing & orientasi otomatis.
    """
    try:
        images = convert_from_path(
            file_path, first_page=page_number, last_page=page_number, poppler_path=POPPLER_PATH
        )
        text = ""
        for img in images:
            img = preprocess_image(img)
            text += pytesseract.image_to_string(img, lang="ind") + "\n"
        return text
    except Exception as e:
        logger.error(f"OCR gagal pada {file_path} halaman {page_number}: {e}")
        return ""


def extract_text_from_pdf(file_path: str) -> str:
    """
    Hybrid extraction:
    - ambil text dengan page.get_text("text") dulu
    - jika halaman kosong/sedikit teks -> OCR
    """
    all_text = []
    ocr_pages = []
    with fitz.open(file_path) as doc:
        for i, page in enumerate(doc):
            raw = page.get_text("text") or ""
            cleaned = clean_text(raw)
            if len(cleaned) < 100:  # threshold, bisa disesuaikan
                ocr_pages.append(i + 1)  # 1-based
                all_text.append("")  # placeholder
            else:
                all_text.append(cleaned)

    if ocr_pages:
        with ThreadPoolExecutor(max_workers=OCR_WORKERS) as ex:
            results = list(ex.map(lambda p: ocr_page_worker(file_path, p), ocr_pages))
        ocr_idx = 0
        for i in range(len(all_text)):
            if all_text[i] == "":
                all_text[i] = clean_text(results[ocr_idx])
                ocr_idx += 1

    return "\n".join([t for t in all_text if t])

# ========== STORAGE: add chunks to DB and FAISS ==========
def add_chunks_to_store(filename: str, chunks: List[str]):
    global faiss_index
    if not chunks:
        return []

    embeddings = []
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i:i + EMBED_BATCH]
        emb = embed_model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)

    dim = embeddings.shape[1]
    if faiss_index is None:
        faiss_index = load_or_create_faiss(dim)

    cur = db_conn.cursor()
    saved_ids = []
    for idx, chunk in enumerate(chunks):
        int_id = int(generate_int64_id())
        u = str(uuid.uuid4())
        cur.execute("INSERT INTO chunks (id, uuid, text, filename, page) VALUES (?, ?, ?, ?, ?)",
                    (int_id, u, chunk, filename, -1))
        saved_ids.append(int_id)
    db_conn.commit()

    ids_array = np.array(saved_ids, dtype=np.int64)
    faiss_index.add_with_ids(embeddings, ids_array)
    faiss.write_index(faiss_index, INDEX_FILE)
    logger.info(f"Menambahkan {len(chunks)} potongan ke index dan metadata.")
    return saved_ids

# ========== RETRIEVAL ==========
def retrieve_relevant_snippets(question: str, top_k=TOP_K):
    global faiss_index
    if faiss_index is None:
        if os.path.exists(INDEX_FILE):
            faiss_index = faiss.read_index(INDEX_FILE)
        else:
            return []

    # Encode pertanyaan
    q_emb = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)

    # Ambil lebih banyak kandidat awal
    D, I = faiss_index.search(q_emb, k=max(top_k * 2, 16))
    ids = I[0]
    scores = D[0]

    snippets = []
    cur = db_conn.cursor()
    for idx, score in zip(ids, scores):
        if idx == -1:
            continue
        cur.execute("SELECT text FROM chunks WHERE id = ?", (int(idx),))
        row = cur.fetchone()
        if row and row[0].strip():
            snippets.append((row[0], float(score)))

    # Filter noise: buang snippet sangat pendek atau skor rendah
    snippets = [(s, sc) for s, sc in snippets if len(s) > 50 and sc > 0.3]

    # Dedup: jaga hanya potongan unik
    seen = set()
    filtered = []
    for s, sc in sorted(snippets, key=lambda x: x[1], reverse=True):
        key = s[:150].lower()
        if key not in seen:
            seen.add(key)
            filtered.append(s)
        if len(filtered) >= top_k:
            break

    return filtered


def rerank_snippets(question: str, snippets: List[str], top_n=RERANK_TOP_K):
    if not snippets:
        return []

    pairs = [[question, s] for s in snippets]
    scores = reranker.predict(pairs)

    # Gabungkan skor reranker dan potong bawah
    combined = list(zip(snippets, scores))
    combined.sort(key=lambda x: x[1], reverse=True)

    # Terapkan ambang batas minimal (misal 0.3)
    reranked = [s for s, sc in combined if sc > 0.3]
    return reranked[:top_n]


# ========== LLM CALL / PROMPT ==========
def call_llm_with_context(question: str, context_snippets: List[str]) -> str:
    if context_snippets:
        context = "\n\n".join(context_snippets)
        context = context[:CONTEXT_CHAR_LIMIT]
    else:
        context = ""

    system_instructions = (
        "Kamu adalah asisten yang menjawab pertanyaan dalam Bahasa Indonesia. "
        "Gunakan konteks yang diberikan jika relevan. "
        "JANGAN menyebutkan nama file secara eksplisit, tapi boleh menyebut halaman atau bagian jika relevan secara alami. "
        "Jika konteks benar-benar tidak cukup, balas dengan: Belum tersedia informasi."
    )

    prompt = f"""{system_instructions}

Konteks (jika ada):
{context}

Pertanyaan:
{question}

Jawaban singkat, profesional, dan mudah dipahami (bahasa Indonesia).
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Hybrid PDF Chatbot"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.25,
        "max_tokens": 800
    }

    try:
        res = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        answer = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Error memanggil LLM: {e}")
        return "Belum tersedia informasi"

    # Bersihkan jawaban
    lowered = answer.lower().strip()

    # Filter yang lebih longgar
    uncertain_markers = [
        "tidak tahu", "tidak ada informasi", "tidak menemukan", "unknown",
        "i don't know", "cannot find", "tidak disebutkan", "tidak tersedia"
    ]

    # Hanya tolak jika jawaban kosong atau jelas tidak relevan
    if not answer or len(answer) < 5 or any(m in lowered for m in uncertain_markers):
        return "Belum tersedia informasi"

    # Tidak langsung blok kata “halaman” atau “sumber”
    # Kecuali jika jawaban hanya isinya referensi
    if re.fullmatch(r".*(lihat|terdapat|tersedia).*?(halaman|file|lampiran).*", lowered):
        return "Belum tersedia informasi"

    # Pastikan tidak terlalu banyak noise
    if answer.count("\n") > 20 or len(answer) > 1500:
        answer = answer[:1200].rsplit(".", 1)[0] + "."

    return answer


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
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)

        logger.info(f"Menerima file: {file.filename}. Memulai ekstraksi...")
        text = extract_text_from_pdf(file_path)
        if not text:
            logger.warning(f"Tidak ada teks diekstrak dari {file.filename}")
            continue

        chunks = chunk_text_by_sentence(text)
        added = add_chunks_to_store(file.filename, chunks)
        total_chunks += len(added)

    return {"message": f"Berhasil memproses {len(files)} file, total potongan baru: {total_chunks}"}

@app.post("/ask")
async def ask(question: str = Form(...)):
    try:
        candidates = retrieve_relevant_snippets(question, top_k=TOP_K)
        if not candidates:
            return {"answer": "Belum tersedia informasi", "context_used": 0}

        top_snips = rerank_snippets(question, candidates, top_n=RERANK_TOP_K)
        answer = call_llm_with_context(question, top_snips)
        return {"answer": answer, "context_used": len(top_snips)}
    except Exception as e:
        logger.exception("Kesalahan di endpoint /ask")
        return JSONResponse(status_code=500, content={"answer": "Belum tersedia informasi", "context_used": 0})

@app.get("/health")
def health():
    return {"status": "ok"}

# ========== RUN ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
