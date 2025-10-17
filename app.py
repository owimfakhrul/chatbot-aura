import os
import re
import json
import sqlite3
import logging
import requests
import faiss
from typing import List
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ========== CONFIG ==========
INDEX_FILE = "pdf_index.faiss"
SQLITE_FILE = "metadata.db"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY belum diatur di environment variable!")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

TOP_K = 10
CONTEXT_CHAR_LIMIT = 9000

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ========== FASTAPI ==========
app = FastAPI(title="AURA Chatbot Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ========== LOAD STATIC INDEX & DB ==========
logger.info("📦 Memuat FAISS index dan database prebuilt...")
faiss_index = None
if os.path.exists(INDEX_FILE):
    try:
        faiss_index = faiss.read_index(INDEX_FILE)
        logger.info("✅ FAISS index dimuat dari file repository.")
    except Exception as e:
        logger.warning(f"Gagal memuat FAISS index: {e}")
else:
    logger.warning("⚠️ Tidak menemukan file pdf_index.faiss — chatbot akan berjalan tanpa konteks PDF.")

db_conn = None
if os.path.exists(SQLITE_FILE):
    db_conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    logger.info("✅ Database metadata dimuat dari file repository.")
else:
    logger.warning("⚠️ Tidak menemukan file metadata.db.")

# ========== RETRIEVAL (DISABLED FOR LIGHT MODE) ==========
def retrieve_relevant_snippets(question: str, top_k=TOP_K) -> List[str]:
    return []  # Tidak digunakan di mode ringan

# ========== LLM CALL ==========
def clean_output(text: str) -> str:
    """Membersihkan tag [OUT]...[/OUT], markdown, dan tanda coret."""
    if not text:
        return "Belum tersedia informasi"

    # Ambil hanya isi dalam [OUT]...[/OUT] jika ada
    out_match = re.findall(r"\[OUT\](.*?)\[/OUT\]", text, flags=re.DOTALL)
    if out_match:
        text = " ".join(out_match).strip()

    # Hapus markdown (bold, italic, coret, dsb)
    text = re.sub(r"[*_~`]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Bersihkan sisa tag atau bracket aneh
    text = text.replace("[/OUT]", "").replace("[OUT]", "")
    return text.strip()

def call_llm_with_context(question: str, context_snippets: List[str]) -> str:
    context = "\n\n".join(context_snippets)[:CONTEXT_CHAR_LIMIT] if context_snippets else ""
    system_instructions = (
        "Kamu adalah asisten AI yang menjawab pertanyaan dalam Bahasa Indonesia. "
        "Gunakan konteks yang diberikan jika relevan. Jika konteks tidak cukup, balas dengan: Belum tersedia informasi. "
        "Keluarkan jawaban final kamu di antara tag [OUT] dan [/OUT]."
    )
    prompt = f"{system_instructions}\n\nKonteks:\n{context}\n\nPertanyaan:\n{question}"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://chatbot-aura.onrender.com",
        "X-Title": "AURA Chatbot"
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
        raw_text = res.json()["choices"][0]["message"]["content"].strip()
        return clean_output(raw_text)
    except Exception as e:
        logger.error(f"Error memanggil LLM: {e}")
        return "Belum tersedia informasi"

# ========== ENDPOINTS ==========
@app.post("/ask")
async def ask(question: str = Form(...)):
    try:
        answer = call_llm_with_context(question, [])
        return {"answer": answer, "context_used": 0}
    except Exception as e:
        logger.exception("Kesalahan di endpoint /ask")
        return JSONResponse(status_code=500, content={"answer": "Belum tersedia informasi", "context_used": 0})

@app.get("/health")
def health():
    return {"status": "ok"}

# ========== RUN ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
