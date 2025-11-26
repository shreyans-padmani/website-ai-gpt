import json
import math
from io import BytesIO
from typing import Optional, Dict

import google.generativeai as genai  # type: ignore
from pypdf import PdfReader
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup

load_dotenv()

GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")
GENAI_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")
GENAI_TEXT_MODEL = os.getenv("GOOGLE_TEXT_MODEL", "models/gemini-2.0-flash")

if not GENAI_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in environment; cannot use embeddings.")

genai.configure(api_key=GENAI_API_KEY)
_text_model = genai.GenerativeModel(GENAI_TEXT_MODEL)


def chunk_text(text: str, max_chars: int = 800):
    text = text.replace("\r", " ").replace("\n", " ")
    return [
        text[i:i + max_chars].strip()
        for i in range(0, len(text), max_chars)
        if text[i:i + max_chars].strip()
    ]


def get_embedding(text: str):
    resp = genai.embed_content(
        model=GENAI_EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document"
    )
    return resp["embedding"]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_text_from_pdf(file_bytes: bytes):
    reader = PdfReader(BytesIO(file_bytes))
    total_pages = len(reader.pages)
    extracted_pages = 0
    page_texts = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            extracted_pages += 1
            page_texts.append(text)
    combined = "\n".join(page_texts)
    meta = {
        "page_count": total_pages,
        "pages_with_text": extracted_pages,
        "char_length": len(combined),
    }
    return combined, meta


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def fetch_url_text(url: str, max_chars: Optional[int] = 15000):
    """
    Download HTML, strip tags/scripts, and return concatenated text + metadata.
    """
    resp = requests.get(url, timeout=15, headers=DEFAULT_HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    removed_counts: Dict[str, int] = {}
    for tag in soup(["script", "style", "noscript"]):
        removed_counts[tag.name] = removed_counts.get(tag.name, 0) + 1
        tag.decompose()
    full_text = " ".join(soup.stripped_strings)
    truncated = False
    if max_chars and len(full_text) > max_chars:
        truncated = True
        text = full_text[:max_chars]
    else:
        text = full_text
    meta = {
        "removed_tags": removed_counts,
        "original_char_length": len(full_text),
        "char_length": len(text),
        "truncated": truncated
    }
    return text, meta


def generate_answer(context: str, question: str) -> str:
    """
    Use a text generation model to summarize context into an answer.
    """
    prompt = (
        "You are a helpful assistant for a retrieval augmented generation system.\n"
        "Given the context chunks below, answer the user's question concisely.\n"
        "If the context does not contain the answer, say you don't know based on the provided data.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    response = _text_model.generate_content(prompt)
    print(response)
    candidate = response.candidates[0]
    parts = getattr(candidate.content, "parts", [])
    text = "".join(getattr(part, "text", "") for part in parts)
    return text.strip() or "No answer generated."
