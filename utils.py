import json
import math
from io import BytesIO
from typing import Optional

import google.generativeai as genai  # type: ignore
from pypdf import PdfReader
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup

load_dotenv()

GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")
GENAI_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")

if not GENAI_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in environment; cannot use embeddings.")

genai.configure(api_key=GENAI_API_KEY)


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


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except:
            text = ""
        if text:
            pages.append(text)
    return "\n".join(pages)


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def fetch_url_text(url: str, max_chars: Optional[int] = 15000) -> str:
    """
    Download HTML, strip tags/scripts, and return concatenated text.
    """
    resp = requests.get(url, timeout=15, headers=DEFAULT_HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.stripped_strings)
    if max_chars:
        return text[:max_chars]
    return text
