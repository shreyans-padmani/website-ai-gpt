"""
Utility functions for RAG chatbot operations.

This module provides:
- Text processing and chunking
- Embedding generation
- PDF and web content extraction
- Answer generation with context
"""
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
GENAI_TEXT_MODEL = os.getenv("GOOGLE_TEXT_MODEL", "models/gemini-2.5-flash")

# Contact information for inquiries
CONTACT_PHONE = os.getenv("CONTACT_PHONE", "")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "")
CONTACT_WHATSAPP = os.getenv("CONTACT_WHATSAPP", CONTACT_PHONE)  # WhatsApp number (can be same as phone)

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
    Use a text generation model to generate an answer with context.
    Handles conversation history and document context.
    Includes contact information for inquiries when appropriate.
    """
    # Build contact information section
    contact_info = ""
    if CONTACT_PHONE or CONTACT_EMAIL or CONTACT_WHATSAPP:
        contact_lines = []
        
        if CONTACT_PHONE:
            contact_lines.append(f" Phone: {CONTACT_PHONE}")
        
        if CONTACT_EMAIL:
            contact_lines.append(f" Email: {CONTACT_EMAIL}")
        
        if CONTACT_WHATSAPP:
            # Format WhatsApp number (remove + if present, add country code if needed)
            whatsapp_num = CONTACT_WHATSAPP.replace("+", "").replace(" ", "").replace("-", "")
            whatsapp_link = f"https://wa.me/{whatsapp_num}"
            contact_lines.append(f" WhatsApp: {CONTACT_WHATSAPP} (Direct message: {whatsapp_link})")
        
        contact_info = "\n\n" + "\n".join(contact_lines)
    
    prompt = (
        "You are a helpful and professional assistant for a retrieval augmented generation system.\n"
        "Use the provided context (which may include previous conversation and relevant documents) "
        "to answer the user's question accurately and concisely.\n"
        "If the context does not contain enough information to answer the question, "
        "politely inform the user and suggest they reach out for more detailed information.\n"
        "Maintain conversation flow and refer to previous exchanges when relevant.\n\n"
        "IMPORTANT INSTRUCTIONS:\n"
        "- When a user asks about pricing, quotes, booking, appointments, purchasing, or any inquiry "
        "that requires direct contact, always provide the contact information below.\n"
        "- If the question requires clarification or personalized assistance beyond what's in the context, "
        "politely direct them to contact support.\n"
        "- Format your responses professionally and include contact information naturally when relevant.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
        f"{contact_info if contact_info else ''}"
    )
    
    try:
        response = _text_model.generate_content(prompt)
        candidate = response.candidates[0]
        parts = getattr(candidate.content, "parts", [])
        text = "".join(getattr(part, "text", "") for part in parts)
        answer = text.strip() or "No answer generated."
        
        # Append contact info if answer indicates inquiry/contact needed and not already included
        if contact_info and any(keyword in question.lower() for keyword in [
            "contact", "reach", "phone", "email", "whatsapp", "inquiry", "inquire", 
            "price", "pricing", "cost", "quote", "book", "booking", "appointment",
            "purchase", "buy", "order", "support", "help", "assistance"
        ]):
            if CONTACT_PHONE not in answer and CONTACT_EMAIL not in answer:
                answer += f"\n\n---\n For inquiries, please reach out:\n"
                if CONTACT_PHONE:
                    answer += f"Phone: {CONTACT_PHONE}\n"
                if CONTACT_EMAIL:
                    answer += f"Email: {CONTACT_EMAIL}\n"
                if CONTACT_WHATSAPP:
                    whatsapp_num = CONTACT_WHATSAPP.replace("+", "").replace(" ", "").replace("-", "")
                    whatsapp_link = f"https://wa.me/{whatsapp_num}"
                    answer += f"WhatsApp: {CONTACT_WHATSAPP} - Click to message: {whatsapp_link}\n"
        
        return answer
    except Exception as e:
        raise RuntimeError(f"Failed to generate answer: {str(e)}") from e
