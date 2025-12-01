import json
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, HttpUrl
from sqlalchemy import Column, Integer, Text, DateTime
from sqlalchemy.orm import Session
import datetime as dt


from db import Base, engine, get_db
from utils import (
    chunk_text,
    get_embedding,
    extract_text_from_pdf,
    cosine_similarity,
    fetch_url_text,
    generate_answer,
)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="MVP RAG Chatbot")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(Text)
    source = Column(Text)
    chunk_text = Column(Text)
    embedding = Column(Text)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class InteractionLog(Base):
    __tablename__ = "interaction_logs"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(Text, nullable=False)
    payload = Column(Text)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class URLIngestRequest(BaseModel):
    url: HttpUrl
    source_type: str | None = "web"


class QuestionRequest(BaseModel):
    question: str


def log_event(db: Session, event_type: str, payload: dict):
    db.add(InteractionLog(
        event_type=event_type,
        payload=json.dumps(payload)
    ))


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>MVP RAG Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f7f7f9; }
        h1 { margin-bottom: 0.5rem; }
        section { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.15); margin-bottom: 20px; }
        label { display: block; margin-bottom: 6px; font-weight: bold; }
        input[type="file"], textarea, input[type="text"] { width: 100%; margin-bottom: 10px; padding: 8px; }
        button { padding: 10px 14px; border: none; background-color: #2563eb; color: #fff; border-radius: 4px; cursor: pointer; }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        pre { background: #0f172a; color: #d1d5db; padding: 15px; border-radius: 6px; overflow: auto; }
    </style>
</head>
<body>
    <h1>MVP RAG Chatbot</h1>
    <p>Upload a PDF, then ask questions against its content.</p>

    <section>
        <h2>1. Upload PDF</h2>
        <form id="upload-form">
            <label for="pdf-file">Choose PDF</label>
            <input type="file" id="pdf-file" accept="application/pdf" required />
            <button type="submit" id="upload-btn">Upload PDF</button>
        </form>
        <div id="upload-result"></div>
    </section>

    <section>
        <h2>2. Crawl Website</h2>
        <form id="url-form">
            <label for="url-input">Page URL</label>
            <input type="text" id="url-input" placeholder="https://example.com/article" required />
            <button type="submit" id="url-btn">Fetch & Store</button>
        </form>
        <div id="url-result"></div>
    </section>

    <section>
        <h2>3. Ask a Question</h2>
        <form id="ask-form">
            <label for="question">Question</label>
            <input type="text" id="question" placeholder="Type your question..." required />
            <button type="submit" id="ask-btn">Ask</button>
        </form>
        <div>
            <h3>Answer</h3>
            <pre id="answer"></pre>
        </div>
    </section>

    <script>
        const uploadForm = document.getElementById("upload-form");
        const uploadBtn = document.getElementById("upload-btn");
        const uploadResult = document.getElementById("upload-result");
        uploadForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById("pdf-file");
            if (!fileInput.files.length) return;
            uploadBtn.disabled = true;
            uploadResult.textContent = "Uploading...";
            const formData = new FormData();
            formData.append("file", fileInput.files[0]);
            try {
                const resp = await fetch("/upload/pdf", { method: "POST", body: formData });
                const data = await resp.json();
                uploadResult.textContent = JSON.stringify(data, null, 2);
            } catch (err) {
                uploadResult.textContent = "Upload failed: " + err;
            } finally {
                uploadBtn.disabled = false;
            }
        });

        const urlForm = document.getElementById("url-form");
        const urlBtn = document.getElementById("url-btn");
        const urlResult = document.getElementById("url-result");
        urlForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const url = document.getElementById("url-input").value.trim();
            if (!url) return;
            urlBtn.disabled = true;
            urlResult.textContent = "Fetching...";
            try {
                const resp = await fetch("/ingest/url", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ url })
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || "Failed");
                urlResult.textContent = JSON.stringify(data, null, 2);
            } catch (err) {
                urlResult.textContent = "Error: " + err;
            } finally {
                urlBtn.disabled = false;
            }
        });

        const askForm = document.getElementById("ask-form");
        const askBtn = document.getElementById("ask-btn");
        const answerEl = document.getElementById("answer");
        askForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const question = document.getElementById("question").value.trim();
            if (!question) return;
            askBtn.disabled = true;
            answerEl.textContent = "Thinking...";
            try {
                const resp = await fetch("/ask", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ question })
                });
                const data = await resp.json();
                answerEl.textContent = JSON.stringify(data, null, 2);
            } catch (err) {
                answerEl.textContent = "Error: " + err;
            } finally {
                askBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
    """


@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_bytes = await file.read()
    text, pdf_meta = extract_text_from_pdf(file_bytes)

    chunks = chunk_text(text)
    chunk_details = []
    for ch in chunks:
        emb = get_embedding(ch)
        db.add(DocumentChunk(
            source_type="pdf",
            source=file.filename,
            chunk_text=ch,
            embedding=json.dumps(emb)
        ))
        chunk_details.append({
            "length": len(ch),
            "preview": ch[:160]
        })

    log_event(db, "pdf_upload", {
        "filename": file.filename,
        "chunk_count": len(chunks),
        "pdf_meta": pdf_meta,
        "chunk_details": chunk_details
    })
    db.commit()

    return {"stored_chunks": len(chunks), "pdf_meta": pdf_meta}


@app.post("/ingest/url")
async def ingest_url(payload: URLIngestRequest, db: Session = Depends(get_db)):
    try:
        text, scrape_meta = fetch_url_text(str(payload.url))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {exc}") from exc
    if not text.strip():
        raise HTTPException(status_code=400, detail="No readable text extracted from the page.")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Unable to chunk extracted text.")

    chunk_details = []
    for ch in chunks:
        emb = get_embedding(ch)
        db.add(DocumentChunk(
            source_type=payload.source_type or "web",
            source=str(payload.url),
            chunk_text=ch,
            embedding=json.dumps(emb)
        ))
        chunk_details.append({
            "length": len(ch),
            "preview": ch[:160]
        })

    log_event(db, "url_ingest", {
        "url": str(payload.url),
        "source_type": payload.source_type or "web",
        "chunk_count": len(chunks),
        "scrape_meta": scrape_meta,
        "chunk_details": chunk_details
    })
    db.commit()

    return {
        "stored_chunks": len(chunks),
        "url": str(payload.url),
        "scrape_meta": scrape_meta
    }


@app.post("/ask")
async def ask_question(payload: QuestionRequest, db: Session = Depends(get_db)):
    q_emb = get_embedding(payload.question)

    docs = db.query(DocumentChunk).all()
    if not docs:
        return {"answer": "No data found. Upload PDF first."}

    scored = []
    for d in docs:
        emb = json.loads(d.embedding)
        scored.append((cosine_similarity(q_emb, emb), d.chunk_text))

    top = [text for s, text in sorted(scored, reverse=True)[:3]]

    context = "\n\n".join(top)

    try:
        answer_text = generate_answer(context, payload.question)
    except Exception as exc:
        answer_text = (
            "Answer generation failed; showing raw context instead.\n\n" + context
        )

    log_event(db, "question", {
        "question": payload.question,
        "context_used": top,
        "answer": answer_text
    })
    db.commit()

    return {
        "context_used": top,
        "answer": answer_text
    }
