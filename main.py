import json
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session
import datetime as dt
import uuid


from db import Base, engine, get_db
from models import (
    DocumentChunk,
    InteractionLog,
    Conversation,
    ConversationMessage,
)
from utils import (
    chunk_text,
    get_embedding,
    extract_text_from_pdf,
    cosine_similarity,
    fetch_url_text,
    generate_answer,
)
from memory import MemoryService

Base.metadata.create_all(bind=engine)

app = FastAPI(title="MVP RAG Chatbot")


class URLIngestRequest(BaseModel):
    url: HttpUrl
    source_type: str | None = "web"


class QuestionRequest(BaseModel):
    question: str
    session_id: str | None = None  # Optional session ID for conversation continuity


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
        <div style="margin-bottom: 10px;">
            <label for="session-id">Session ID (for conversation continuity):</label>
            <input type="text" id="session-id" placeholder="Leave empty for new session" style="width: 70%;" />
            <button type="button" id="new-session-btn" style="width: 28%; margin-left: 2%;">New Session</button>
        </div>
        <form id="ask-form">
            <label for="question">Question</label>
            <input type="text" id="question" placeholder="Type your question..." required />
            <button type="submit" id="ask-btn">Ask</button>
        </form>
        <div>
            <h3>Answer</h3>
            <pre id="answer"></pre>
            <div id="session-info" style="margin-top: 10px; font-size: 0.9em; color: #666;"></div>
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

        let currentSessionId = null;
        
        const newSessionBtn = document.getElementById("new-session-btn");
        newSessionBtn.addEventListener("click", async () => {
            try {
                const resp = await fetch("/conversations/new", { method: "POST" });
                const data = await resp.json();
                currentSessionId = data.session_id;
                document.getElementById("session-id").value = currentSessionId;
                document.getElementById("session-info").textContent = 
                    `New session created: ${currentSessionId}`;
            } catch (err) {
                alert("Failed to create new session: " + err);
            }
        });

        const askForm = document.getElementById("ask-form");
        const askBtn = document.getElementById("ask-btn");
        const answerEl = document.getElementById("answer");
        const sessionInfoEl = document.getElementById("session-info");
        const sessionIdInput = document.getElementById("session-id");
        
        askForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const question = document.getElementById("question").value.trim();
            if (!question) return;
            
            const sessionId = sessionIdInput.value.trim() || currentSessionId;
            askBtn.disabled = true;
            answerEl.textContent = "Thinking...";
            sessionInfoEl.textContent = "";
            
            try {
                const payload = { question };
                if (sessionId) {
                    payload.session_id = sessionId;
                }
                
                const resp = await fetch("/ask", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const data = await resp.json();
                
                // Display answer
                answerEl.textContent = data.answer || JSON.stringify(data, null, 2);
                
                // Update session info
                if (data.session_id) {
                    currentSessionId = data.session_id;
                    sessionIdInput.value = data.session_id;
                    sessionInfoEl.textContent = 
                        `Session: ${data.session_id} | Conversation ID: ${data.conversation_id}`;
                }
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
    """Ask a question with conversation context memory."""
    memory = MemoryService(db)
    
    # Get or create conversation session
    session_id = payload.session_id or f"session_{dt.datetime.utcnow().timestamp()}"
    conversation = memory.get_or_create_conversation(session_id)
    
    # Get conversation history for context
    conversation_history = memory.get_recent_context(conversation.id, max_messages=10)
    
    # Store user question
    memory.add_message(conversation.id, "user", payload.question)
    
    # Retrieve relevant document chunks
    q_emb = get_embedding(payload.question)
    docs = db.query(DocumentChunk).all()
    
    if not docs:
        answer_text = "No data found. Please upload a PDF or crawl a website first."
        memory.add_message(conversation.id, "assistant", answer_text)
        db.commit()
        return {
            "answer": answer_text,
            "session_id": session_id,
            "conversation_id": conversation.id
        }

    # Score and rank document chunks
    scored = []
    for d in docs:
        emb = json.loads(d.embedding)
        scored.append((cosine_similarity(q_emb, emb), d.chunk_text))

    top = [text for s, text in sorted(scored, reverse=True)[:3]]
    document_context = "\n\n".join(top)

    # Combine document context with conversation history
    full_context = ""
    if conversation_history:
        full_context = f"Previous conversation:\n{conversation_history}\n\n"
    full_context += f"Relevant documents:\n{document_context}"

    # Generate answer with context
    try:
        answer_text = generate_answer(full_context, payload.question)
    except Exception as exc:
        log_event(db, "question_error", {
            "question": payload.question,
            "session_id": session_id,
            "error": str(exc)
        })
        answer_text = (
            "I apologize, but I encountered an error generating the answer. "
            "Please try rephrasing your question."
        )

    # Store assistant response
    memory.add_message(conversation.id, "assistant", answer_text)

    log_event(db, "question", {
        "question": payload.question,
        "session_id": session_id,
        "conversation_id": conversation.id,
        "context_used": top,
        "answer": answer_text
    })
    db.commit()

    return {
        "answer": answer_text,
        "session_id": session_id,
        "conversation_id": conversation.id,
        "context_used": top
    }


@app.get("/conversations/{session_id}")
async def get_conversation(session_id: str, db: Session = Depends(get_db)):
    """Get conversation history for a session."""
    memory = MemoryService(db)
    conversation = memory.get_or_create_conversation(session_id)
    history = memory.get_conversation_history(conversation.id)
    summary = memory.get_conversation_summary(conversation.id)
    
    return {
        "conversation": summary,
        "messages": history
    }


@app.delete("/conversations/{session_id}")
async def clear_conversation(session_id: str, db: Session = Depends(get_db)):
    """Clear conversation history for a session."""
    memory = MemoryService(db)
    conversation = memory.get_or_create_conversation(session_id)
    cleared = memory.clear_conversation(conversation.id)
    
    return {
        "session_id": session_id,
        "cleared": cleared,
        "message": "Conversation history cleared" if cleared else "No messages to clear"
    }


@app.post("/conversations/new")
async def create_new_conversation(db: Session = Depends(get_db)):
    """Create a new conversation session."""
    session_id = f"session_{uuid.uuid4().hex[:12]}"
    memory = MemoryService(db)
    conversation = memory.get_or_create_conversation(session_id)
    
    return {
        "session_id": session_id,
        "conversation_id": conversation.id,
        "created_at": conversation.created_at.isoformat() if conversation.created_at else None
    }
