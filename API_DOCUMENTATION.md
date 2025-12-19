# RAG Chatbot API Documentation

## Overview

This API provides a Retrieval Augmented Generation (RAG) chatbot service that allows you to:
- Upload PDF documents and extract content
- Ingest web content from URLs
- Ask questions based on uploaded/injected content
- Maintain conversation history with session management

## Base URL

```
http://localhost:8000
```

**Note:** Replace `localhost:8000` with your actual server URL when deploying.

## API Endpoints

---

### 1. Upload PDF Document

Upload a PDF file to extract and store its content for later querying.

**Endpoint:** `POST /upload/pdf`

**Content-Type:** `multipart/form-data`

**Request Parameters:**
- `file` (file, required): PDF file to upload

**Response:** `200 OK`

```json
{
  "stored_chunks": 15,
  "pdf_meta": {
    "page_count": 10,
    "pages_with_text": 10,
    "char_length": 12500
  }
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/upload/pdf" \
  -F "file=@document.pdf"
```

**JavaScript/Fetch Example:**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/upload/pdf', {
  method: 'POST',
  body: formData
});

const data = await response.json();
console.log(data);
```

---

### 2. Ingest URL Content

Extract and store content from a web page URL.

**Endpoint:** `POST /ingest/url`

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "url": "https://example.com/article",
  "source_type": "web"
}
```

**Request Fields:**
- `url` (string, required): Valid HTTP/HTTPS URL
- `source_type` (string, optional): Source type identifier (default: "web")

**Response:** `200 OK`

```json
{
  "stored_chunks": 8,
  "url": "https://example.com/article",
  "scrape_meta": {
    "removed_tags": {
      "script": 5,
      "style": 3,
      "noscript": 1
    },
    "original_char_length": 12000,
    "char_length": 12000,
    "truncated": false
  }
}
```

**Error Response:** `400 Bad Request`

```json
{
  "detail": "Failed to fetch URL: [error message]"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/ingest/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article"
  }'
```

**JavaScript/Fetch Example:**
```javascript
const response = await fetch('http://localhost:8000/ingest/url', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    url: 'https://example.com/article'
  })
});

const data = await response.json();
console.log(data);
```

---

### 3. Ask a Question

Ask a question based on uploaded documents and conversation history.

**Endpoint:** `POST /ask`

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "question": "What is the main topic of the document?",
  "session_id": "session_abc123"
}
```

**Request Fields:**
- `question` (string, required): The question to ask
- `session_id` (string, optional): Session ID for conversation continuity. If not provided, a new session will be created automatically.

**Response:** `200 OK`

```json
{
  "answer": "The main topic of the document is about artificial intelligence and machine learning...",
  "session_id": "session_abc123",
  "conversation_id": 42,
  "context_used": [
    "Relevant chunk 1 text...",
    "Relevant chunk 2 text...",
    "Relevant chunk 3 text..."
  ]
}
```

**Response Fields:**
- `answer` (string): The generated answer
- `session_id` (string): Session ID used for this conversation
- `conversation_id` (integer): Internal conversation ID
- `context_used` (array): Top 3 relevant document chunks used to generate the answer

**Error Response (No documents uploaded):**

```json
{
  "answer": "No data found. Please upload a PDF or crawl a website first.",
  "session_id": "session_abc123",
  "conversation_id": 42
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic?",
    "session_id": "session_abc123"
  }'
```

**JavaScript/Fetch Example:**
```javascript
const response = await fetch('http://localhost:8000/ask', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    question: 'What is the main topic?',
    session_id: 'session_abc123' // Optional
  })
});

const data = await response.json();
console.log(data.answer);
```

---

### 4. Create New Conversation Session

Create a new conversation session.

**Endpoint:** `POST /conversations/new`

**Response:** `200 OK`

```json
{
  "session_id": "session_a1b2c3d4e5f6",
  "conversation_id": 43,
  "created_at": "2024-01-15T10:30:00.000000"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/conversations/new"
```

**JavaScript/Fetch Example:**
```javascript
const response = await fetch('http://localhost:8000/conversations/new', {
  method: 'POST'
});

const data = await response.json();
const sessionId = data.session_id;
```

---

### 5. Get Conversation History

Retrieve the conversation history for a specific session.

**Endpoint:** `GET /conversations/{session_id}`

**Path Parameters:**
- `session_id` (string, required): The session ID

**Response:** `200 OK`

```json
{
  "conversation": {
    "conversation_id": 42,
    "session_id": "session_abc123",
    "message_count": 6,
    "created_at": "2024-01-15T10:00:00.000000",
    "updated_at": "2024-01-15T10:35:00.000000"
  },
  "messages": [
    {
      "role": "user",
      "content": "What is AI?"
    },
    {
      "role": "assistant",
      "content": "AI stands for Artificial Intelligence..."
    },
    {
      "role": "user",
      "content": "Can you give more details?"
    },
    {
      "role": "assistant",
      "content": "Certainly! Artificial Intelligence refers to..."
    }
  ]
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/conversations/session_abc123"
```

**JavaScript/Fetch Example:**
```javascript
const sessionId = 'session_abc123';
const response = await fetch(`http://localhost:8000/conversations/${sessionId}`);

const data = await response.json();
console.log(data.messages);
```

---

### 6. Clear Conversation History

Delete all messages from a conversation session.

**Endpoint:** `DELETE /conversations/{session_id}`

**Path Parameters:**
- `session_id` (string, required): The session ID

**Response:** `200 OK`

```json
{
  "session_id": "session_abc123",
  "cleared": true,
  "message": "Conversation history cleared"
}
```

**cURL Example:**
```bash
curl -X DELETE "http://localhost:8000/conversations/session_abc123"
```

**JavaScript/Fetch Example:**
```javascript
const sessionId = 'session_abc123';
const response = await fetch(`http://localhost:8000/conversations/${sessionId}`, {
  method: 'DELETE'
});

const data = await response.json();
console.log(data.message);
```

---

## Integration Examples

### Complete Workflow Example (JavaScript)

```javascript
// 1. Upload a PDF
async function uploadPDF(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:8000/upload/pdf', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
}

// 2. Create a new session
async function createSession() {
  const response = await fetch('http://localhost:8000/conversations/new', {
    method: 'POST'
  });
  
  const data = await response.json();
  return data.session_id;
}

// 3. Ask a question
async function askQuestion(question, sessionId) {
  const response = await fetch('http://localhost:8000/ask', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      question: question,
      session_id: sessionId
    })
  });
  
  return await response.json();
}

// 4. Get conversation history
async function getHistory(sessionId) {
  const response = await fetch(`http://localhost:8000/conversations/${sessionId}`);
  return await response.json();
}

// Usage example
async function example() {
  // Upload PDF
  const fileInput = document.getElementById('pdf-file');
  const uploadResult = await uploadPDF(fileInput.files[0]);
  console.log('Uploaded:', uploadResult);
  
  // Create session
  const sessionId = await createSession();
  console.log('Session ID:', sessionId);
  
  // Ask questions
  const answer1 = await askQuestion('What is this document about?', sessionId);
  console.log('Answer 1:', answer1.answer);
  
  const answer2 = await askQuestion('Can you summarize the key points?', sessionId);
  console.log('Answer 2:', answer2.answer);
  
  // Get history
  const history = await getHistory(sessionId);
  console.log('Conversation:', history);
}
```

### Python Integration Example

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Upload PDF
def upload_pdf(file_path):
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/upload/pdf", files=files)
    return response.json()

# 2. Ingest URL
def ingest_url(url):
    response = requests.post(
        f"{BASE_URL}/ingest/url",
        json={"url": url}
    )
    return response.json()

# 3. Create session
def create_session():
    response = requests.post(f"{BASE_URL}/conversations/new")
    return response.json()['session_id']

# 4. Ask question
def ask_question(question, session_id=None):
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    
    response = requests.post(
        f"{BASE_URL}/ask",
        json=payload
    )
    return response.json()

# Usage
if __name__ == "__main__":
    # Upload document
    pdf_result = upload_pdf("document.pdf")
    print(f"Uploaded {pdf_result['stored_chunks']} chunks")
    
    # Or ingest URL
    url_result = ingest_url("https://example.com/article")
    print(f"Ingested {url_result['stored_chunks']} chunks")
    
    # Create session
    session_id = create_session()
    print(f"Session ID: {session_id}")
    
    # Ask questions
    answer1 = ask_question("What is this about?", session_id)
    print(f"Answer: {answer1['answer']}")
    
    answer2 = ask_question("Tell me more", session_id)
    print(f"Answer: {answer2['answer']}")
```

---

## Error Handling

### Common Error Responses

**400 Bad Request**
```json
{
  "detail": "Error message describing what went wrong"
}
```

**422 Unprocessable Entity** (Validation Error)
```json
{
  "detail": [
    {
      "loc": ["body", "url"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**500 Internal Server Error**
```json
{
  "detail": "Internal server error"
}
```

### Best Practices

1. **Always check response status** before processing data
2. **Handle errors gracefully** with try-catch blocks
3. **Store session_id** after creating or receiving it to maintain conversation context
4. **Validate URLs** before calling the `/ingest/url` endpoint
5. **Check for document availability** before asking questions

---

## Rate Limiting

Currently, there are no rate limits implemented. However, consider implementing rate limiting in production environments.

---

## Authentication

Currently, the API does not require authentication. Consider adding authentication for production use.

---

## Data Models

### DocumentChunk
- `id` (integer): Unique identifier
- `source_type` (string): Type of source (e.g., "pdf", "web")
- `source` (string): Source identifier (filename or URL)
- `chunk_text` (string): Text content of the chunk
- `embedding` (string): JSON-encoded embedding vector
- `created_at` (datetime): Creation timestamp

### Conversation
- `id` (integer): Unique identifier
- `session_id` (string): Unique session identifier
- `created_at` (datetime): Creation timestamp
- `updated_at` (datetime): Last update timestamp

### ConversationMessage
- `id` (integer): Unique identifier
- `conversation_id` (integer): Reference to conversation
- `role` (string): Message role ("user" or "assistant")
- `content` (string): Message content
- `created_at` (datetime): Creation timestamp

---

## Notes

1. **Session Management**: Session IDs are used to maintain conversation context across multiple questions. Always include the `session_id` in subsequent requests to maintain context.

2. **Document Storage**: Documents are chunked and stored with embeddings. The system retrieves the top 3 most relevant chunks when answering questions.

3. **Conversation History**: The system maintains the last 10 messages in conversation history to provide context for generating answers.

4. **Embedding Model**: Uses Google's embedding model (text-embedding-004) for semantic search.

5. **Text Generation**: Uses Google's Gemini model (models/gemini-2.5-flash) for answer generation.

6. **Contact Information**: The AI assistant automatically includes contact information (phone, email, WhatsApp) in responses when users ask about inquiries, pricing, bookings, or need direct contact. Configure these via environment variables:
   - `CONTACT_PHONE`: Phone number (e.g., "+1234567890")
   - `CONTACT_EMAIL`: Email address (e.g., "support@example.com")
   - `CONTACT_WHATSAPP`: WhatsApp number (defaults to CONTACT_PHONE if not set)
   
   The assistant will automatically:
   - Include contact info when users ask about inquiries, pricing, quotes, bookings, etc.
   - Generate clickable WhatsApp links in the format: `https://wa.me/{number}`
   - Format responses professionally with contact details

---

## Environment Variables

To configure contact information, add these to your `.env` file:

```env
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional - Contact Information
CONTACT_PHONE=+1234567890
CONTACT_EMAIL=support@example.com
CONTACT_WHATSAPP=+1234567890
```

---

## Support

For issues or questions, please check the project repository or contact the development team.

