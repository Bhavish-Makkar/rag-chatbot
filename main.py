import os
import zipfile
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

# =========================
# App Initialization
# =========================
app = FastAPI(title="RAG-based Chatbot", description="A FastAPI RAG chatbot with Groq", version="1.0")

# Enable CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Globals
client = None
retriever = None

# Load env vars
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in .env file")

# File paths
CWD = os.getcwd()
ZIP_PATH = os.path.join(CWD, "docs_db.zip")
PERSIST_DIR = os.path.join(CWD, "docs_db")
MARKER_FILE = os.path.join(PERSIST_DIR, ".extracted")
DOCUMENT_COLLECTION = "Assignment_documents"

# Models
MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# =========================
# Helper Functions
# =========================
def ensure_docs_extracted():
    """Ensure that docs_db folder exists by extracting docs_db.zip if needed."""
    if not os.path.exists(PERSIST_DIR):
        if os.path.exists(ZIP_PATH):
            print(f"Extracting {ZIP_PATH} ...")
            with zipfile.ZipFile(ZIP_PATH, "r") as z:
                z.extractall(CWD)

            # Flatten nested docs_db/docs_db if exists
            inner_path = os.path.join(PERSIST_DIR, "docs_db")
            if os.path.exists(inner_path):
                for item in os.listdir(inner_path):
                    os.rename(os.path.join(inner_path, item), os.path.join(PERSIST_DIR, item))
                os.rmdir(inner_path)

            with open(MARKER_FILE, "w") as f:
                f.write("extracted")
        else:
            raise FileNotFoundError("No docs_db folder or docs_db.zip found.")
    else:
        if not os.path.exists(MARKER_FILE):
            with open(MARKER_FILE, "w") as f:
                f.write("extracted")

# =========================
# Startup Event
# =========================
@app.on_event("startup")
def startup_event():
    """Load Groq client, embeddings, and retriever at startup."""
    global client, retriever

    ensure_docs_extracted()

    client = Groq(api_key=GROQ_API_KEY)

    embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        collection_name=DOCUMENT_COLLECTION,
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    print(" Startup complete — retriever and Groq client ready.")

# =========================
# Health Check Endpoint
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "message": "Backend is alive"}

# =========================
# Request/Response Models
# =========================
class AskRequest(BaseModel):
    question: str
    k: Optional[int] = 5
    include_context: Optional[bool] = False

class RetrievedChunk(BaseModel):
    content: str

class AskResponse(BaseModel):
    answer: str
    retrieved: Optional[List[RetrievedChunk]] = None

# =========================
# Main /ask Endpoint
# =========================
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Handles Q&A requests using RAG pipeline."""
    global client, retriever

    if client is None or retriever is None:
        raise HTTPException(status_code=500, detail="Server not initialized yet")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question provided")

    # Retrieve documents
    try:
        relevant_docs = retriever.get_relevant_documents(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retriever error: {str(e)}")

    if not relevant_docs:
        return AskResponse(answer="I don't know", retrieved=[] if req.include_context else None)

    # Prepare context
    context_list = [doc.page_content for doc in relevant_docs[: req.k or 5]]
    context_for_query = ". ".join(context_list)

    # Prompt
    system_prompt = """
You are a helpful assistant for Indigo Users.
Only use the provided ###Context to answer the question.
If the answer is not in the context, say "I don't know".
Do not reference the context explicitly in your answer.
"""
    user_prompt = f"""
###Context
{context_for_query}

###Question
{question}
"""

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]

    # Query Groq API
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0
        )
        answer_text = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    # Return result
    retrieved_payload = [RetrievedChunk(content=c) for c in context_list] if req.include_context else None
    return AskResponse(answer=answer_text, retrieved=retrieved_payload)
