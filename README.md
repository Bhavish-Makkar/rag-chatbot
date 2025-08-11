# RAG-based Chatbot with Groq & ChromaDB

A Retrieval-Augmented Generation (RAG) chatbot powered by **Groq LLM**, **Chroma vector database**, and **Streamlit**.  
It allows you to query a pre-built knowledge base and get precise, context-aware answers.

---

## Features
- **FastAPI backend** serving a `/ask` endpoint for RAG queries.
- **Chroma vector database** for semantic search & context retrieval.
- **Groq LLM API** for high-performance responses.
- **Streamlit frontend** for an interactive chat-like interface.
- **Optional context view** to debug or understand retrieved chunks.
- Automatic extraction of `docs_db.zip` on backend startup.

---

##  Project Structure
```
.
├── Creating_A_Vector_Database.ipynb    # Notebook to create the Chroma vector DB
├── main.py                              # FastAPI backend
├── streamlit_app.py                     # Streamlit frontend
├── docs_db.zip                          # Pre-built vector DB (zipped)
├── .env                                 # Environment variables (GROQ_API_KEY)
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## Prerequisites
- **Python** ≥ 3.9
- **Groq API Key** → [Get it here](https://console.groq.com/)
- `docs_db.zip` generated from your dataset (created via the Jupyter notebook).

---

##  Installation

1️ **Clone the repository**
```bash
git clone https://github.com/Bhavish-Makkar/rag-chatbot
cd rag-chatbot
```

2️ **Install dependencies**
```bash
pip install -r requirements.txt
```

3️ **Set environment variables**  
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

4️ **Ensure your vector DB exists**  
If you don’t have `docs_db.zip`, run the `Creating_A_Vector_Database.ipynb` to generate it.

---

##  Running the Application

### 1. Start the Backend
```bash
uvicorn main:app --reload
```
This will run FastAPI at `http://127.0.0.1:8000`



### 2. Start the Frontend
```bash
streamlit run streamlit_app.py
```
This will open the chatbot UI in your browser.

---

##  Usage
- Enter your question in the text box.
- (Optional) Enable **"Show retrieved context"** to see the actual chunks retrieved from the DB.
- Click **Ask** to get a precise, context-based answer from the Groq model.

---

##  API Reference

### `POST /ask`
**Request body:**
```json
{
  "question": "What is RAG?",
  "k": 5,
  "include_context": true
}
```

**Response:**
```json
{
  "answer": "RAG stands for Retrieval-Augmented Generation...",
  "retrieved": [
    {"content": "Chunk text 1..."},
    {"content": "Chunk text 2..."}
  ]
}
```

---

##  Creating the Vector Database
Run the provided notebook:
```bash
jupyter notebook Creating_A_Vector_Database.ipynb
```
This:
1. Loads your documents.
2. Generates embeddings using `Qwen/Qwen3-Embedding-0.6B`.
3. Saves them into `docs_db/` which can be zipped for portability.

---

