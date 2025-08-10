import os
import sys
import zipfile
from dotenv import load_dotenv
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

# === Load environment variables ===
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in .env.")

# Paths
zip_path = os.path.join(os.getcwd(), "docs_db.zip")
persist_dir = os.path.join(os.getcwd(), "docs_db")
marker_file = os.path.join(persist_dir, ".extracted")

# === Extract docs_db.zip only if needed ===
if not os.path.exists(persist_dir):
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path} for the first time...")
        with zipfile.ZipFile(~zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.getcwd())

        # Flatten nested docs_db/docs_db structure if exists
        inner_path = os.path.join(persist_dir, "docs_db")
        if os.path.exists(inner_path) and os.path.isdir(inner_path):
            print("Flattening extracted docs_db structure...")
            for item in os.listdir(inner_path):
                src_path = os.path.join(inner_path, item)
                dst_path = os.path.join(persist_dir, item)
                os.rename(src_path, dst_path)
            os.rmdir(inner_path)

        # Create marker so we skip this next time
        with open(marker_file, "w") as f:
            f.write("extracted")
    else:
        raise FileNotFoundError("No docs_db folder or docs_db.zip found.")
else:
    if not os.path.exists(marker_file):
        print("docs_db exists but no extraction marker found — skipping unzip.")

# === Initialize Groq client ===
client = Groq(api_key=api_key)

# === Model settings ===
model_name = 'openai/gpt-oss-20b'
embedding_model = SentenceTransformerEmbeddings(model_name='Qwen/Qwen3-Embedding-0.6B')

# === Load vector store ===
document_collections = "Assignment_documents"
vectorstore_persisted = Chroma(
    collection_name=document_collections,
    persist_directory=persist_dir,
    embedding_function=embedding_model
)

# Debug: Check document count
doc_count = vectorstore_persisted._collection.count()
print(f"Loaded Chroma collection '{document_collections}' with {doc_count} documents.")
if doc_count == 0:
    print(" Warning: Your vector DB is empty. Retrieval will return nothing.")

retriever = vectorstore_persisted.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 5}
)

# === Prompt templates ===
qna_system_message = """
You are a helpful assistant for Indigo Users.
User input will have the context required by you to answer user questions.
This context will begin with the token: ###Context.
The context contains references to specific portions of a document relevant to the user query.

User questions will begin with the token: ###Question.

Please answer user questions only using the context provided in the input.
Do not mention anything about the context in your final answer. Your response should only contain the answer to the question.

If the answer is not found in the context, respond "I don't know".
"""

qna_user_message_template = """
###Context
Here are some documents that are relevant to the question mentioned below.
{context}

###Question
{question}
"""

def ask_question(user_input: str):
    relevant_document_chunks = retriever.get_relevant_documents(user_input)

    if not relevant_document_chunks:
        print(" No relevant documents found for this query.")
        return "I don't know"

    print("\n Retrieved context chunks:")
    for i, doc in enumerate(relevant_document_chunks, 1):
        print(f"\n--- Chunk {i} ---\n{doc.page_content}")

    context_list = [d.page_content for d in relevant_document_chunks]
    context_for_query = ". ".join(context_list)

    prompt = [
        {'role': 'system', 'content': qna_system_message},
        {'role': 'user', 'content': qna_user_message_template.format(
            context=context_for_query,
            question=user_input
        )}
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=prompt,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    # Get question from CLI or use default
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "How do you create a new sourcing request in Ariba Sourcing?"

    answer = ask_question(question)
    print(f"\nQ: {question}\nA: {answer}")
