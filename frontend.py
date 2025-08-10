import streamlit as st
import requests

# -----------------------
# Backend API URL
# -----------------------
API_URL = "http://127.0.0.1:8000/ask"  # Change if deployed elsewhere

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("📚 RAG-based Chatbot")
st.write("Ask me questions based on my knowledge base.")

# User input
question = st.text_input("Your question:", placeholder="Type your question here...")
include_context = st.checkbox("Show retrieved context", value=False)

# Button to send request
if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        try:
            payload = {
                "question": question,
                "include_context": include_context
            }
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                data = response.json()
                st.subheader("💡 Answer:")
                st.write(data["answer"])

                if include_context and data.get("retrieved"):
                    st.subheader("📄 Retrieved Context:")
                    for idx, chunk in enumerate(data["retrieved"], start=1):
                        st.markdown(f"**Chunk {idx}:** {chunk['content']}")
            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {str(e)}")
