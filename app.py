
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from transformers import pipeline

st.title("E-Commerce Security Risk Management Chatbot")

model = SentenceTransformer("all-MiniLM-L6-v2")

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct"
)

index = faiss.read_index(
    r"C:\Users\Pooja\ecommerce_security_chatbot\security.index"
)

with open(
    r"C:\Users\Pooja\ecommerce_security_chatbot\chunks.pkl",
    "rb"
) as file:
    chunks = pickle.load(file)

question = st.text_input("Ask a security question:")

if question:

    query_embedding = model.encode([question])

    distances, indices = index.search(
        query_embedding.astype("float32"),
        5
    )

    retrieved_chunks = [chunks[i] for i in indices[0]]

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
Answer the question using only the information in the context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = generator(
        prompt,
        max_new_tokens=100,
        return_full_text=False
    )

    st.write(response[0]["generated_text"])
