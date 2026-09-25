# This version uses the actual FAISS-retrieved chunks as context.
# It does not use question-specific keyword-based evidence selection.

import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import pickle
from transformers import pipeline

# ---------------------------------------------------------
# E-Commerce Security Risk Management Chatbot
# ---------------------------------------------------------
# Knowledge-base preparation used LangChain's
# RecursiveCharacterTextSplitter.
#
# Runtime application:
# Sentence Transformers -> FAISS -> retrieved context -> Qwen
# ---------------------------------------------------------

st.title("E-Commerce Security Risk Management Chatbot")

# Path handling: relative to this app.py file
BASE_DIR = Path(__file__).resolve().parent

# Simple semantic-relevance guard for out-of-scope questions.
# This is a heuristic threshold and can be tuned using validation questions.
RELEVANCE_THRESHOLD = 1.40
FAISS_K = 10
FINAL_K = 2


@st.cache_resource
def load_resources():
    """Load the embedding model, FAISS index, chunks and Qwen model once."""
    model = SentenceTransformer("all-MiniLM-L6-v2")

    index = faiss.read_index(
        str(BASE_DIR / "security.index")
    )

    with open(BASE_DIR / "chunks.pkl", "rb") as file:
        chunks = pickle.load(file)

    generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct"
    )
 
    reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return model, index, chunks, generator, reranker


model, index, chunks, generator, reranker = load_resources()

question = st.text_input("Ask a security question:")

if question.strip():

    # 1. Convert the user's question into an embedding
    query_embedding = model.encode([question])

    # 2. Retrieve the most semantically similar chunks from FAISS
    distances, indices = index.search(
        query_embedding.astype("float32"),
        FAISS_K
    )

    
    

    top_distance = float(distances[0][0])
    st.write("Top FAISS distance:", top_distance)

    # 3. Simple fallback for questions outside the knowledge base
    if top_distance > RELEVANCE_THRESHOLD:

        st.info(
            "This question appears to be outside the scope of the "
            "e-commerce security knowledge base."
        )

    else:

        # 4. Rerank the FAISS results using the Cross-Encoder
        candidate_indices = indices[0].tolist()
        candidate_chunks = [
            chunks[int(i)]
            for i in candidate_indices
        ]

        rerank_inputs = [
            (question, chunk)
            for chunk in candidate_chunks
        ]

        rerank_scores = reranker.predict(rerank_inputs)

        ranked_chunks = sorted(
            zip(
                rerank_scores,
                candidate_indices,
                candidate_chunks
            ),
            key=lambda x: x[0],
            reverse=True
        )

        # Keep only the best-ranked chunks
        final_chunks = ranked_chunks[:FINAL_K]

        context = "\n\n".join(
            f"[Retrieved Chunk {chunk_id}]\n{chunk}"
            for score, chunk_id, chunk in final_chunks
        )

        # 5. Show the final Cross-Encoder ranked evidence
                
        with st.expander("View Retrieved Evidence"):
            for rank, (score, chunk_id, chunk) in enumerate(final_chunks):
                st.write(f"Rank: {rank + 1}")
                st.write(f"Chunk ID: {chunk_id}")
                st.write(
                    f"Cross-Encoder Score: {float(score):.4f}"
                )
                st.write(chunk)
                st.write("---")

        # 6. Generate an answer using only the retrieved context
        prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the retrieved context below.

Rules:
- Answer only from the retrieved context.
- Give the direct answer first.
- Do not add information that is not necessary to answer the question.
- Do not mention authors, studies, methods, or references unless they directly answer the question.
- Do not contradict the retrieved context.
- If the answer is explicitly stated in the context, repeat that answer clearly.
- If the context does not support the answer, say:
  "The provided document does not contain enough information to answer this question."
- Keep the answer concise.
- If the question asks for names, methods, technologies, or a list, give only the directly supported items.
- Do not add explanations unless they are necessary to answer the question.
- Do not output references such as [Retrieved Chunk 5], [Chunk 34], or similar labels.
- Preserve the exact relationship between methods and their roles; do not reverse which method performs which action.
- Only use an acronym expansion when the exact expansion is explicitly present in the retrieved context.
- Never infer an acronym expansion from nearby words or concepts.
- When the question asks what a method is used for, state only its purpose in this research and do not add examples from other domains.
- Never invent or infer an acronym expansion. If the exact expansion is not explicitly present in the retrieved context, use only the acronym.

Retrieved Context:
{context}

Question:
{question}

Answer:
"""

        response = generator(
            prompt,
            max_new_tokens=40,
            do_sample=False,
            return_full_text=False
        )

        answer = response[0]["generated_text"].strip()

        st.subheader("Answer")
        st.write(answer)

        st.caption(
            "Source: Security Risk Management in E-commerce Systems: "
            "A Threat-driven Approach"
        )
