# gda_search_app.py (upgraded)

import streamlit as st
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load all text files in extracted_text/
def load_documents():
    docs = {}
    folder_path = "extracted_text"
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                docs[filename] = f.read()
    return docs

# Break text into sentences (basic Hindi + English handling)
def split_sentences(text):
    import re
    return [s.strip() for s in re.split(r'[\n।.!?]', text) if len(s.strip()) > 10]

# Perform TF-IDF based search
def search_documents(query, documents, top_k=5):
    results = []
    for filename, text in documents.items():
        sentences = split_sentences(text)
        vectorizer = TfidfVectorizer().fit(sentences + [query])
        vectors = vectorizer.transform(sentences + [query])
        sim = cosine_similarity(vectors[-1], vectors[:-1]).flatten()

        top_indices = sim.argsort()[-top_k:][::-1]
        top_sentences = [(sentences[i], sim[i]) for i in top_indices if sim[i] > 0]
        results.append((filename, top_sentences))
    return results

# Streamlit UI
st.title("🔎 GDA Document Search (Hindi + English)")
st.markdown("Search across board meeting documents using any keyword or phrase.")

query = st.text_input("Enter your query (in Hindi or English):", max_chars=200)
show_all = st.checkbox("Show all matching sentences (not just top 5)")

if query:
    with st.spinner("Searching documents..."):
        docs = load_documents()
        result = search_documents(query, docs, top_k=999 if show_all else 5)

    for filename, matches in result:
        if matches:
            st.subheader(f"📄 {filename}")
            for sent, score in matches:
                st.markdown(f"- {sent}  ")

    if all(len(m) == 0 for _, m in result):
        st.info("No results found. Try a different term or spelling.")

