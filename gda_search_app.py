# gda_search_app.py (upgraded with full view + raw keyword search)

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

# Split text into paragraphs instead of just sentences
def split_paragraphs(text):
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]

# TF-IDF based search

def search_documents_tfidf(query, documents, top_k=5):
    results = []
    for filename, text in documents.items():
        paragraphs = split_paragraphs(text)
        vectorizer = TfidfVectorizer().fit(paragraphs + [query])
        vectors = vectorizer.transform(paragraphs + [query])
        sim = cosine_similarity(vectors[-1], vectors[:-1]).flatten()

        top_indices = sim.argsort()[-top_k:][::-1]
        top_paragraphs = [(paragraphs[i], sim[i]) for i in top_indices if sim[i] > 0]
        results.append((filename, top_paragraphs))
    return results

# Raw keyword match search

def search_documents_keyword(query, documents):
    results = []
    for filename, text in documents.items():
        paragraphs = split_paragraphs(text)
        matches = [(p, 1.0) for p in paragraphs if query.lower() in p.lower()]
        results.append((filename, matches))
    return results

# Streamlit UI
st.title("🔎 GDA Document Search (Hindi + English)")
st.markdown("Search across board meeting documents using keyword or phrase.")

query = st.text_input("Enter your query (in Hindi or English):", max_chars=200)
method = st.radio("Search Method:", ["🔬 Ranked (TF-IDF)", "🔤 Keyword Match (Full)"])

if query:
    with st.spinner("Searching documents..."):
        docs = load_documents()
        if method == "🔬 Ranked (TF-IDF)":
            results = search_documents_tfidf(query, docs, top_k=10)
        else:
            results = search_documents_keyword(query, docs)

    for filename, matches in results:
        if matches:
            st.subheader(f"📄 {filename}")
            for para, score in matches:
                highlighted = para.replace(query, f"**{query}**")
                st.markdown(f"- {highlighted}")

            with st.expander("📝 View Full Document"):
                st.markdown(docs[filename])

    if all(len(m) == 0 for _, m in results):
        st.info("No results found. Try a different term or spelling.")
