# gda_search_app.py
import streamlit as st
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import MarianMTModel, MarianTokenizer

# ----------- Load Documents ----------- #
def load_documents():
    docs = {}
    folder_path = "extracted_text"
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                docs[filename] = f.read()
    return docs

def split_paragraphs(text):
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]

# ----------- Translation Setup ----------- #
@st.cache_resource
def load_translation_models():
    hi_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    hi_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    en_hi_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    en_hi_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    return (hi_en_tokenizer, hi_en_model), (en_hi_tokenizer, en_hi_model)

def translate(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def is_hindi(text):
    return any('\u0900' <= ch <= '\u097F' for ch in text)

# ----------- TF-IDF Search ----------- #
def search_documents_tfidf(query, documents, hi_en, en_hi, top_k=5):
    results = []
    query_lang = "hi" if is_hindi(query) else "en"
    translated_query = translate(query, *hi_en) if query_lang == "hi" else translate(query, *en_hi)

    for filename, text in documents.items():
        paragraphs = split_paragraphs(text)
        all_queries = [query, translated_query]

        vectorizer = TfidfVectorizer().fit(paragraphs + all_queries)
        para_vecs = vectorizer.transform(paragraphs)
        query_vecs = vectorizer.transform(all_queries)

        sim_scores = np.max(cosine_similarity(query_vecs, para_vecs), axis=0)
        top_indices = sim_scores.argsort()[-top_k:][::-1]
        top_paragraphs = [(paragraphs[i], sim_scores[i]) for i in top_indices if sim_scores[i] > 0]
        results.append((filename, top_paragraphs))
    return results

# ----------- Keyword Match Search ----------- #
def search_documents_keyword(query, documents, hi_en, en_hi):
    results = []
    query_lang = "hi" if is_hindi(query) else "en"
    translated_query = translate(query, *hi_en) if query_lang == "hi" else translate(query, *en_hi)

    for filename, text in documents.items():
        paragraphs = split_paragraphs(text)
        matches = [
            (p, 1.0)
            for p in paragraphs
            if query.lower() in p.lower() or translated_query.lower() in p.lower()
        ]
        results.append((filename, matches))
    return results

# ----------- Streamlit UI ----------- #
st.title("🔎 GDA Document Search (Hindi + English)")
st.markdown("Search across board meeting documents using keyword or phrase. Hindi ↔ English supported.")

query = st.text_input("Enter your query (in Hindi or English):", max_chars=200)
method = st.radio("Search Method:", ["🔬 Ranked (TF-IDF)", "🔤 Keyword Match (Full)"])

if query:
    with st.spinner("Searching documents..."):
        docs = load_documents()
        hi_en, en_hi = load_translation_models()

        if method == "🔬 Ranked (TF-IDF)":
            results = search_documents_tfidf(query, docs, hi_en, en_hi, top_k=10)
        else:
            results = search_documents_keyword(query, docs, hi_en, en_hi)

    for filename, matches in results:
        if matches:
            st.subheader(f"📄 {filename}")
            for para, score in matches:
                st.markdown(f"- {para}  \n  *(Score: {score:.2f})*")

            with st.expander("📝 View Full Document"):
                st.markdown(docs[filename])

    if all(len(m) == 0 for _, m in results):
        st.info("No results found. Try a different term or spelling.")
