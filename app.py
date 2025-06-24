import streamlit as st
import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load all text files in extracted_text/
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

def is_hindi(text):
    return any('\u0900' <= ch <= '\u097F' for ch in text)

def translate_libretranslate(text, source_lang, target_lang):
    try:
        response = requests.post("https://libretranslate.de/translate", json={
            "q": text,
            "source": source_lang,
            "target": target_lang,
            "format": "text"
        })
        if response.status_code == 200:
            return response.json()["translatedText"]
    except Exception as e:
        print("Translation error:", e)
    return text  # fallback: return original

def search_documents(query, documents, method="tfidf", top_k=5):
    query_lang = "hi" if is_hindi(query) else "en"
    translated_query = translate_libretranslate(query, query_lang, "en" if query_lang == "hi" else "hi")
    queries = [query, translated_query]

    results = []
    for filename, text in documents.items():
        paragraphs = split_paragraphs(text)
        if method == "tfidf":
            vectorizer = TfidfVectorizer().fit(paragraphs + queries)
            para_vecs = vectorizer.transform(paragraphs)
            query_vecs = vectorizer.transform(queries)
            sim_scores = cosine_similarity(query_vecs, para_vecs).max(axis=0)
            top_indices = sim_scores.argsort()[-top_k:][::-1]
            top_paragraphs = [(paragraphs[i], sim_scores[i]) for i in top_indices if sim_scores[i] > 0.1]
        else:
            top_paragraphs = [(p, 1.0) for p in paragraphs if any(q.lower() in p.lower() for q in queries)]

        if top_paragraphs:
            results.append((filename, top_paragraphs))
    return results

# Streamlit UI
st.title("🔎 GDA Document Search (Hindi + English + Translation)")
query = st.text_input("Enter your query (Hindi or English):")
method = st.radio("Search Method:", ["🔬 Ranked (TF-IDF)", "🔤 Keyword Match (Full)"])

if query:
    with st.spinner("Translating and searching documents..."):
        docs = load_documents()
        search_mode = "tfidf" if method.startswith("🔬") else "keyword"
        results = search_documents(query, docs, method=search_mode)

    if not results:
        st.warning("No relevant results found.")
    else:
        for fname, matches in results:
            st.subheader(f"📄 {fname}")
            for para, score in matches:
                st.markdown(f"- {para}\n\n  _Score: {score:.2f}_")

            with st.expander("📝 View Full Document"):
                st.markdown(docs[fname])
