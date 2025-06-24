import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the extracted text
with open("169th-Board-Meeting.txt", "r", encoding="utf-8") as f:
    data = f.read()

# Create a search function
def search(query, documents, top_n=5):
    docs = [query] + documents
    tfidf = TfidfVectorizer().fit_transform(docs)
    cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    ranked = sorted(list(enumerate(cosine_sim)), key=lambda x: x[1], reverse=True)
    return [documents[i] for i, _ in ranked[:top_n]]

# Streamlit UI
st.title("🔍 GDA Board Meeting Search")
query = st.text_input("Enter your question (English or Hindi)")

if query:
    results = search(query, data.split("\n\n"))
    st.subheader("Top Matching Passages:")
    for r in results:
        st.write("- " + r)
