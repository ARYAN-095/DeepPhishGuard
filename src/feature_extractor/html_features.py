     # src/feature_extractor/html_features.py

import os
import re
import pickle
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

VECTORIZER_PATH = "models/tfidf_char_vectorizer.pkl"
MAX_FEATURES = 25000

# ─── Preprocessing Helpers ─────────────────────────────────

def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.extract()

    text = soup.get_text(separator=" ")
    text = re.sub(r"[^\w\s]", " ", text)            # remove punctuation
    text = re.sub(r"\d+", " ", text)                # remove numbers
    text = re.sub(r"\s+", " ", text).lower()        # normalize whitespace

    # Tokenize & stem
    tokens = word_tokenize(text)
    stemmer = SnowballStemmer("english")
    stop_words = set(stopwords.words("english"))
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]

    return " ".join(tokens)

# ─── TF-IDF Functions ──────────────────────────────────────

def build_tfidf_vectorizer(docs, save_path=VECTORIZER_PATH):
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 3),
        max_features=MAX_FEATURES
    )
    vectorizer.fit(docs)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(vectorizer, f)
    return vectorizer

def load_tfidf_vectorizer(path=VECTORIZER_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)

def html_to_vector(html, vectorizer) -> 'sparse matrix':
    clean = clean_html(html)
    return vectorizer.transform([clean])

# ─── Example Usage ─────────────────────────────────────────

 

# Inside html_features.py

if __name__ == "__main__":
    from glob import glob
    import numpy as np

    print("Collecting benign HTML pages...")
    files = glob("data/raw/benign_html/*.html")
    docs = []

    for path in files[:5]:  # test on first 5 pages
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
            cleaned = clean_html(html)
            docs.append(cleaned)

    print(f"Fitting TF-IDF on {len(docs)} documents...")
    vec = build_tfidf_vectorizer(docs)
    print("TF-IDF vectorizer saved.")

    print(f"Fitting TF-IDF on {len(docs)} documents...")
    vectorizer = build_tfidf_vectorizer(docs)
    print("TF-IDF vectorizer saved.")

    # Transform a few test HTML pages
    for i, doc in enumerate(docs):
        vec = vectorizer.transform([doc])
        print(f"Vector {i+1}: shape = {vec.shape}, non-zeros = {vec.nnz}")

