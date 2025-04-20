# src/feature_extractor/feature_builder.py

import numpy as np
from urllib.parse import urlparse

# URL features
from .url_features import load_url_tokenizer, urls_to_sequences

# HTML TF‑IDF features
from .html_features import load_tfidf_vectorizer, html_to_vector

# Link & resource features (F3–F13)
from .link_features import extract_link_features

# Form features (F14–F15)
from .form_features import extract_form_features

def build_feature_vector(url: str, html: str) -> np.ndarray:
    """
    Build the full feature vector [F1 ... F15] for a single webpage.

    Returns:
      1D numpy array of shape (URL_seq_len + TFIDF_dim + 11 + 2,)
    """
    # 1) F1: URL char‑level sequence
    tokenizer = load_url_tokenizer()
    # texts_to_sequences expects a list
    url_seq = urls_to_sequences([url], tokenizer)[0]  # shape: (MAX_URL_LENGTH,)

    # 2) F2: HTML char‑level TF‑IDF
    vectorizer = load_tfidf_vectorizer()
    tfidf_vec = html_to_vector(html, vectorizer).toarray()[0]  # shape: (MAX_FEATURES,)

    # 3) F3–F13: hyperlink & resource ratios/counts
    link_feats = extract_link_features(html, url)
    link_array = np.array(list(link_feats.values()), dtype=float)  # length 11

    # 4) F14–F15: form counts & ratio
    form_feats = extract_form_features(html, url)
    form_array = np.array(list(form_feats.values()), dtype=float)  # length 2

    # 5) Concatenate into one vector
    return np.concatenate([url_seq, tfidf_vec, link_array, form_array], axis=0)


if __name__ == "__main__":
    # Quick test on one example page
    import glob

    # 1) Pick an HTML file
    files = glob.glob("data/raw/benign_html/*.html")
    if not files:
        print("No HTML files found under data/raw/benign_html/")
        exit(1)

    test_html_path = files[0]
    with open(test_html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # 2) Choose a base URL matching your file (or hard‑code one)
    #    This only matters for domain-based features, e.g., F10/F11.
    url = "https://example.com"

    # 3) Build vector
    feature_vector = build_feature_vector(url, html)
    print("Feature vector shape:", feature_vector.shape)
    print("First 10 values (URL seq):", feature_vector[:10])
    print("… last 13 values (link + form):", feature_vector[-13:])
