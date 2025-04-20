# src/feature_extractor/url_features.py

import os
import pickle
from typing import List
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration
TOKENIZER_PATH = "models/url_tokenizer.pkl"
MAX_URL_LENGTH = 200   # adjust if needed
VOCAB_SIZE     = 300   # adjust based on your dataset

def build_url_tokenizer(
    urls: List[str],
    save_path: str = TOKENIZER_PATH,
    vocab_size: int = VOCAB_SIZE
) -> Tokenizer:
    """Fit a char-level tokenizer on given URLs and save it."""
    tok = Tokenizer(
        num_words=vocab_size,
        char_level=True,
        oov_token="<UNK>"
    )
    tok.fit_on_texts(urls)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(tok, f)
    return tok

def load_url_tokenizer(save_path: str = TOKENIZER_PATH) -> Tokenizer:
    """Load a previously saved URL tokenizer."""
    with open(save_path, "rb") as f:
        return pickle.load(f)

def urls_to_sequences(
    urls: List[str],
    tokenizer: Tokenizer,
    max_len: int = MAX_URL_LENGTH
):
    """Convert list of URLs to padded integer sequences."""
    seqs = tokenizer.texts_to_sequences(urls)
    padded = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
    return padded

# Example usage
if __name__ == "__main__":
    # 1) Read your URLs (e.g. from data/raw/phishtank.csv or benign_list.txt)
    import pandas as pd
    df = pd.read_csv("data/raw/phishtank.csv")
    all_urls = df['url'].tolist()

    # 2) Build & save tokenizer
    tok = build_url_tokenizer(all_urls)

    # 3) Convert to sequences
    X = urls_to_sequences(all_urls, tok)
    print("URL sequences shape:", X.shape)
