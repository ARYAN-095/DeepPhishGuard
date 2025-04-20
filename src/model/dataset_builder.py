# src/model/dataset_builder.py

import os
import glob
import pickle
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import mlflow
import mlflow.xgboost

from src.feature_extractor.feature_builder import build_feature_vector

RAW_PHISH_CSV    = "data/raw/phishtank.csv"
PHISH_HTML_DIR   = "data/raw/phishing_html"
BENIGN_HTML_DIR  = "data/raw/benign_html"
BENIGN_URL_LIST  = "data/external/benign_list.txt"
OUTPUT_DIR       = "data/processed"

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def fetch_and_save_phish_html():
    """Download phishing page HTMLs to PHISH_HTML_DIR."""
    ensure_dir(PHISH_HTML_DIR)
    df = pd.read_csv(RAW_PHISH_CSV)
    for idx, url in enumerate(tqdm(df['url'], desc="Crawling phishing"), start=1):
        path = os.path.join(PHISH_HTML_DIR, f"phish_{idx}.html")
        if not os.path.exists(path):
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                with open(path, "w", encoding="utf-8") as f:
                    f.write(r.text)
            except:
                # skip failures
                pass

def build_dataset():
    """Build X (features) and y (labels) and save them."""
    ensure_dir(OUTPUT_DIR)
    # 1) Ensure HTML
    fetch_and_save_phish_html()

    records = []
    # 2) Phishing records (label=1)
    phish_files = sorted(glob.glob(f"{PHISH_HTML_DIR}/*.html"))
    for path in tqdm(phish_files, desc="Processing phishing HTML"):
        idx = int(os.path.basename(path).split("_")[1].split(".")[0]) - 1
        url = pd.read_csv(RAW_PHISH_CSV)['url'][idx]
        html = open(path, "r", encoding="utf-8").read()
        vec  = build_feature_vector(url, html)
        records.append((vec, 1))

    # 3) Benign records (label=0)
    with open(BENIGN_URL_LIST) as f:
        benign_urls = [u.strip() for u in f if u.strip()]
    benign_files = sorted(glob.glob(f"{BENIGN_HTML_DIR}/*.html"))
    for path, url in tqdm(zip(benign_files, benign_urls), desc="Processing benign HTML", total=len(benign_urls)):
        html = open(path, "r", encoding="utf-8").read()
        vec  = build_feature_vector(url, html)
        records.append((vec, 0))

    # 4) Split vectors and labels
    X = np.vstack([r[0] for r in records])
    y = np.array([r[1] for r in records])

    # 5) Save
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    print(f"Dataset built: X={X.shape}, y={y.shape}")
    return X, y


def train_with_mlflow():
    mlflow.set_experiment("phishing_detection")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(grid.best_params_)

        # Log CV metrics
        mlflow.log_metric("best_cv_auc", grid.best_score_)

        # After final eval
        mlflow.log_metric("test_auc", roc_auc_score(y_test, y_proba))

        # Save and log the model
        mlflow.xgboost.log_model(best_clf, artifact_path="model")

    print(f"Run info: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    build_dataset()
