# src/data_loader.py

import os
import csv
import requests
import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
from urllib.parse import urlparse

# ─── PhishTank Downloader ──────────────────────────────────────────────────────

def fetch_phishtank_data(
    output_csv: str = "data/raw/phishtank.csv",
    phishtank_url: str = "https://data.phishtank.com/data/online-valid.csv"
):
    """
    Download the latest PhishTank 'online-valid.csv' and save to disk.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    print(f"Downloading PhishTank data from {phishtank_url} …")
    resp = requests.get(phishtank_url)
    resp.raise_for_status()
    with open(output_csv, "wb") as f:
        f.write(resp.content)
    print(f"Saved PhishTank data to {output_csv} (size={len(resp.content)} bytes)")

# ─── Benign Site Crawler ──────────────────────────────────────────────────────

def crawl_benign_sites(
    url_list_txt: str = "data/external/benign_list.txt",
    output_dir: str   = "data/raw/benign_html"
):
    """
    Read benign_list.txt (one URL per line), fetch each page, and save its HTML.
    Filenames will be benign_1.html, benign_2.html, …
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(url_list_txt, "r", newline="") as f:
        urls = [line.strip() for line in f if line.strip()]
    
    for idx, url in enumerate(urls, start=1):
        try:
            print(f"[{idx}/{len(urls)}] Fetching {url} …")
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            filename = f"benign_{idx}.html"
            path = os.path.join(output_dir, filename)
            with open(path, "w", encoding="utf-8") as out:
                out.write(resp.text)
        except Exception as e:
            print(f"  → ERROR fetching {url}: {e}")

# ─── Optional: Store Raw Data into PostgreSQL ────────────────────────────────

def push_raw_to_postgres(
    db_url: str,
    csv_path: str = "data/raw/phishtank.csv",
    table_name: str = "raw_phishtank"
):
    """
    Load the downloaded CSV into a pandas DataFrame and push into PostgreSQL.
    Requires `sqlalchemy` and a running database.
    """
    print(f"Loading {csv_path} into PostgreSQL table '{table_name}' …")
    df = pd.read_csv(csv_path)
    engine = create_engine(db_url)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print("Done.")

# ─── Example Usage ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1) Download phishing URLs CSV
    fetch_phishtank_data()

    # 2) Crawl benign URLs (make sure data/external/benign_list.txt exists)
    crawl_benign_sites()

    # 3) (Optional) Push to PostgreSQL—configure your DATABASE_URL
    # DATABASE_URL = "postgresql://user:pass@localhost:5432/phishingdb"
    # push_raw_to_postgres(DATABASE_URL)
