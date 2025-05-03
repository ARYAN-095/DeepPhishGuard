 
# DeepPhishGuard

**Phishing Web Page Detection with Semi‑Supervised Deep Anomaly Detection**

---

## 🚀 Project Overview

DeepPhishGuard is a Python‑based system to automatically detect phishing websites by combining URL‑level features, HTML/text analysis, hyperlink metrics, and form characteristics. We train an XGBoost classifier on handcrafted features (F1–F15) and offer a future extension to semi‑supervised anomaly detection using autoencoders.

---

## 🔑 Key Features

- **URL Character Sequences (F1):** Tokenize URLs at the character level to catch subtle obfuscations.  
- **Textual Content TF‑IDF (F2):** Clean HTML and noisy tag attributes → char‑level TF‑IDF vectors.  
- **Hyperlink Metrics (F3–F13):** Ratios/counts of `<a>`, `<img>`, `<script>`, CSS links, broken links, internal vs external, etc.  
- **Form Analysis (F14–F15):** Total vs suspicious forms (missing/JS action, GET method, cross‑domain).  
- **Classifier:** XGBoost outperforms other baselines (RandomForest, LogisticRegression, DNN).  
- **Extensible:** Phase 6 extension for autoencoder‑based semi‑supervised anomaly detection.

---

## 📦 Folder Structure
```
phishing-detector/
├── data/                    # Raw and processed datasets
│   ├── raw/                 # untouched HTML/CSV/JSON dumps
│   └── processed/           # cleaned CSV/Parquet feature tables
├── notebooks/               # Jupyter notebooks for EDA & prototyping
├── src/                     # All source code
│   ├── data_loader.py       # load from DB or files
│   ├── feature_extractor/   # modular F1–F15 extractors
│   │   ├── __init__.py
│   │   ├── url_features.py
│   │   ├── html_features.py
│   │   ├── link_features.py
│   │   └── form_features.py
│   ├── model/               
│   │   ├── trainer.py       # training pipelines
│   │   ├── evaluator.py     # metrics & reports
│   │   └── xgboost_wrapper.py
│   └── utils/               
│       ├── cleaner.py       # HTML/URL cleaning helpers
│       └── validator.py     # URL & HTML sanity checks
├── models/                  # output of training (e.g. .pkl, .h5)
├── outputs/                 # logs, plots, reports
├── app/                     
│   ├── main.py              # FastAPI service entrypoint
│   └── service.py           # prediction logic
├── tests/                   # pytest unit tests
├── Dockerfile               # containerization
├── requirements.txt         # production dependencies
├── dev-requirements.txt     # linting, formatting, testing tools
└── README.md                # project overview & instructions
```



---

## ⚙️ Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourorg/DeepPhishGuard.git
   cd DeepPhishGuard


2. Create & activate virtualenv

   python3 -m venv venv
source venv/bin/activate       # macOS/Linux
.\venv\Scripts\activate        # Windows

3.Install dependencies

   pip install -r requirements.txt

4.Download NLTK corpora

   python download_nltk.py



🗄️ Data Setup
  1.Phishing data
           -python -c "from src.data_loader import fetch_phishtank; fetch_phishtank()"

           saves data/raw/phishtank.csv

2.Benign data

 
python -c "from src.data_loader import crawl_benign; crawl_benign('data/external/benign_list.txt')"
saves HTML files under data/raw/.

3,Database (optional)
Create a PostgreSQL DB and update DATABASE_URL in .env:

 
DATABASE_URL=postgresql://user:pass@localhost:5432/phishingdb


# 🛠️ Feature Extraction & Training

Assemble features
 ```
python -c "from src.feature_extractor import build_dataset; build_dataset()"
outputs data/processed/features.parquet and labels.
```

# Train XGBoost

 ```
python -c "from src.model.trainer import Trainer; Trainer.train('data/processed/features.parquet')"
saves model to models/xgboost.pkl.
```

# Evaluate
``` 
python -c "from src.model.evaluator import Evaluator; Evaluator.evaluate('models/xgboost.pkl', 'data
```


# 🌐 API Usage

   Run the FastAPI service:

 ```
uvicorn app.main:app --reload
Then visit http://127.0.0.1:8000/docs for interactive docs.
```

Example request:

 ```
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"url":"http://example.com"}'
```


# ✅ Testing

 ```
pip install -r dev-requirements.txt
pytest --cov=src
```


# 🤝 Contributing

Fork the repo

Create a feature branch (git checkout -b feat/your-feature)

Commit your changes (git commit -m "Add feature")

Push (git push origin feat/your-feature)

Open a PR




# 📄 License
This project is licensed under the MIT License. See LICENSE for details.



Built with ♥ for robust phishing detection
  
This README gives readers a clear overview, step‑by‑step setup instructions, and pointers to every major component of your professional project. Let me know if you’d like any sections expanded or customized!
