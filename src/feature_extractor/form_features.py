# src/feature_extractor/form_features.py

from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

def is_suspicious_form(form, base_domain: str) -> bool:
    """
    A form is considered suspicious if:
    - action is empty, missing, or JavaScript-based
    - method is GET
    - form action submits to an external domain
    """
    action = form.get("action", "").strip().lower()
    method = form.get("method", "").strip().lower() or "get"  # default method is GET

    # Suspicious if action is empty or JavaScript
    if not action or action.startswith("javascript") or action in ("#", "void(0)"):
        return True

    # Suspicious if method is GET (not POST)
    if method == "get":
        return True

    # Suspicious if form submits to another domain
    target_url = urljoin(f"https://{base_domain}", action)
    target_domain = urlparse(target_url).netloc
    if target_domain != base_domain:
        return True

    return False

def extract_form_features(html: str, base_url: str) -> dict:
    """
    Extracts F14 (total_forms) and F15 (suspicious_form_ratio) from page HTML.

    Args:
      html      : the page's raw HTML
      base_url  : the full page URL (used to determine domain)

    Returns:
      {
        'F14_total_forms'        : int
        'F15_suspicious_form_ratio' : float
      }
    """
    soup = BeautifulSoup(html, "html.parser")
    forms = soup.find_all("form")
    total_forms = len(forms)
    base_domain = urlparse(base_url).netloc

    suspicious_count = 0
    for form in forms:
        if is_suspicious_form(form, base_domain):
            suspicious_count += 1

    suspicious_ratio = suspicious_count / total_forms if total_forms > 0 else 0.0

    return {
        "F14_total_forms": total_forms,
        "F15_suspicious_form_ratio": suspicious_ratio,
    }

# ─── Example Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob

    files = glob.glob("data/raw/benign_html/*.html")
    if not files:
        print("No HTML files found.")
    else:
        path = files[0]
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()

        base_url = "https://example.com"
        form_features = extract_form_features(html, base_url)
        print(f"Form features from {path}:")
        for k, v in form_features.items():
            print(f"  {k:35s} = {v}")
