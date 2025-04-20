# src/feature_extractor/link_features.py

import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

def is_valid_syntax_url(href: str) -> bool:
    """
    Syntactic URL check (no network I/O):
      • http:// or https:// with a netloc
      • mailto:, tel:
      • relative (/, ./, ../), fragments (#...)
    """
    h = href.strip()
    low = h.lower()
    if low.startswith(("http://", "https://")):
        p = urlparse(h)
        return bool(p.scheme in ("http", "https") and p.netloc)
    if low.startswith(("mailto:", "tel:")):
        return True
    if h.startswith(("/", "./", "../", "#")):
        return True
    return False

def extract_link_features(html: str, base_url: str) -> dict:
    """
    Extracts features F3–F13 from a page's HTML.

    Args:
      html     : the page's raw HTML
      base_url : the page's URL (for internal/external link distinction)

    Returns:
      A dict with keys:
        'F3_script_files'           : int
        'F4_css_files'              : int
        'F5_img_files'              : int
        'F6_a_files'                : int
        'F7_a_null_hyperlinks'      : int
        'F8_null_hyperlinks'        : int  # same as F7 by default
        'F9_total_hyperlinks'       : int
        'F10_internal_hyperlinks'   : int
        'F11_external_hyperlinks'   : int
        'F12_external_internal_ratio': float
        'F13_error_hyperlinks_rate' : float
    """
    soup = BeautifulSoup(html, "html.parser")

    # All <a href=...>
    anchors = soup.find_all("a", href=True)
    total_links = len(anchors)

    # F3: <script src=...>
    script_files = len(soup.find_all("script", src=True))

    # F4: <link rel="stylesheet" href=...>
    css_files = 0
    for link in soup.find_all("link", href=True):
        rels = link.get("rel") or []
        if any(r.lower() == "stylesheet" for r in rels):
            css_files += 1

    # F5: <img src=...>
    img_files = len(soup.find_all("img", src=True))

    # F6: # of <a href>
    a_files = total_links

    # F7: "null" anchors: href="" or "#" or javascript:...
    null_anchors = 0
    for a in anchors:
        h = a["href"].strip().lower()
        if h in ("", "#") or h.startswith("javascript:"):
            null_anchors += 1

    # F8: Null hyperlinks (same as F7 here; adjust if needed)
    null_hyperlinks = null_anchors

    # F10, F11: internal vs external anchors
    base_domain = urlparse(base_url).netloc
    internal_links = external_links = 0
    for a in anchors:
        href = a["href"]
        full = urljoin(base_url, href)
        dom = urlparse(full).netloc
        if dom == base_domain:
            internal_links += 1
        else:
            external_links += 1

    # F12: external/internal ratio
    if internal_links > 0:
        ext_int_ratio = external_links / internal_links
    else:
        # if no internal links, fallback to external/total (or zero if total=0)
        ext_int_ratio = external_links / total_links if total_links else 0.0

    # F13: error hyperlinks rate (invalid syntax only)
    invalid_links = 0
    for a in anchors:
        if not is_valid_syntax_url(a["href"]):
            invalid_links += 1
    error_rate = invalid_links / total_links if total_links else 0.0

    return {
        "F3_script_files": script_files,
        "F4_css_files": css_files,
        "F5_img_files": img_files,
        "F6_a_files": a_files,
        "F7_a_null_hyperlinks": null_anchors,
        "F8_null_hyperlinks": null_hyperlinks,
        "F9_total_hyperlinks": total_links,
        "F10_internal_hyperlinks": internal_links,
        "F11_external_hyperlinks": external_links,
        "F12_external_internal_ratio": ext_int_ratio,
        "F13_error_hyperlinks_rate": error_rate,
    }


# ─── Example Usage ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load an example HTML file and test
    import glob

    # Pick the first benign HTML and use its filename as base_url placeholder
    files = glob.glob("data/raw/benign_html/*.html")
    if not files:
        print("No HTML files found under data/raw/benign_html/")
    else:
        test_file = files[0]
        with open(test_file, "r", encoding="utf-8") as f:
            html = f.read()
        # Derive a base_url from the filename or hard‑code a real URL
        base_url = "https://example.com"
        feats = extract_link_features(html, base_url)
        print(f"Features from {test_file}:")
        for k, v in feats.items():
            print(f"  {k:30s} = {v}")
