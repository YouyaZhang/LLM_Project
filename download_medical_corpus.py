# download_medical_corpus.py
from __future__ import annotations
import os, re, time, sys
from pathlib import Path
from typing import Dict, List, Tuple
import requests
from bs4 import BeautifulSoup

# ---------- Config ----------
DATA_DIR = Path("data")
INDEX_DIR = Path("indexes/bge_small_en")
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# 公开、权威、易读的英文健康文章（可增删）
URLS: Dict[str, str] = {
    # MedlinePlus (NIH)
    "headache.txt": "https://medlineplus.gov/ency/article/003024.htm",
    "sore_throat.txt": "https://medlineplus.gov/ency/article/000655.htm",
    "chest_pain.txt": "https://medlineplus.gov/ency/article/003079.htm",
    "fever.txt": "https://medlineplus.gov/ency/article/003090.htm",
    "abdominal_pain.txt": "https://medlineplus.gov/ency/article/003120.htm",
    # CDC
    "handwashing.txt": "https://www.cdc.gov/handwashing/about-handwashing.html",
    "flu_prevention.txt": "https://www.cdc.gov/flu/prevent/index.html",
    "dehydration.txt": "https://www.cdc.gov/nutrition/data-statistics/dehydration.html",
    "cold_vs_flu.txt": "https://www.cdc.gov/flu/symptoms/coldflu.htm",
    # WHO
    "healthy_diet.txt": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
    "physical_activity.txt": "https://www.who.int/news-room/fact-sheets/detail/physical-activity",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MedRAG/1.0; +https://example.org)"
}
TIMEOUT = 20
RETRIES = 3
RETRY_BACKOFF = 2.0


# ---------- Helpers ----------
def fetch_html(url: str) -> str:
    last_err = None
    for i in range(RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            time.sleep(RETRY_BACKOFF * (i + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")

def html_to_text(html: str) -> str:
    # 粗清洗：去脚本/样式、取正文文本、合并空白
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "iframe"]):
        tag.decompose()

    # 尝试优先 main/article 区域
    main = soup.find("main") or soup.find("article") or soup
    text = main.get_text(separator="\n")

    # 规范化空白
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t\u00A0]+", " ", text)
    # 压缩多行空白
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # 去一些常见页脚/导航噪声关键字（保守处理）
    noisy_patterns = [
        r"Page last reviewed.*", r"Page last updated.*", r"Back to top", r"Privacy Policy.*",
        r"Related links.*", r"Disclaimer.*", r"Site map.*"
    ]
    for pat in noisy_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # 再次压缩空行
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def save_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8", errors="ignore")

def download_all(urls: Dict[str, str]) -> List[Tuple[str, int]]:
    results = []
    for fname, url in urls.items():
        try:
            print(f"[DL] {fname} <- {url}")
            html = fetch_html(url)
            txt = html_to_text(html)
            save_text(DATA_DIR / fname, txt)
            results.append((fname, len(txt)))
        except Exception as e:
            print(f"[Error] {fname}: {e}")
    return results


# ---------- Validation with your RAG retriever ----------
def validate_with_retriever(test_query: str = "red flags for headache"):
    print("\n[Validate] Building/Loading FAISS index and querying top-3 ...")
    try:
        # 你的 DenseRetriever（我给你的版本）
        from retrieval.retriever import DenseRetriever
    except Exception as e:
        print("[Warn] Cannot import DenseRetriever:", e)
        print("       Skipping RAG validation. Files are downloaded and cleaned.")
        return

    ret = DenseRetriever(data_dir=str(DATA_DIR), index_dir=str(INDEX_DIR))
    print(f"[Validate] Passages: {len(ret.passages)}")
    if ret.index is not None:
        try:
            print(f"[Validate] Index size: {ret.index.ntotal}")
        except Exception:
            pass

    # 做一次实际检索
    hits = ret.retrieve(test_query, top_k=3)
    if not hits:
        print("[Validate] No hits for query:", repr(test_query))
        print("           Possible reasons: empty corpus, blocked downloads, or index mismatch.")
        return

    for i, h in enumerate(hits, 1):
        snippet = (h["text"] or "").replace("\n", " ")
        if len(snippet) > 180:
            snippet = snippet[:180] + " ..."
        print(f"\n[{i}] source={h['source']}  score={h['score']:.3f}\n{snippet}")


def main():
    print(f"[Init] Download target dir: {DATA_DIR.resolve()}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = download_all(URLS)
    total_kb = sum(sz for _, sz in results) / 1024.0
    print(f"\n[Done] Downloaded & cleaned {len(results)} files, total ~{total_kb:.1f} KB.")
    for fname, sz in results:
        print(f" - {fname}: {sz} chars")

    # 直接做验证（重建/加载索引 + 一次查询）
    validate_with_retriever(test_query="I have had a headache and mild fever. When should I seek care?")

if __name__ == "__main__":
    main()
