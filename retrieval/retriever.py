# retrieval/retriever.py
from __future__ import annotations
import os, json, glob, faiss, math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np

# 尝试使用 sentence-transformers（更省心）；如无则退回 transformers
_USE_ST = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    _USE_ST = False
    from transformers import AutoTokenizer, AutoModel
    import torch


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # 兼容 {"text": "..."} 或其他字段名
                if isinstance(obj, dict):
                    if "text" in obj and obj["text"]:
                        yield (obj["text"], f"{Path(path).name}:{i}")
                    else:
                        # 没有 text 字段时，尝试把整行当成文本（保底）
                        yield (json.dumps(obj, ensure_ascii=False), f"{Path(path).name}:{i}")
                else:
                    yield (str(obj), f"{Path(path).name}:{i}")
            except json.JSONDecodeError:
                # 非 JSONL 纯文本兜底
                yield (line, f"{Path(path).name}:{i}")


def _iter_txt(path: str):
    # .txt/.md 等整体作为一个文档；分行切成小段
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    buf, ln_no = [], 0
    for ln in lines:
        ln_no += 1
        if ln:
            buf.append(ln)
        if len(buf) >= 5:  # 累积到一定行数就提交一段
            yield (" ".join(buf), f"{Path(path).name}:{ln_no}")
            buf = []
    if buf:
        yield (" ".join(buf), f"{Path(path).name}:{ln_no}")


def _word_chunks(text: str, win: int = 120, overlap: int = 40) -> List[str]:
    toks = text.split()
    n = len(toks)
    i, out = 0, []
    while i < n:
        j = min(n, i + win)
        out.append(" ".join(toks[i:j]).strip())
        if j == n:
            break
        i = max(i + win - overlap, i + 1)
    return [c for c in out if c]


def _normalize(a: np.ndarray) -> np.ndarray:
    # Cosine 相似度 = 归一化后点积
    n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    return a / n


@dataclass
class Passage:
    text: str
    source: str  # filename:line 或自定义来源字符串


class DenseRetriever:
    """
    FAISS + BGE-small-en-v1.5 稠密检索。
    - 构建：扫描 data/ 目录下 .jsonl / .txt / .md，切块（120词窗口，40词重叠）
    - 编码：BGE-small（384维），可选归一化；FAISS IndexFlatIP（点积=余弦）
    - 查询：对 query 编码后，与索引做 top-k 近邻检索；返回 [{'text','source','score'}, ...]
    """

    def __init__(
        self,
        data_dir: str = "data",
        index_dir: str = "indexes/bge_small_en",
        model_name: str = "BAAI/bge-small-en-v1.5",
        normalize_embeddings: bool = True,
        chunk_win: int = 120,
        chunk_overlap: int = 40,
        batch_size: int = 128,
    ):
        self.data_dir = data_dir
        self.index_dir = index_dir
        self.model_name = model_name
        self.normalize = normalize_embeddings
        self.chunk_win = chunk_win
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

        os.makedirs(self.index_dir, exist_ok=True)
        self.index_path = os.path.join(self.index_dir, "faiss.index")
        self.meta_path = os.path.join(self.index_dir, "passages.jsonl")

        self.passages: List[Passage] = []
        self.index: Optional[faiss.Index] = None

        # init encoder
        if _USE_ST:
            self.encoder = SentenceTransformer(self.model_name, device="cpu")
            try:
                self.encoder = self.encoder.to("cuda")
            except Exception:
                pass
        else:
            self.tok = AutoTokenizer.from_pretrained(self.model_name)
            self.enc = AutoModel.from_pretrained(self.model_name)
            try:
                self.enc = self.enc.to("cuda")
            except Exception:
                pass

        # try load existing index
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self._load_index()
        else:
            self.build_index()

    # ----------- public API -----------

    def build_index(self):
        """重建索引：扫描文件 -> 切块 -> 编码 -> 建 FAISS -> 保存。"""
        self.passages = self._scan_and_chunk()
        if not self.passages:
            # 保底：空索引
            self.index = faiss.IndexFlatIP(384)
            self._save_meta()
            faiss.write_index(self.index, self.index_path)
            return

        embs = self._embed_passages([p.text for p in self.passages])
        if self.normalize:
            embs = _normalize(embs)

        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs.astype(np.float32))

        self._save_meta()
        faiss.write_index(self.index, self.index_path)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if not query or not query.strip():
            return []
        if self.index is None:
            self._load_index()
        qvec = self._embed_queries([query])
        if self.normalize:
            qvec = _normalize(qvec)
        D, I = self.index.search(qvec.astype(np.float32), top_k)
        res = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or idx >= len(self.passages):
                continue
            p = self.passages[idx]
            res.append({"text": p.text, "source": p.source, "score": float(score)})
        return res

    # ----------- inner helpers -----------

    def _scan_and_chunk(self) -> List[Passage]:
        psgs: List[Passage] = []
        # 支持 .jsonl / .txt / .md
        files = []
        files += glob.glob(os.path.join(self.data_dir, "*.jsonl"))
        files += glob.glob(os.path.join(self.data_dir, "*.txt"))
        files += glob.glob(os.path.join(self.data_dir, "*.md"))
        for fp in sorted(files):
            if fp.endswith(".jsonl"):
                for raw_text, src in _iter_jsonl(fp):
                    for ch in _word_chunks(raw_text, self.chunk_win, self.chunk_overlap):
                        psgs.append(Passage(text=ch, source=src))
            else:
                for raw_text, src in _iter_txt(fp):
                    for ch in _word_chunks(raw_text, self.chunk_win, self.chunk_overlap):
                        psgs.append(Passage(text=ch, source=src))
        return psgs

    def _embed_passages(self, texts: List[str]) -> np.ndarray:
        if _USE_ST:
            return self.encoder.encode(texts, batch_size=self.batch_size, show_progress_bar=True, normalize_embeddings=False)
        else:
            self.enc.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i+self.batch_size]
                    tokens = self.tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                    tokens = {k: v.to(self.enc.device) for k, v in tokens.items()}
                    model_out = self.enc(**tokens)
                    # 取 CLS 池化或 mean pooling（BGE/e5 通常建议 mean）
                    emb = model_out.last_hidden_state.mean(dim=1).cpu().numpy()
                    out.append(emb)
            return np.concatenate(out, axis=0)

    def _embed_queries(self, texts: List[str]) -> np.ndarray:
        # BGE/e5 查询可在前缀加 "query: "；简单起见这里不加
        return self._embed_passages(texts)

    def _save_meta(self):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for p in self.passages:
                f.write(json.dumps({"text": p.text, "source": p.source}, ensure_ascii=False) + "\n")

    def _load_index(self):
        self.index = faiss.read_index(self.index_path)
        self.passages = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for ln in f:
                obj = json.loads(ln)
                self.passages.append(Passage(text=obj["text"], source=obj["source"]))
