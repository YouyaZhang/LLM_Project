# prepare_dataset.py
# Purpose: Download & prepare ChatDoctor-HealthCareMagic-100k for SFT/LoRA.
# Output: data/train.jsonl and data/val.jsonl with fields: system, user, assistant

from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from typing import Dict, Iterable, Tuple

from datasets import load_dataset, DatasetDict

def norm_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r", "\n")
    s = re.sub(r"\u200b|\u200e|\u200f|\ufeff", "", s)  # zero-width chars
    s = re.sub(r"[ \t]+", " ", s)                      # collapse spaces
    s = re.sub(r"\n{3,}", "\n\n", s)                   # collapse blank lines
    return s.strip()

def record_to_pair(ex: Dict) -> Tuple[str, str]:
    """
    Map dataset fields to (user, assistant).
    Common fields in lavita/ChatDoctor-HealthCareMagic-100k:
      - instruction: user question
      - output:      assistant answer
    """
    user = ex.get("instruction") or ex.get("question") or ""
    assistant = ex.get("output") or ex.get("answer") or ""
    return norm_text(user), norm_text(assistant)

def hash_pair(user: str, assistant: str) -> str:
    h = hashlib.sha256()
    h.update(user.encode("utf-8"))
    h.update(b"\x00")
    h.update(assistant.encode("utf-8"))
    return h.hexdigest()

def keep_example(user: str, assistant: str, min_chars: int, max_chars: int) -> bool:
    if not user or not assistant:
        return False
    total_len = len(user) + len(assistant)
    if total_len < min_chars:
        return False
    if max_chars > 0 and total_len > max_chars:
        return False
    # optional: drop obvious non-English or garbage lines
    if re.search(r"[\u4e00-\u9fff]", user + assistant):  # contains CJK
        return False
    return True

def write_jsonl(path: str, pairs: Iterable[Tuple[str, str]]) -> int:
    n = 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for user, assistant in pairs:
            obj = {"system": "", "user": user, "assistant": assistant}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser(description="Prepare ChatDoctor-HealthCareMagic-100k for LoRA SFT")
    ap.add_argument("--out_dir", default="data", help="Output directory")
    ap.add_argument("--val_ratio", type=float, default=0.02, help="Validation split ratio (0~1)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling/splitting")
    ap.add_argument("--max_samples", type=int, default=0, help="Cap total samples (0 = all)")
    ap.add_argument("--min_chars", type=int, default=80, help="Drop very short pairs")
    ap.add_argument("--max_chars", type=int, default=4000, help="Drop overly long pairs (0 = no limit)")
    ap.add_argument("--dataset", default="lavita/ChatDoctor-HealthCareMagic-100k",
                    help="HF dataset repo id")
    args = ap.parse_args()

    print(f"[1/5] Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="train")  # parquet dataset, no scripts needed
    # ds is a Dataset; wrap in DatasetDict to use train_test_split easily
    dsd = DatasetDict({"full": ds})

    print("[2/5] Shuffling & optional sub-sampling")
    dsd["full"] = dsd["full"].shuffle(seed=args.seed)
    if args.max_samples and args.max_samples > 0:
        dsd["full"] = dsd["full"].select(range(min(args.max_samples, len(dsd["full"]))))

    print("[3/5] Mapping fields & basic cleaning")
    def mapper(example):
        user, assistant = record_to_pair(example)
        return {"user": user, "assistant": assistant,
                "keep": keep_example(user, assistant, args.min_chars, args.max_chars),
                "hash": hash_pair(user, assistant)}
    mapped = dsd["full"].map(mapper, remove_columns=dsd["full"].column_names, desc="Formatting")

    print("[4/5] Filtering empties/short/long & deduplicating")
    filtered = mapped.filter(lambda x: x["keep"], desc="Filter len/empty")
    # Dedup by hash
    seen = set()
    def not_seen(x):
        h = x["hash"]
        if h in seen:
            return False
        seen.add(h)
        return True
    filtered = filtered.filter(not_seen, desc="Deduplicate")

    print("[5/5] Train/val split & write JSONL")
    split = filtered.train_test_split(test_size=args.val_ratio, seed=args.seed)
    train_pairs = [(ex["user"], ex["assistant"]) for ex in split["train"]]
    val_pairs   = [(ex["user"], ex["assistant"]) for ex in split["test"]]

    out_train = os.path.join(args.out_dir, "train.jsonl")
    out_val   = os.path.join(args.out_dir, "val.jsonl")
    n_tr = write_jsonl(out_train, train_pairs)
    n_va = write_jsonl(out_val,   val_pairs)

    print(f"Done. Wrote {n_tr} train and {n_va} val samples.")
    print(f"Train: {out_train}")
    print(f"Val:   {out_val}")

if __name__ == "__main__":
    main()
