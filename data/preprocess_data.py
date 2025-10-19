# prepare_dataset.py
# Purpose: Download & prepare ChatDoctor-HealthCareMagic-100k (or similar) for LoRA.
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
    s = re.sub(r"\u200b|\u200e|\u200f|\ufeff", "", s)  # remove zero-width characters
    s = re.sub(r"[ \t]+", " ", s)                      # collapse consecutive spaces
    s = re.sub(r"\n{3,}", "\n\n", s)                   # collapse multiple blank lines

    # ---- Remove branding / polite phrases (Chat Doctor etc.) ----
    patterns = [
        # 1) Signature endings
        r"(?:Best wishes|Kind regards|Warm regards|Regards|Sincerely|Take care|Stay safe|Be well)[\s,]*"
        r"(?:,|\.)?\s*(?:Dr\.?\s*)?(?:Chat\s*Doctor|ChatDoctor)(?:\s*\.?\s*com)?\b[^\n]*",

        # 2) "Thank you for choosing/using Chat Doctor..."
        r"(?:Thank you|Thanks)[^.\n]*?(?:using|choosing|contacting|visiting)\s+(?:the\s+)?"
        r"(?:Chat\s*Doctor|ChatDoctor)(?:\s*\.?\s*com)?\b[^.\n]*\.?",

        # 3) Greetings / introductions / brand phrases
        r"Welcome\s+to\s+(?:the\s+)?(?:Chat\s*Doctor|ChatDoctor)(?:\s*\.?\s*com)?\b[^\n]*",
        r"I\s+am\s+(?:Dr\.?\s*)?(?:Chat\s*Doctor|ChatDoctor)\b[^\n]*",

        # 4) Standalone brand lines, e.g., “— Chat Doctor”, “- Chat Doctor”
        r"[–—-]\s*(?:Chat\s*Doctor|ChatDoctor)(?:\s*\.?\s*com)?\b[^\n]*",
        r"^\s*(?:Dr\.?\s*)?(?:Chat\s*Doctor|ChatDoctor)(?:\s*\.?\s*com)?\s*[.,!:-]*\s*$",

        # 5) Greeting lines (remove if you want to keep greetings)
        r"^\s*Dear\s+[A-Z][a-z]+[^\n]*$",
    ]
    for pat in patterns:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE | re.MULTILINE)

    # 6) Remove “preposition + Chat Doctor(.com)” phrases to avoid leftover “on .”
    s = re.sub(
        r"\b(?:on|at|via|from|using|with|through|over|in)\s+(?:the\s+)?"
        r"(?:Chat\s*Doctor|ChatDoctor)(?:\s*\.?\s*com)?\b\.?",
        " ", s, flags=re.IGNORECASE)

    # 7) Remove bare brand terms (including “Dr. Chat Doctor”, “ChatDoctor”, “Chat Doctor .com”)
    s = re.sub(
        r"\b(?:Dr\.?\s*)?(?:Chat\s*Doctor|ChatDoctor)(?:\s*\.?\s*com)?\b",
        " ", s, flags=re.IGNORECASE)

    # 8) Clean up stray prepositions before punctuation after brand removal
    s = re.sub(
        r"\b(?:on|at|via|from|using|with|through|over|in)(?:\s+the)?\s*(?=[.,;:!?])",
        "", s, flags=re.IGNORECASE)

    # ---- Final cleanup: spaces, punctuation, and blank lines ----
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)  # remove spaces before punctuation
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\n\s+\n", "\n\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip(" .,!?:;-–—\n\t ")

    return s.strip()

def record_to_triplet(ex: Dict, map_style: str = "merge") -> Tuple[str, str, str]:
    """
    Map dataset fields to (system, user, assistant).

    Source fields (typical for lavita/ChatDoctor-HealthCareMagic-100k):
      - instruction: the task/prompt
      - input:       optional extra context
      - output:      assistant answer

    map_style:
      - "merge":   system="", user="### Instruction:\\n{instruction}\\n\\n### Input:\\n{input}" (if input exists), assistant=output
      - "sys_inst": system=instruction, user=input (may be empty), assistant=output
    """
    instruction = norm_text(ex.get("instruction") or ex.get("question") or "")
    inctx = norm_text(ex.get("input") or "")
    assistant = norm_text(ex.get("output") or ex.get("answer") or "")

    if map_style == "sys_inst":
        system = instruction
        user = inctx if inctx else ""
    else:  # "merge"
        system = ""
        if inctx:
            user = f"### Instruction:\n{instruction}\n\n### Input:\n{inctx}"
        else:
            user = instruction

    return system, user, assistant

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
    # Optionally remove non-English or invalid samples
    if re.search(r"[\u4e00-\u9fff]", user + assistant):  # contains CJK characters
        return False
    return True

def write_jsonl(path: str, triplets: Iterable[Tuple[str, str, str]]) -> int:
    n = 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for system, user, assistant in triplets:
            obj = {"system": system, "user": user, "assistant": assistant}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser(description="Prepare ChatDoctor-HealthCareMagic-100k for LoRA SFT")
    ap.add_argument("--out_dir", default="data", help="Output directory")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio (0~1)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling/splitting")
    ap.add_argument("--max_samples", type=int, default=0, help="Cap total samples (0 = all)")
    ap.add_argument("--min_chars", type=int, default=100, help="Drop very short pairs")
    ap.add_argument("--max_chars", type=int, default=4000, help="Drop overly long pairs (0 = no limit)")
    ap.add_argument("--dataset", default="lavita/ChatDoctor-HealthCareMagic-100k",
                    help="HF dataset repo id")
    ap.add_argument("--map_style", choices=["merge", "sys_inst"], default="merge",
                    help="How to map instruction/input/output to system/user/assistant")
    args = ap.parse_args()

    print(f"[1/5] Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="train")  # parquet dataset, no scripts needed
    dsd = DatasetDict({"full": ds})

    print("[2/5] Shuffling & optional sub-sampling")
    dsd["full"] = dsd["full"].shuffle(seed=args.seed)
    if args.max_samples and args.max_samples > 0:
        dsd["full"] = dsd["full"].select(range(min(args.max_samples, len(dsd["full"]))))

    print("[3/5] Mapping fields & basic cleaning")
    def mapper(example):
        system, user, assistant = record_to_triplet(example, map_style=args.map_style)
        return {
            "system": system,
            "user": user,
            "assistant": assistant,
            "keep": keep_example(user, assistant, args.min_chars, args.max_chars),
            "hash": hash_pair(user, assistant),
        }

    mapped = dsd["full"].map(mapper, remove_columns=dsd["full"].column_names, desc="Formatting")

    print("[4/5] Filtering empties/short/long & deduplicating")
    filtered = mapped.filter(lambda x: x["keep"], desc="Filter len/empty")

    # Deduplicate by (user, assistant) hash
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

    train_triplets = [(ex["system"], ex["user"], ex["assistant"]) for ex in split["train"]]
    val_triplets   = [(ex["system"], ex["user"], ex["assistant"]) for ex in split["test"]]

    out_train = os.path.join(args.out_dir, "train.jsonl")
    out_val   = os.path.join(args.out_dir, "val.jsonl")
    n_tr = write_jsonl(out_train, train_triplets)
    n_va = write_jsonl(out_val,   val_triplets)

    print(f"Done. Wrote {n_tr} train and {n_va} val samples.")
    print(f"Train: {out_train}")
    print(f"Val:   {out_val}")

if __name__ == "__main__":
    main()
