
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from datasets import load_dataset
import json

# 1. 加载 Hugging Face 数据
dataset = load_dataset("LinhDuong/chatdoctor-200k", split="train")

# 2. 转换为 LoRA 格式
with open("chatdoctor_200k_lora.jsonl", "w", encoding="utf-8") as f:
    for ex in dataset:
        f.write(json.dumps({
            "system": "You are a careful, friendly medical assistant. Provide safe and factual answers.",
            "user": f"{ex['instruction']}\n{ex['input']}",
            "assistant": ex["output"]
        }, ensure_ascii=False) + "\n")

print("Finish preparing data: data/chatdoctor_200k_lora.jsonl")
