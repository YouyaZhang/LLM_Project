"""
evaluate_rag_local_v3.py
比较 Base / LoRA / LoRA+RAG 三种模型在本地医学验证集上的表现。
适配 val.jsonl 格式：{"system": "", "user": "...", "assistant": "..."}
"""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os, json, random, numpy as np, torch
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rag.rag_retriever_v1 import retrieve_context
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path

# ==================== ⚙️ 基础配置 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_DIR = r"D:\pycharm_project\pythonProject\LLM\LLM_Project_zx\finetune\adapter-20251017T224222Z-1-001\adapter"
VAL_PATH = r"D:\pycharm_project\pythonProject\LLM\LLM_Project_zx\evaluate\val_1.jsonl"
SAMPLE_SIZE = 50
REPORT_PATH = "evaluation_report_local_v3.md"

SYSTEM_PROMPT = """
You are **MedCareBot**, a reliable and calm AI medical assistant.
Answer factually, clearly, and concisely. Avoid hallucination and unsafe advice.
"""

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ==================== 🧠 模型加载 ====================
print("📥 Loading base model and tokenizer...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, device_map={"": DEVICE}
)
base_model.eval()

print("🔧 Loading LoRA adapter...")
base_for_lora = deepcopy(base_model).to("cpu")
lora_model = PeftModel.from_pretrained(base_for_lora, ADAPTER_DIR)
lora_model.to(DEVICE).eval()

# ==================== 📚 数据集加载 ====================
print(f"📚 Loading validation data: {VAL_PATH}")
data = []
with open(VAL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line.strip())
        if "user" in ex and "assistant" in ex:
            data.append({"input": ex["user"], "output": ex["assistant"]})

if len(data) > SAMPLE_SIZE:
    data = random.sample(data, SAMPLE_SIZE)
print(f"✅ Loaded {len(data)} samples for evaluation.")

# ==================== 💬 生成函数 ====================
smooth = SmoothingFunction().method1

def generate_answer(model, question, use_rag=False):
    if use_rag:
        try:
            context = retrieve_context(question)
        except Exception as e:
            print(f"[RAG WARNING] Retrieval failed: {e}")
            context = ""
        question = (
            f"{question}\n\nUse the following reference to improve factual accuracy (if helpful):\n"
            f"-----\n{context}\n-----\n"
        )
    prompt = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{question} [/INST]"
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, temperature=0.6, top_p=0.9)
    return tok.decode(output[0], skip_special_tokens=True)

# ==================== 📊 评估函数 ====================
rouge = Rouge()
bertscore = load("bertscore")
bleurt = load("bleurt", "bleurt-base-128")

def compute_bleu_rouge(preds, refs):
    bleu_scores, rouge_scores = [], []
    for p, r in zip(preds, refs):
        bleu = sentence_bleu([r.split()], p.split(), smoothing_function=smooth)
        rouge_score = rouge.get_scores(p, r)[0]["rouge-l"]["f"]
        bleu_scores.append(bleu)
        rouge_scores.append(rouge_score)
    return {"BLEU": np.mean(bleu_scores), "ROUGE-L": np.mean(rouge_scores)}

def compute_bert_bleurt(preds, refs):
    berts = bertscore.compute(predictions=preds, references=refs, lang="en")
    bleurts = bleurt.compute(predictions=preds, references=refs)
    return {"BERTScore-F1": np.mean(berts["f1"]), "BLEURT": np.mean(bleurts["scores"])}

def evaluate_set(preds, refs):
    m1 = compute_bleu_rouge(preds, refs)
    m2 = compute_bert_bleurt(preds, refs)
    return {**m1, **m2}

# ==================== 🚀 主评估流程 ====================
preds_base, preds_lora, preds_rag, refs = [], [], [], []
print("🧠 Generating answers (Base / LoRA / LoRA+RAG)...")

for ex in tqdm(data):
    q, a = ex["input"], ex["output"]
    preds_base.append(generate_answer(base_model, q))
    preds_lora.append(generate_answer(lora_model, q))
    preds_rag.append(generate_answer(lora_model, q, use_rag=True))
    refs.append(a)

# ==================== 📈 计算指标 ====================
print("📊 Computing metrics...")
metrics_base = evaluate_set(preds_base, refs)
metrics_lora = evaluate_set(preds_lora, refs)
metrics_rag = evaluate_set(preds_rag, refs)

# ==================== 🏆 排名榜 ====================
def rank_best(metric_name):
    scores = {
        "Base": metrics_base[metric_name],
        "LoRA": metrics_lora[metric_name],
        "LoRA+RAG": metrics_rag[metric_name],
    }
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return f"🏆 {sorted_scores[0][0]} ({metric_name}: {sorted_scores[0][1]:.4f})"

# ==================== 🧾 报告生成 ====================
def delta(a, b): return (b - a) * 100

results_table = f"""
| Metric | Base | LoRA | LoRA + RAG | Δ LoRA vs Base | Δ RAG vs LoRA |
|---------|------|-------|-------------|----------------|---------------|
| BLEU | {metrics_base['BLEU']:.4f} | {metrics_lora['BLEU']:.4f} | **{metrics_rag['BLEU']:.4f}** | {delta(metrics_base['BLEU'], metrics_lora['BLEU']):+.2f}% | {delta(metrics_lora['BLEU'], metrics_rag['BLEU']):+.2f}% |
| ROUGE-L | {metrics_base['ROUGE-L']:.4f} | {metrics_lora['ROUGE-L']:.4f} | **{metrics_rag['ROUGE-L']:.4f}** | {delta(metrics_base['ROUGE-L'], metrics_lora['ROUGE-L']):+.2f}% | {delta(metrics_lora['ROUGE-L'], metrics_rag['ROUGE-L']):+.2f}% |
| BERTScore-F1 | {metrics_base['BERTScore-F1']:.4f} | {metrics_lora['BERTScore-F1']:.4f} | **{metrics_rag['BERTScore-F1']:.4f}** | {delta(metrics_base['BERTScore-F1'], metrics_lora['BERTScore-F1']):+.2f}% | {delta(metrics_lora['BERTScore-F1'], metrics_rag['BERTScore-F1']):+.2f}% |
| BLEURT | {metrics_base['BLEURT']:.4f} | {metrics_lora['BLEURT']:.4f} | **{metrics_rag['BLEURT']:.4f}** | {delta(metrics_base['BLEURT'], metrics_lora['BLEURT']):+.2f}% | {delta(metrics_lora['BLEURT'], metrics_rag['BLEURT']):+.2f}% |
"""

summary_rank = "\n".join([
    rank_best("BLEU"),
    rank_best("ROUGE-L"),
    rank_best("BERTScore-F1"),
    rank_best("BLEURT"),
])

# 写入报告
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("# 🩺 MedCareBot Evaluation Report (Base vs LoRA vs LoRA+RAG)\n\n")
    f.write("## 🔬 Metric Comparison\n")
    f.write(results_table)
    f.write("\n## 🏆 Best Performing Models by Metric\n")
    f.write(summary_rank + "\n")
    f.write("\n## 💬 Example Outputs\n")
    for i in range(min(3, len(data))):
        f.write(f"### Example {i+1}\n")
        f.write(f"**User:** {data[i]['input']}\n\n")
        f.write(f"**Reference:** {refs[i]}\n\n")
        f.write(f"**Base:** {preds_base[i][:600]}...\n\n")
        f.write(f"**LoRA:** {preds_lora[i][:600]}...\n\n")
        f.write(f"**LoRA+RAG:** {preds_rag[i][:600]}...\n\n")
    f.write("\n✅ Report generated successfully.\n")

print("\n📊 Evaluation Summary:")
print(results_table)
print("\n🏆 Rankings:\n" + summary_rank)
print(f"\n✅ Evaluation report saved to: {REPORT_PATH}")

# ==================== 📉 可视化 ====================
labels = ["Base", "LoRA", "LoRA+RAG"]
bert_f1 = [
    metrics_base["BERTScore-F1"],
    metrics_lora["BERTScore-F1"],
    metrics_rag["BERTScore-F1"]
]
plt.bar(labels, bert_f1, color=["#8da0cb", "#66c2a5", "#fc8d62"])
plt.title("BERTScore-F1 Comparison (Local Validation)")
plt.ylabel("Score")
plt.savefig("bert_f1_comparison_v3.png")
print("📈 Saved chart: bert_f1_comparison_v3.png")
