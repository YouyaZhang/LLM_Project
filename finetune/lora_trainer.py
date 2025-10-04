# finetune/lora_trainer.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

# ---------- Prompt utils ----------
def build_system_block(prompts: dict | None) -> str:
    if not prompts:
        return (
            "You are a careful, friendly medical information assistant. "
            "Provide general health info and care-seeking guidance. "
            "No diagnoses or dosages. Structure output as: "
            "Takeaway & Risk / Self-care / When to seek care / Follow-up questions."
        )
    blocks = []
    for k in ("system_en", "guardrails_en", "style_en"):
        v = prompts.get(k, "")
        if v and v.strip():
            blocks.append(v.strip())
    return "\n\n".join(blocks)

def render_llama2_prompt(system_block: str, user: str) -> str:
    return f"<s>[INST] <<SYS>>\n{system_block}\n<</SYS>>\n\n{user.strip()} [/INST]\n"

# ---------- JSONL dataset ----------
# --- Map-style：快但占内存（原理同你之前） ---
class MapJsonlSFTDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, system_block: str, max_len: int = 768, max_samples: int = 0):
        self.tok = tokenizer
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:  # 可选子采样
                    break
                ex = json.loads(line)
                user = (ex.get("user") or "").strip()
                assistant = (ex.get("assistant") or "").strip()
                if not user or not assistant:
                    continue
                sys_blk = (ex.get("system") or "").strip() or system_block
                prompt = f"<s>[INST] <<SYS>>\n{sys_blk}\n<</SYS>>\n\n{user} [/INST]\n"
                target = assistant + tokenizer.eos_token
                enc_prompt = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                enc_target = tokenizer(target, add_special_tokens=False)["input_ids"]
                keep_prompt = max_len - len(enc_target)
                if keep_prompt < 1:
                    enc_prompt = []
                    enc_target = enc_target[-(max_len-1):]
                else:
                    enc_prompt = enc_prompt[-keep_prompt:]
                input_ids = enc_prompt + enc_target
                attn = [1] * len(input_ids)
                labels = [-100] * len(enc_prompt) + enc_target[:]
                self.samples.append((
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(attn, dtype=torch.long),
                    torch.tensor(labels, dtype=torch.long),
                ))
        if not self.samples:
            raise ValueError(f"No usable samples from {jsonl_path}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        ids, attn, labels = self.samples[idx]
        return {"input_ids": ids, "attention_mask": attn, "labels": labels}


# --- Iterable-style：省内存（每步现读现分词） ---
from torch.utils.data import IterableDataset

class StreamJsonlSFTDataset(IterableDataset):
    def __init__(self, jsonl_path: str, tokenizer, system_block: str, max_len: int = 768, max_samples: int = 0, seed: int = 42):
        self.path = jsonl_path
        self.tok = tokenizer
        self.max_len = max_len
        self.system_block = system_block
        self.max_samples = max_samples
        self.seed = seed  # 你可以用它做简单的打乱（这里保持顺序，省事稳定）

    def __iter__(self):
        cnt = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if self.max_samples and cnt >= self.max_samples:
                    break
                ex = json.loads(line)
                user = (ex.get("user") or "").strip()
                assistant = (ex.get("assistant") or "").strip()
                if not user or not assistant:
                    continue
                sys_blk = (ex.get("system") or "").strip() or self.system_block
                prompt = f"<s>[INST] <<SYS>>\n{sys_blk}\n<</SYS>>\n\n{user} [/INST]\n"
                target = assistant + self.tok.eos_token
                enc_prompt = self.tok(prompt, add_special_tokens=False)["input_ids"]
                enc_target = self.tok(target, add_special_tokens=False)["input_ids"]
                keep_prompt = self.max_len - len(enc_target)
                if keep_prompt < 1:
                    enc_prompt = []
                    enc_target = enc_target[-(self.max_len-1):]
                else:
                    enc_prompt = enc_prompt[-keep_prompt:]
                input_ids = enc_prompt + enc_target
                attn = [1] * len(input_ids)
                labels = [-100] * len(enc_prompt) + enc_target[:]
                cnt += 1
                yield {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attn, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }

def collate_batch(features: List[Dict[str, torch.Tensor]], pad_id: int):
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids, attn, labels = [], [], []
    for f in features:
        pad_len = max_len - len(f["input_ids"])
        input_ids.append(torch.cat([f["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        attn.append(torch.cat([f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        lab_pad = torch.full((pad_len,), -100, dtype=torch.long)
        labels.append(torch.cat([f["labels"], lab_pad]))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attn),
        "labels": torch.stack(labels),
    }

# ---------- LoRA training ----------
@dataclass
class LoraTrainConfig:
    model_id: str = "epfl-llm/meditron-7b"
    prompts_path: Optional[str] = "configs/prompts_en.yaml"
    train_path: str = "data/train.jsonl"
    val_path: Optional[str] = "data/val.jsonl"
    output_dir: str = "outputs/meditron7b-lora"
    use_4bit: bool = True
    max_seq_len: int = 1024  # 16GB 建议从 1024 起步

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = (
        "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"
    )

    # Train args
    epochs: float = 1.0
    batch_size: int = 1
    grad_accum: int = 16
    lr: float = 2e-4
    logging_steps: int = 25
    save_steps: int = 500
    eval_steps: int = 500
    seed: int = 42
    bf16: bool = False  # 16GB/多数显卡更稳用 fp16 计算
    max_samples: int = 0            # 0=all；可做快速子采样
    stream_dataset: bool = False     # ← 开启流式，省内存
def _load_yaml(path: str | None) -> dict:
    if not path or not os.path.exists(path):
        return {}
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def train_lora(cfg: LoraTrainConfig, hf_token: Optional[str] = None) -> str:
    """
    Run LoRA SFT training and return adapter dir path.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    prompts = _load_yaml(cfg.prompts_path)
    system_block = build_system_block(prompts)

    # tokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True, token=hf_token, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # quantization config (4-bit; fp16 compute for compatibility)
    quant = None
    if cfg.use_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,  # ← 关键：避免 bf16 带来的 CPU 分配
        )

    # base model
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        token=hf_token,
        trust_remote_code=True,
        device_map={"": 0},          # 全部放 GPU，避免 CPU/disk offload 报错
        low_cpu_mem_usage=True,
        quantization_config=quant,
        torch_dtype=torch.float16 if not cfg.use_4bit else None,
    )

    # k-bit 训练常规准备
    base = prepare_model_for_kbit_training(base)
    base.config.use_cache = False  # 与 gradient checkpointing 兼容
    base.enable_input_require_grads()

    # LoRA wrap
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.target_modules),
    )
    model = get_peft_model(base, lora)
    # datasets（根据配置选择 Map 或 Stream）
    if cfg.stream_dataset:
        train_ds = StreamJsonlSFTDataset(cfg.train_path, tok, system_block, max_len=cfg.max_seq_len, max_samples=cfg.max_samples)
        eval_ds = StreamJsonlSFTDataset(cfg.val_path, tok, system_block, max_len=cfg.max_seq_len, max_samples=min(2000, cfg.max_samples) if cfg.max_samples else 2000) \
                if cfg.val_path and os.path.exists(cfg.val_path) else None
    else:
        train_ds = MapJsonlSFTDataset(cfg.train_path, tok, system_block, max_len=cfg.max_seq_len, max_samples=cfg.max_samples)
        eval_ds = MapJsonlSFTDataset(cfg.val_path, tok, system_block, max_len=cfg.max_seq_len, max_samples=min(2000, cfg.max_samples) if cfg.max_samples else 2000) \
                if cfg.val_path and os.path.exists(cfg.val_path) else None


    # training args
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        #evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.eval_steps,
        bf16=False,                 # 用 fp16 计算以提升 16GB 兼容性
        fp16=True,
        optim="paged_adamw_8bit",   # ← 关键：更省显存
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to=[],
        seed=cfg.seed,
        gradient_checkpointing=True,  # ← 关键：显存友好
        dataloader_num_workers=0 if cfg.stream_dataset else 2,  # 流式时设 0，避免多进程复制占内存
        dataloader_pin_memory=True,
        remove_unused_columns=False,  # ← 必须，配合自定义 collator/IterableDataset
    )

    data_collator = lambda batch: collate_batch(batch, pad_id=tok.pad_token_id)

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
    )

    trainer.train()

    adapter_dir = os.path.join(cfg.output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(cfg.output_dir)
    return adapter_dir

def attach_lora_adapter(base_model, adapter_dir: str):
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    return model
