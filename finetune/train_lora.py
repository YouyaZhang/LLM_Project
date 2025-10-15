# finetune/train_lora.py
# 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
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json, torch
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

@dataclass
class LoRAConfig:
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    train_path: str = r"D:\pycharm_project\pythonProject\LLM\LLM_Project_zx\data\chatdoctor_200k_lora.jsonl"
    output_dir: str = "adapter"
    use_4bit: bool = True
    epochs: int = 1
    batch_size: int = 1
    grad_accum: int = 16
    lr: float = 2e-4
    max_seq_len: int = 512
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: tuple = ("q_proj", "v_proj")
# @dataclass
# class LoRAConfig:
#     model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
#     train_path: str = "data/chatdoctor_200k_lora.jsonl"
#     output_dir: str = "finetune/adapter"
#     use_4bit: bool = True
#     epochs: int = 2
#     batch_size: int = 1
#     grad_accum: int = 16
#     lr: float = 2e-4
#     max_seq_len: int = 1024
#     lora_r: int = 16
#     lora_alpha: int = 32
#     lora_dropout: float = 0.05
#     target_modules: tuple = ("q_proj","v_proj")

class LoRADataset(torch.utils.data.Dataset):
    def __init__(self, path, tok, max_len=1024):
        self.tok, self.data = tok, []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                sys = ex.get("system", "").strip()
                user = ex.get("user", "").strip()
                asst = ex.get("assistant", "").strip()

                # 跳过无效样本
                if not user or not asst:
                    continue

                # 🔧 手动构建对话格式
                text = f"<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{user} [/INST] {asst}</s>"

                # 分词
                enc = tok(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_len,
                )
                ids = enc["input_ids"]
                self.data.append({
                    "input_ids": torch.tensor(ids),
                    "labels": torch.tensor(ids),
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def main():
    cfg = LoRAConfig()
    tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if cfg.use_4bit else None
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.float16, device_map="auto", quantization_config=quant)
    base = prepare_model_for_kbit_training(base)
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, task_type="CAUSAL_LM", target_modules=list(cfg.target_modules))
    model = get_peft_model(base, lora)
    train_ds = LoRADataset(cfg.train_path, tok, cfg.max_seq_len)
    args = TrainingArguments(output_dir=cfg.output_dir, per_device_train_batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.grad_accum, num_train_epochs=cfg.epochs, learning_rate=cfg.lr, fp16=True, optim="paged_adamw_8bit", logging_steps=50, save_strategy="epoch", report_to=[])
    Trainer(model=model, args=args, train_dataset=train_ds, tokenizer=tok).train()
    model.save_pretrained(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    print(f" LoRA adapter saved at {cfg.output_dir}")

if __name__ == "__main__":
    main()
