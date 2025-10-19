# finetune/train_lora.py



from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os, json, torch, math

from dataclasses import dataclass
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model
from typing import Dict


# ==========================================================

# ==========================================================
@dataclass

class LoRAConfig:
    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    train_path: str = r"data\train.jsonl"
    val_path: str = r"data\val.jsonl"
    output_dir: str = "adapter"
    use_4bit: bool = False   # 关闭4bit量化
    epochs: int = 2
    batch_size: int = 1
    grad_accum: int = 16
    lr: float = 2e-4
    max_seq_len: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")



# ==========================================================
#  Dataset 
# ==========================================================
class LoRADataset(torch.utils.data.Dataset):
    def __init__(self, path: str, tok, max_len: int = 1024):
        self.tok = tok
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                sys = ex.get("system", "").strip()
                user = ex.get("user", "").strip()
                asst = ex.get("assistant", "").strip()

                if not user or not asst:
                    continue

                # 格式化对话模板
                text = f"<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{user} [/INST] {asst}</s>"
                enc = tok(text, add_special_tokens=False, truncation=True, max_length=max_len)
                ids = enc["input_ids"]

                self.data.append({
                    "input_ids": torch.tensor(ids),
                    "labels": torch.tensor(ids),
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]



def compute_metrics(eval_pred):
    loss = eval_pred.metrics["eval_loss"]
    ppl = math.exp(loss) if loss < 10 else float("inf")
    return {"perplexity": round(ppl, 3)}



def main():
    cfg = LoRAConfig()

    print("📦 Loading tokenizer & base model...")
    tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    quant = None

    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        # quantization_config=quant
    )

    # print(" Preparing model for LoRA training...")
    # base = prepare_model_for_kbit_training(base)

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        task_type="CAUSAL_LM",
        target_modules=list(cfg.target_modules),
    )
    model = get_peft_model(base, lora)

    print("📚 Loading datasets...")
    train_ds = LoRADataset(cfg.train_path, tok, cfg.max_seq_len)
    val_ds = LoRADataset(cfg.val_path, tok, cfg.max_seq_len)

    print(f"🧠 Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        fp16=False,  # 或 bf16=True（
        bf16=True,
        optim="adamw_torch",  
        logging_steps=50,
        save_strategy="epoch",
        report_to=[],  # 关闭 wandb
        do_eval=True
      
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
    )
    print("🚀 Starting LoRA fine-tuning...")
    trainer.train()
    trainer.evaluate()  



    print("\n✅ Saving LoRA adapter...")
    model.save_pretrained(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)

    print(f"✅ LoRA adapter saved at: {cfg.output_dir}")


if __name__ == "__main__":
    main()
