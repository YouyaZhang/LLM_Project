# models/model_loader.py (replace the load function with this)
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclass
class ModelRuntime:
    tokenizer: Any
    model: Any

def load_model_from_config(cfg_path: str = "configs/model.yaml") -> ModelRuntime:
    # 你自己已有的 YAML 读取函数
    from utils.config import load_yaml
    cfg = load_yaml(cfg_path)

    model_id = cfg.get("model_id", "epfl-llm/meditron-7b")
    trust_remote_code = bool(cfg.get("trust_remote_code", True))
    device_map = cfg.get("device_map", "auto")

    # 优先 bf16，环境不支持再回落到 fp16
    want_dtype = str(cfg.get("dtype", "bfloat16")).lower()
    if want_dtype == "auto":
        torch_dtype = None
    elif want_dtype in ("bfloat16", "bf16"):
        torch_dtype = torch.bfloat16
    elif want_dtype in ("float16", "fp16"):
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # 必要：Hugging Face token（meditron-7b 是 gated）
    hf_token = os.environ.get("HF_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        token=hf_token,
        use_fast=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,   # None = auto
        device_map=device_map,
        token=hf_token,
    )

    # 对 Llama-系：pad_token_id 设为 eos，防止 generate/padding 报错
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.generation_config, "pad_token_id", None) is None:
        model.generation_config.pad_token_id = tokenizer.eos_token_id
    if getattr(model.generation_config, "eos_token_id", None) is None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    # 可选：开启 BetterTransformer/FA2（按你环境与 configs 决定）
    if bool(cfg.get("use_flash_attention_2", False)):
        try:
            model = model.to_bettertransformer()
        except Exception:
            pass

    return ModelRuntime(tokenizer=tokenizer, model=model)
