# demo/app.py
from __future__ import annotations
import os
import threading
from typing import List, Dict

import gradio as gr
import torch
from transformers import TextIteratorStreamer
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.model_loader import load_model_from_config
from utils.config import load_yaml
from inference.pipeline import (
    InferencePipeline,
    _has_chat_template,
    _render_with_template,
    _render_llama2_fallback,
)
from finetune.lora_trainer import (
    LoraTrainConfig,
    train_lora,
    attach_lora_adapter,
)

CFG_PATH = "configs/model.yaml"

# ========= Load configs & prompts =========
cfg = load_yaml(CFG_PATH)
prompts = load_yaml(cfg.get("prompts_file", "configs/prompts_en.yaml"))

# ========= Load base model/tokenizer & pipeline =========
runtime = load_model_from_config(CFG_PATH)
tokenizer = runtime.tokenizer
base_model = runtime.model

# 当前用于推理的模型（可切换为 LoRA 版本）
CURRENT_MODEL = base_model

pipe = InferencePipeline(
    tokenizer=tokenizer,
    model=CURRENT_MODEL,        # 初始化时引用，推理时我们用全局 CURRENT_MODEL
    cfg=cfg,
    prompts=prompts,
)

# ========= Chat helpers =========
def _history_pairs_to_msgs(history_pairs: List[List[str]]) -> List[Dict[str, str]]:
    """[[user, assistant], ...] -> [{'role': 'user'|'assistant', 'content': '...'}, ...]"""
    msgs: List[Dict[str, str]] = []
    for u, a in history_pairs:
        if u:
            msgs.append({"role": "user", "content": u})
        if a:
            msgs.append({"role": "assistant", "content": a})
    return msgs

def _build_prompt_for_stream(user_message: str, history_pairs: List[List[str]]) -> str:
    """Use same rendering path as pipeline (HF template if present, else Llama-2 fallback)."""
    # 复用 pipeline 的消息构造（系统+guardrails+style+fewshots+历史+当前用户）
    msgs = pipe._format_messages(user_message=user_message, history=_history_pairs_to_msgs(history_pairs))
    if _has_chat_template(tokenizer):
        try:
            return _render_with_template(tokenizer, msgs)
        except Exception:
            return _render_llama2_fallback(msgs)
    else:
        return _render_llama2_fallback(msgs)

# ========= Streaming chat =========
def stream_reply(user_text: str, chat_history: List[List[str]]):
    """
    Generator for streaming: yields updated chat_history as the assistant types.
    """
    global CURRENT_MODEL
    # 1) Append user turn
    chat_history = chat_history + [[user_text, ""]]
    yield chat_history

    # 2) Build prompt & tokenize
    prompt_text = _build_prompt_for_stream(user_text, chat_history[:-1])
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(CURRENT_MODEL.device) for k, v in inputs.items()}

    # 3) Text streamer
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    # 4) Generation config（与 pipeline 保持一致）
    gen_cfg = {
        "do_sample": bool(cfg.get("do_sample", True)),
        "max_new_tokens": int(cfg.get("max_new_tokens", 512)),
        "temperature": float(cfg.get("temperature", 0.5)),
        "top_p": float(cfg.get("top_p", 0.9)),
        "repetition_penalty": float(cfg.get("repetition_penalty", 1.05)),
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    # 5) Background generation thread
    def _worker():
        with torch.inference_mode():
            CURRENT_MODEL.generate(**inputs, **gen_cfg)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    # 6) Yield partial text
    partial = ""
    for new_text in streamer:
        partial += new_text
        chat_history[-1][1] = partial.strip()
        yield chat_history

# ========= LoRA panel callbacks =========
def on_train_lora(out_dir, train_path, val_path, use_4bit,
                  lora_r, lora_alpha, lora_dropout,
                  epochs, batch, grad_accum, lr, max_len,
                  stream_ds, max_samples):
    try:
        status = "Starting LoRA training..."
        yield status
        cfg_train = LoraTrainConfig(
            model_id="epfl-llm/meditron-7b",
            prompts_path="configs/prompts_en.yaml",
            train_path=train_path,
            val_path=val_path if val_path and os.path.exists(val_path) else None,
            output_dir=out_dir,
            use_4bit=bool(use_4bit),
            max_seq_len=int(max_len),
            lora_r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            epochs=float(epochs),
            batch_size=int(batch),
            grad_accum=int(grad_accum),
            lr=float(eval(lr)),
            stream_dataset=bool(stream_ds),
            max_samples=int(max_samples or 0),
        )
        adapter_dir = train_lora(cfg_train, hf_token=os.environ.get("HF_TOKEN"))
        status += f"\nDone. Adapter saved to: {adapter_dir}"
        yield status
    except Exception as e:
        yield f"Error: {repr(e)}"

def on_load_adapter(adapter_dir: str):
    """
    把 LoRA 适配器加载到当前模型；不重启服务。
    """
    global CURRENT_MODEL
    try:
        CURRENT_MODEL = attach_lora_adapter(base_model, adapter_dir)
        return "Adapter loaded successfully."
    except Exception as e:
        return f"Load failed: {repr(e)}"

def on_unload_adapter():
    """
    切回基础模型。
    """
    global CURRENT_MODEL
    CURRENT_MODEL = base_model
    return "Returned to base model."

# ========= UI =========
def build_ui():
    title = "🩺 Meditron-7B Medical Info Assistant (English)"
    desc = (
        "Educational use only. Provides general health information and care-seeking guidance. "
        "Not a diagnosis or prescription. For emergencies or red-flag symptoms, seek urgent in-person care."
    )
    examples = [
        "Headache for two days, worse in the morning, mild nausea, no fever.",
        "Sore throat and mild fever for 24 hours, no cough, I can swallow liquids.",
        "Mild chest tightness after exercise, no known heart disease. What should I consider?",
    ]

    with gr.Blocks(css="""
        .gradio-container { max-width: 900px !important; }
    """) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(desc)

        # Chat section
        chatbot = gr.Chatbot(
            height=520,
            show_copy_button=True,
            placeholder="Describe your symptoms in English. The assistant replies with structured guidance."
        )
        with gr.Row():
            msg = gr.Textbox(
                label="Your message",
                placeholder="Type here and press Enter…",
                scale=8,
                lines=2,
                autofocus=True,
            )
        with gr.Row():
            send_btn = gr.Button("Send ➤", variant="primary")
            clear_btn = gr.Button("🗑️ Clear")
        with gr.Row():
            gr.Markdown("**Try an example:**")
        with gr.Row():
            ex_btns = [gr.Button(e, size="sm") for e in examples]

        # State for chat history
        state = gr.State([])

        # Submit/Send bindings
        def on_submit(user_text, history):
            if not user_text or not user_text.strip():
                return gr.update(), history
            return gr.update(value=""), history  # clear textbox, keep state

        msg.submit(on_submit, [msg, state], [msg, state]).then(
            stream_reply, [msg, state], [chatbot],
        ).then(lambda h: h, [chatbot], [state])

        send_btn.click(on_submit, [msg, state], [msg, state]).then(
            stream_reply, [msg, state], [chatbot],
        ).then(lambda h: h, [chatbot], [state])

        # Clear
        def on_clear():
            return [], []
        clear_btn.click(on_clear, outputs=[chatbot, state], queue=False)

        # Examples fill
        for b in ex_btns:
            b.click(lambda t=b.value: t, outputs=msg)

        gr.Markdown("---")

        # LoRA fine-tuning panel
        with gr.Accordion("🔧 LoRA fine-tuning (advanced)", open=False):
            with gr.Row():
                out_dir = gr.Textbox(label="Output dir", value="outputs/meditron7b-lora", scale=2)
                train_path = gr.Textbox(label="Train JSONL", value="data/train.jsonl", scale=2)
                val_path = gr.Textbox(label="Val JSONL (optional)", value="data/val.jsonl", scale=2)
            with gr.Row():
                use_4bit = gr.Checkbox(label="Use 4-bit (QLoRA)", value=True)
                lora_r = gr.Slider(4, 64, value=16, step=1, label="LoRA r")
                lora_alpha = gr.Slider(8, 128, value=32, step=1, label="LoRA alpha")
                lora_dropout = gr.Slider(0.0, 0.2, value=0.05, step=0.01, label="LoRA dropout")
            with gr.Row():
                epochs = gr.Slider(0.5, 5.0, value=2.0, step=0.5, label="Epochs")
                batch = gr.Slider(1, 4, value=1, step=1, label="Batch size / device")
                grad_accum = gr.Slider(1, 64, value=16, step=1, label="Grad accumulation")
                lr = gr.Textbox(label="Learning rate", value="2e-4")
                max_len = gr.Slider(512, 4096, value=2048, step=128, label="Max seq length")
            with gr.Row():
                train_btn = gr.Button("🚀 Train LoRA", variant="primary")
                status_box = gr.Textbox(label="Status", value="", lines=6)
            with gr.Row():
                stream_ds = gr.Checkbox(label="Stream dataset (low RAM)", value=True)
                max_samples = gr.Number(label="Max samples (0=all)", value=0, precision=0)

            gr.Markdown("**Load/Unload adapter**")
            with gr.Row():
                adapter_dir_in = gr.Textbox(label="Adapter dir", value="outputs/meditron7b-lora/adapter", scale=2)
                load_btn = gr.Button("📦 Load LoRA Adapter")
                unload_btn = gr.Button("♻️ Reload Base Model")
            load_status = gr.Textbox(label="Adapter status", value="", lines=2)

            # Bind train
            train_btn.click(
                on_train_lora,
                inputs=[out_dir, train_path, val_path, use_4bit, lora_r, lora_alpha, lora_dropout,
                        epochs, batch, grad_accum, lr, max_len, stream_ds, max_samples],
                outputs=[status_box],
            )

            # Bind load/unload
            load_btn.click(on_load_adapter, inputs=[adapter_dir_in], outputs=[load_status])
            unload_btn.click(on_unload_adapter, outputs=[load_status])

        gr.Markdown(
            "— *This assistant follows strict safety guardrails: no diagnoses or dosages; "
            "seek in-person care for red-flag symptoms.*"
        )

    return demo

if __name__ == "__main__":
    ui = build_ui()
    # queue() enables concurrency and smooth streaming in Gradio
    ui.queue(max_size=64)
    ui.launch(server_name="172.18.161.14", server_port=7860)