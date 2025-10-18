# demo/app.py  (fixed)
from __future__ import annotations
import os
import threading
from typing import List, Dict, Optional, Any

import gradio as gr
import torch
from transformers import TextIteratorStreamer
import sys
from pathlib import Path

# make root importable
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.model_loader import load_model_from_config
from utils.config import load_yaml
from inference.pipeline import (
    InferencePipeline,
    _has_chat_template,
    _render_with_template,
    _render_llama2_fallback,
)
from retrieval.retriever import DenseRetriever  

CFG_PATH = "configs/model.yaml"

# ========= Load configs & prompts =========
cfg = load_yaml(CFG_PATH)
prompts = load_yaml(cfg.get("prompts_file", "configs/prompts_en.yaml"))

# ========= Load base model/tokenizer & pipeline =========
runtime = load_model_from_config(CFG_PATH)
tokenizer = runtime.tokenizer
base_model = runtime.model

# hold current model (you can later swap it for adapters)
CURRENT_MODEL = base_model
pipe = InferencePipeline(tokenizer=tokenizer, model=base_model, cfg=cfg, prompts=prompts)

# Retriever for RAG (simple file-backed retriever)
retriever = DenseRetriever(data_dir="data")

# ========= Utils =========
def _model_device(m) -> torch.device:
    """Robustly get model device (some HF models lack .device)."""
    return getattr(m, "device", next(m.parameters()).device)

def _history_pairs_to_msgs(history_pairs: List[List[str]]) -> List[Dict[str, str]]:
    """[[user, assistant], ...] -> [{'role': 'user'|'assistant', 'content': '...'}, ...]"""
    msgs: List[Dict[str, str]] = []
    for u, a in history_pairs:
        if u:
            msgs.append({"role": "user", "content": u})
        if a:
            msgs.append({"role": "assistant", "content": a})
    return msgs

def _compose_extra_system(citations: list | None, chunks: list | None) -> Optional[str]:
    """
    Build an extra system block containing retrieved texts (not仅仅是source名)以便模型真正看到内容。
    - Prefer chunks (text bodies); fallback to citations['text'].
    """
    blocks: List[str] = []
    if chunks:
        for i, ch in enumerate(chunks):
            text = ch.get("text") if isinstance(ch, dict) else str(ch)
            if text:
                blocks.append(f"[{i+1}] {text}")
    elif citations:
        for i, c in enumerate(citations):
            text = (c.get("text") or "").strip()
            src = c.get("source", "")
            if text:
                suff = f" (Source: {src})" if src else ""
                blocks.append(f"[{i+1}] {text}{suff}")
    if blocks:
        return "Retrieved passages:\n" + "\n\n".join(blocks)
    else:
        # warning: no retrieved texts
        print("[RAG WARN] No retrieved texts to compose extra system block.")
        return None

def _build_prompt_for_stream(
    user_message: str,
    history_pairs: List[List[str]],
    extra_system: str | None = None,
    tokenizer=None,
    gen_cfg: dict | None = None,
    model_max_input_tokens: int = 4096,
) -> str:
    """
    目标：
    1) 把 RAG 文本塞进用户轮，而不是第二条 system（兼容不吃多 system 的模板）；
    2) 智能截断：优先丢最老历史，保留 [CONTEXT]+本轮问题；实在超长再截断 RAG 文本。

    返回：渲染后的 prompt 文本
    """
    assert tokenizer is not None, "tokenizer is required for truncation"

    # --- 0) 生成时预留空间，避免把输入截断到模型吐不出字 ---
    max_new = int((gen_cfg or {}).get("max_new_tokens", 512))
    SAFETY_MARGIN = 32                   # 给特殊符号留点空间
    INPUT_BUDGET = max(256, model_max_input_tokens - max_new - SAFETY_MARGIN)

    if extra_system and extra_system.strip():
        user_with_ctx = (
            f"[CONTEXT]\n{extra_system.strip()}\n\n"
            f"[QUESTION]\n{user_message.strip()}\n\n"
            f"Instructions: The CONTEXT above may be useful background information. "
            f"Use it as a reference when relevant, but you can also rely on your own knowledge "
            f"if the context is insufficient or unrelated. "
            f"Prefer citing snippet numbers like [1], [2] when you use the context."
        )

    else:
        user_with_ctx = user_message.strip()

    base_system = pipe.DEFAULT_SYSTEM if hasattr(pipe, "DEFAULT_SYSTEM") else "You are a helpful assistant."
    msgs = [{"role": "system", "content": base_system}]
    for u, a in history_pairs:
        if u is None and a is None:
            continue
        if u is not None:
            msgs.append({"role": "user", "content": str(u)})
        if a is not None:
            msgs.append({"role": "assistant", "content": str(a)})
    msgs.append({"role": "user", "content": user_with_ctx})  # ✅ 本轮用户，含 RAG

    def _render(mm):
        if _has_chat_template(tokenizer):
            try:
                return _render_with_template(tokenizer, mm)
            except Exception:
                return _render_llama2_fallback(mm)
        else:
            return _render_llama2_fallback(mm)

    prompt_text = _render(msgs)
    input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    while input_ids.size(0) > INPUT_BUDGET and len(msgs) > 2:
        if len(msgs) > 2:
            del msgs[1]
            if len(msgs) > 2 and msgs[1]["role"] == "assistant":
                del msgs[1]
        prompt_text = _render(msgs)
        input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    if input_ids.size(0) > INPUT_BUDGET:

        if msgs and msgs[-1]["role"] == "user":
            ctx_user = msgs[-1]["content"]
            if "[CONTEXT]" in ctx_user and "[QUESTION]" in ctx_user:
                ctx_part, q_part = ctx_user.split("[QUESTION]", 1)
                for limit in (2000, 1200, 800, 500, 300, 150):
                    ctx_short = ctx_part[:limit] + " ...\n"
                    msgs[-1]["content"] = ctx_short + "[QUESTION]" + q_part
                    prompt_text = _render(msgs)
                    input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                    if input_ids.size(0) <= INPUT_BUDGET:
                        break

                if input_ids.size(0) > INPUT_BUDGET:
                    msgs[-1]["content"] = "[QUESTION]" + q_part
                    prompt_text = _render(msgs)
    # print("[DBG] prompt_has_RAG =", ("[CONTEXT]" in prompt_text or "Retrieved passages:" in prompt_text))
    return prompt_text

def format_citations_md(citations: List[Dict], max_chars: int = 420) -> str:
    """
    将 [{'text','source','score'}] 渲染为 Markdown，带 blockquote 与来源。
    """
    if not citations:
        return ""
    lines = ["### Retrieved sources"]
    for i, c in enumerate(citations, 1):
        txt = (c.get("text") or "").strip().replace("\n", " ")
        if len(txt) > max_chars:
            txt = txt[:max_chars].rstrip() + " ..."
        src = c.get("source", "")
        score = c.get("score", 0.0)
        lines.append(f"\n**[{i}] Source:** `{src}`  —  **score:** {score:.3f}\n")
        lines.append(f"> {txt}")
    return "\n".join(lines).strip()


# ========= Streaming chat =========
def stream_reply(user_text: str, chat_history: List[List[str]]):
    """
    Streaming generator: yields updated chat_history as the assistant types.
    修复点：
    - 先使用当前 msg 的值生成，再清空输入框（在 UI 绑定里处理），避免空输入传入。
    - RAG: 将检索到的正文放入 extra_system，确保模型真的“看见”检索文本。
    """
    global CURRENT_MODEL

    # 0) 防御：空输入直接回传
    if not isinstance(user_text, str) or not user_text.strip():
        yield chat_history
        return

    # 1) Append user turn
    chat_history = chat_history + [[user_text, ""]]
    yield chat_history

    # 2) RAG（可选）
    rag_enabled = getattr(stream_reply, "rag_enabled", False)
    extra_system = None
    citations = []
    # ---- app.py ----

    if rag_enabled:
        if rag_enabled:
            try:
                rag_res = pipe.generate_rag(
                    user_message=user_text,
                    retriever=retriever,
                    history=_history_pairs_to_msgs(chat_history[:-1]),
                    top_k=3,
                )
                citations    = rag_res.get("citations", []) or []
                extra_system = rag_res.get("extra_system")  # ✅ 直接使用 pipeline 准备好的 extra_system
            except Exception as e:
                print("[RAG ERR]", e)
                rag_enabled = False
    
    # 4) Text streamer 原样
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

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
    # 3) Build prompt & tokenize  —— 改这里
    prompt_text = _build_prompt_for_stream(
        user_text,
        chat_history[:-1],
        extra_system=extra_system,           # RAG 文本
        tokenizer=tokenizer,                 
        gen_cfg=gen_cfg,                     
        model_max_input_tokens=getattr(CURRENT_MODEL.config, "max_position_embeddings", 4096),  # 兜底
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    dev = _model_device(CURRENT_MODEL)
    inputs = {k: v.to(dev) for k, v in inputs.items()}


    # 5) Generation config（与 pipeline 保持一致）


    # 6) Background generation thread
    def _worker():
        with torch.inference_mode():
            CURRENT_MODEL.generate(**inputs, **gen_cfg)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    # 7) Yield partial text
    partial = ""
    for new_text in streamer:
        partial += new_text
        chat_history[-1][1] = partial.strip()
        yield chat_history

    # 8) Generation finished：如果启用 RAG，追加 citation 信息（写入 chat 历史，供下游解析）
    if rag_enabled and citations:
        md = format_citations_md(citations)
        chat_history.append(["__RAG_CITATIONS__", md])
        yield chat_history

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
            rag_chk = gr.Checkbox(label="Enable RAG (retrieval-augmented)", value=False)

        # State for chat history
        state = gr.State([])
        rag_debug = gr.Textbox(label="RAG citations / debug", value="", lines=6)

        # helper: set rag flag on streaming function
        def set_rag_flag(flag: bool):
            setattr(stream_reply, "rag_enabled", bool(flag))
            return None

        # helper: after streaming, sync Chatbot -> state
        def sync_state_from_chat(h):
            return h

        # helper: extract citations from history into rag_debug
        def extract_citations(history_pairs: List[List[str]]):
            dbg = ""
            for u, a in history_pairs:
                if u == "__RAG_CITATIONS__":
                    dbg = a
            return dbg

        # ------------- FIXED BINDINGS (顺序调整：先流式，后清空) -------------
        # ENTER提交
        msg.submit(
            set_rag_flag, [rag_chk], []
        ).then(
            stream_reply, [msg, state], [chatbot]   # ✅ 先把当前 msg 传进去
        ).then(
            sync_state_from_chat, [chatbot], [state]
        ).then(
            lambda: "", None, [msg]                 # ✅ 再清空输入框
        ).then(
            extract_citations, [state], [rag_debug]
        )

        # 点击发送
        send_btn.click(
            set_rag_flag, [rag_chk], []
        ).then(
            stream_reply, [msg, state], [chatbot]   # ✅ 先流式
        ).then(
            sync_state_from_chat, [chatbot], [state]
        ).then(
            lambda: "", None, [msg]                 # ✅ 后清空
        ).then(
            extract_citations, [state], [rag_debug]
        )

        # 清空
        def on_clear():
            return [], [], ""
        clear_btn.click(on_clear, outputs=[chatbot, state, rag_debug], queue=False)

        # 示例按钮：把示例文本填入输入框
        for e in examples:
            gr.Button(e, size="sm").click(lambda x=e: x, outputs=msg)

        gr.Markdown("---")
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