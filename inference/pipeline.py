# inference/pipeline.py
from __future__ import annotations
from typing import List, Dict, Optional, Sequence
import torch

# -------- helpers to assemble prompts --------

def _combine_system_prompt(cfg: dict, prompts: dict) -> str:
    sys_key = cfg.get("prompt_key_system", "system_en")
    grd_key = cfg.get("prompt_key_guard",  "guardrails_en")
    sty_key = cfg.get("prompt_key_style",  "style_en")
    blocks = [prompts.get(k, "") for k in (sys_key, grd_key, sty_key)]
    return "\n\n".join([b.strip() for b in blocks if b and b.strip()]).strip()

def _gather_few_shots(cfg: dict, prompts: dict) -> List[Dict[str, str]]:
    if not cfg.get("enable_few_shots", True):
        return []
    keys: Sequence[str] = cfg.get("few_shot_keys", [])
    limit = int(cfg.get("few_shot_limit", 2))
    shots: List[Dict[str, str]] = []
    for k in keys:
        arr = prompts.get(k, []) or []
        for ex in arr:
            if len(shots) >= limit * 2:  # one example = user + assistant
                break
            u, a = ex.get("user"), ex.get("assistant")
            if u:
                shots.append({"role": "user", "content": u})
            if a:
                shots.append({"role": "assistant", "content": a})
    return shots[: limit * 2] if limit > 0 else []

def _has_chat_template(tokenizer) -> bool:
    # transformers sets attribute (possibly None). Treat empty/None as no template.
    tpl = getattr(tokenizer, "chat_template", None)
    return bool(tpl and str(tpl).strip())

def _render_with_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Use HF chat template if available."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def _render_llama2_fallback(messages: List[Dict[str, str]]) -> str:
    """
    Render chat into a Llama-2-like prompt.
    Format:
      <s>[INST] <<SYS>>...<</SYS>> [/INST]
      assistant-first-reply
      <s>[INST] user [/INST]
      assistant ...
    We only need to end with an empty assistant turn (generation prompt).
    """
    sys_txt = ""
    parts = []
    i = 0

    # extract system if present
    if messages and messages[0].get("role") == "system":
        sys_txt = messages[0].get("content", "").strip()
        i = 1

    def inst(user_text: str, system_block: Optional[str] = None) -> str:
        if system_block:
            return f"<s>[INST] <<SYS>>\n{system_block}\n<</SYS>>\n\n{user_text.strip()} [/INST]\n"
        else:
            return f"<s>[INST] {user_text.strip()} [/INST]\n"

    # if we have a system, first turn must be a user to pair with it
    # We will pair (system + first user) -> first assistant (if any)
    # Then subsequent turns use user/assistant pairs.
    # We'll conclude with a trailing [INST] ... [/INST] for the final user message.

    # collect dialog turns after optional system
    dialog = messages[i:]

    # build conversation
    j = 0
    while j < len(dialog):
        m = dialog[j]
        if m["role"] == "user":
            # lookahead for assistant answer
            nxt = dialog[j+1]["content"] if (j + 1 < len(dialog) and dialog[j+1]["role"] == "assistant") else None
            if sys_txt:
                parts.append(inst(m["content"], system_block=sys_txt))
                sys_txt = ""  # system only injected once
            else:
                parts.append(inst(m["content"]))

            if nxt is not None:
                parts.append(nxt.strip() + "</s>\n")
                j += 2
            else:
                # last user (no assistant yet) => leave for generation
                j += 1
        else:
            # if an assistant turn appears without preceding user, just append raw
            parts.append((m.get("content") or "").strip() + "</s>\n")
            j += 1

    # ensure we end after a user [INST] ... [/INST] for generation
    # If last turn was assistant, user didn't ask anything—no generation prompt needed.
    # But typically we call generate with a final user turn appended.
    return "".join(parts)

# -------- Inference pipeline --------

class InferencePipeline:
    """
    Pipeline tailored for EPFL 'meditron-7b' (no chat_template).
    - English-only prompts
    - Structured style
    - Optional few-shot exemplars
    - Robust rendering: uses HF template if present, else Llama-2 fallback
    """

    def __init__(self, tokenizer, model, cfg: dict, prompts: dict):
        self.tokenizer = tokenizer
        self.model = model
        self.cfg = cfg
        self.prompts = prompts

        self.system_prompt = _combine_system_prompt(cfg, prompts)
        self.few_shots = _gather_few_shots(cfg, prompts)

        self.gen_cfg = {
            "max_new_tokens": int(cfg.get("max_new_tokens", 512)),
            "temperature": float(cfg.get("temperature", 0.5)),
            "top_p": float(cfg.get("top_p", 0.9)),
            "repetition_penalty": float(cfg.get("repetition_penalty", 1.05)),
            "do_sample": bool(cfg.get("do_sample", True)),
        }
        self.stop_words = cfg.get("stop_words", []) or []

    def _format_messages(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        extra_system: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        sys_txt = self.system_prompt
        if extra_system and extra_system.strip():
            sys_txt = (sys_txt + "\n\n" + extra_system.strip()).strip()
        if sys_txt:
            msgs.append({"role": "system", "content": sys_txt})

        msgs.extend(self.few_shots)

        if history:
            # expecting [{"role": "user"/"assistant", "content": "..."}]
            for turn in history:
                r, c = turn.get("role"), turn.get("content", "")
                if r in {"user", "assistant"} and c:
                    msgs.append({"role": r, "content": c})

        msgs.append({"role": "user", "content": user_message})
        return msgs

    @torch.inference_mode()
    def generate(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        extra_system: Optional[str] = None,
        override_gen_cfg: Optional[dict] = None,
    ) -> str:
        msgs = self._format_messages(user_message, history, extra_system)

        # 1) try chat template
        use_tpl = _has_chat_template(self.tokenizer)
        if use_tpl:
            try:
                prompt_text = _render_with_template(self.tokenizer, msgs)
            except Exception:
                prompt_text = _render_llama2_fallback(msgs)
        else:
            # 2) fallback to Llama-2 style
            prompt_text = _render_llama2_fallback(msgs)

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gcfg = dict(self.gen_cfg)
        if override_gen_cfg:
            gcfg.update(override_gen_cfg or {})
        gcfg["max_new_tokens"] = max(1, int(gcfg.get("max_new_tokens", 512)))

        out = self.model.generate(
            **inputs,
            do_sample=bool(gcfg.get("do_sample", True)),
            max_new_tokens=int(gcfg["max_new_tokens"]),
            temperature=float(gcfg["temperature"]),
            top_p=float(gcfg["top_p"]),
            repetition_penalty=float(gcfg["repetition_penalty"]),
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Only decode the continuation
        input_len = inputs["input_ids"].shape[-1]
        new_tokens = out[0, input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # optional stop phrase truncation
        if self.stop_words:
            cut = len(text)
            for s in self.stop_words:
                if not s:
                    continue
                idx = text.find(s)
                if idx != -1:
                    cut = min(cut, idx)
            text = text[:cut].strip()

        return text

    def generate_rag(
        self,
        user_message: str,
        retriever,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 3,
        extra_system: Optional[str] = None,
        override_gen_cfg: Optional[dict] = None,
    ) -> Dict[str, object]:
        """
        Prepare RAG context: retrieve + build extra_system.
        DO NOT generate here. Return data for the frontend to do streaming generation once.
        """
        try:
            retrieved = retriever.retrieve(user_message, top_k=top_k) or []
        except Exception:
            retrieved = []

        # 兼容抽取 text/source/score（如需更鲁棒，可用你之前的 _extract_tss）
        hits = []
        for r in retrieved:
            text  = (r.get("text") or r.get("page_content") or "").strip()
            src   = r.get("source") or (r.get("metadata") or {}).get("source", "")
            score = r.get("score") or 0.0
            try:
                score = float(score)
            except Exception:
                score = 0.0
            hits.append({"text": text, "source": src, "score": score})

        # 组装 extra_system
        rag_block = None
        if hits:
            lines = ["Retrieved passages:"]
            for i, h in enumerate(hits, start=1):
                lines.append(f"[{i}] (src={h['source']}, score={h['score']:.3f})\n{h['text']}")
            rag_block = "\n\n".join(lines)

        if extra_system and rag_block:
            extra_system = f"{extra_system}\n\n{rag_block}"
        else:
            extra_system = extra_system or rag_block  # 二者其一

        # 不生成，返回流式所需材料
        return {
            "extra_system": extra_system,  # ✅ 给前端唯一一次生成使用
            "citations": hits,             # ✅ 包含 text/source/score，供 UI 展示
            "debug_chunks": [h["text"] for h in hits],
        }
