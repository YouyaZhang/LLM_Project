# app/gradio_app.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rag.rag_retriever_v1 import retrieve_context

# =========================================================

# =========================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =========================================================
# 1) System Prompt
# =========================================================
SYSTEM_PROMPT = """
You are **MedCareBot**, a reliable, concise, and empathetic AI medical assistant.

Your goals:
- Understand the user's question precisely.
- If medical reference text is provided, use it only to improve accuracy. Do NOT copy verbatim.
- Give short, structured, safe guidance.

Safety & Style:
- 4–6 short sentences (~100–150 words).
- Clear, factual, neutral tone; no greetings, names, links, emails, or promotional text.
- Avoid diagnosis or prescribing; recommend seeking care when appropriate.
- Stop after one complete answer (no follow-on phrases or invitations).

If the user has symptoms, include:
- Likely causes (very brief).
- Whether medical attention is needed (and urgency).
- 2–3 practical home-care steps.
"""


INSTRUCT_INPUT_RE = re.compile(r"###\s*Input:\s*(.*)", re.DOTALL | re.IGNORECASE)

def extract_real_question(user_text: str) -> str:
   
    if not user_text:
        return ""
    m = INSTRUCT_INPUT_RE.search(user_text)
    return m.group(1).strip() if m else user_text.strip()

def is_medical_like(q: str) -> bool:
    if len(q.split()) < 5:
        return False
    kw = [
        "fever", "cough", "pain", "headache", "chest", "abdomen", "throat",
        "urine", "vomit", "diarrhea", "injury", "bleeding", "rash", "swelling",
        "blood test", "x-ray", "ct", "mri", "antibiotic", "infection",
        "pregnan", "diabetes", "hypertension", "asthma", "cancer"
    ]
    ql = q.lower()
    return any(k in ql for k in kw)

def truncate_by_tokens(text: str, tokenizer, max_tokens: int) -> str:
   
    if not text:
        return text
    enc = tokenizer(text, add_special_tokens=False)
    ids = enc["input_ids"]
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)

def clean_output(text: str) -> str:
   

    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[.*?\]|\(.*?\)", "", text)
    text = re.sub(r"(?i)\b(reindia|apollo|mayoclinic|medscape|nhs|healthtap)\b.*", "", text)


    forbidden = [
        r"(?i)dear\s+(patient|sir|madam|friend)[,;:]?",
        r"(?i)(dr\.?|doctor)\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)*[,;:]?",
        r"(?i)(md|mbbs|dnb|phd|consultant|specialist)[,;:]?",
        r"(?i)(regards|best wishes|sincerely|yours truly|thank you).*",
        r"(?i)feel free to ask.*",
        r"(?i)happy to assist.*",
        r"(?i)if you wish to ask further.*",
        r"(?i)for more health tips.*",
        r"(?i)this answers your query.*",
        r"(?i)visit.*(clinic|hospital|website).*",
        r"(?i)call.*(helpline|number).*",
        r"(?i)please contact.*",
    ]
    for pat in forbidden:
        text = re.sub(pat, "", text)


    text = re.sub(r"[\*\#\_]{2,}", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.replace(" ,", ",").replace(" .", ".").strip()


    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen, uniq = set(), []
    for s in sentences:
        s_clean = s.strip().lower()
        if s_clean and s_clean not in seen:
            uniq.append(s.strip())
            seen.add(s_clean)
    text = " ".join(uniq)


    if len(uniq) > 3:
        for i in range(1, len(uniq)):
            if uniq[i][:25].lower() in uniq[0].lower():
                text = " ".join(uniq[:i])
                break


    end_phrases = [
        "take care", "have a great day", "hope you feel better",
        "get well soon", "wish you good health", "hope this helps"
    ]
    low = text.lower()
    cut_idx = min([low.find(p) for p in end_phrases if low.find(p) != -1] + [len(text)])
    text = text[:cut_idx].rstrip(". ")  
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s*([?.!,])", r"\1", text).strip()


    words = text.split()
    if len(words) > 150:
        text = " ".join(words[:150]) + "..."
    return text


model_id = "meta-llama/Llama-3.2-3B-Instruct"
adapter_dir = r"\finetune\adapter-20251017T224222Z-1-001\adapter"

print(" Loading model and adapter...")
tok = AutoTokenizer.from_pretrained(model_id)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
base = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
).to(DEVICE)
model = PeftModel.from_pretrained(base, adapter_dir)
model.to(DEVICE)
model.eval()
print(f" Model ready on: {DEVICE}")


def build_user_block(question: str, tok, enable_rag: bool = True) -> str:
  
    q = extract_real_question(question)

    reference_block = ""
    if enable_rag and is_medical_like(q):
        context = retrieve_context(q, k=6).strip()
        if context:
           
            context = truncate_by_tokens(context, tok, max_tokens=800)
            # reference_block = (
            #     "\n\nUse the following medical reference to ensure factual accuracy. "
            #     "Do NOT copy verbatim; integrate only helpful facts.\n"
            #     "-----\n" + context + "\n-----\n"
            # )
            reference_block = (
                    "\n\nHere is some relevant medical information that may help you answer more accurately. "
                    "Use it only as factual support, not as part of the response.\n"
                    "-----\n" + context + "\n-----\n"
            )

  
    return f"{q}{reference_block}\nRespond once, concisely, and stop after your answer."

def medcarebot(message, history):
 
    history = history[-3:] if history else []


    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in history:
        if u: messages.append({"role": "user", "content": u})
        if a: messages.append({"role": "assistant", "content": a})

    user_block = build_user_block(message, tok, enable_rag=True)
    messages.append({"role": "user", "content": user_block})


    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)


    with torch.no_grad():
        # outputs = model.generate(
        #     input_ids=inputs,
        #     max_new_tokens=800,
        #     do_sample=True,
        #     temperature=0.55,
        #     top_p=0.9,
        #     repetition_penalty=1.18,
        #     no_repeat_ngram_size=4,
        #     pad_token_id=tok.eos_token_id,
        #     eos_token_id=tok.eos_token_id,
        # )
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=800,
            min_new_tokens=80,
            do_sample=True,
            temperature=0.55,
            top_p=0.9,
            repetition_penalty=1.18,
            no_repeat_ngram_size=4,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )


    input_len = inputs.shape[-1]
    gen_tokens = outputs[0, input_len:]
    # reply = tok.decode(gen_tokens, skip_special_tokens=True)
    # reply = clean_output(reply)
    reply = tok.decode(gen_tokens, skip_special_tokens=True)
    reply = clean_output(reply)


    if re.search(r"\b(and|or|with|to|for|your|the|a|an)$", reply.strip().lower()):
        reply += " doctor for a proper evaluation."


    elif not reply.endswith(('.', '!', '?')):
        reply += '.'
    return reply

# =========================================================
# 5) Gradio 
# =========================================================
chatbot = gr.ChatInterface(
    fn=medcarebot,
    title="🩺 MedCareBot — RAG-Enhanced Medical Assistant",
    description=(
        "Describe symptoms or ask medical questions.\n"
        "⚠️ For research & education only — not a substitute for professional medical advice."
    ),
    theme="soft",
    textbox=gr.Textbox(
        placeholder="Describe your symptoms or question here...",
        container=False,
        scale=7
    ),
    retry_btn="🔄 Retry",
    undo_btn="↩️ Undo",
    clear_btn="🧹 Clear Chat",
)

if __name__ == "__main__":
    chatbot.launch(server_name="127.0.0.1", server_port=7860)
