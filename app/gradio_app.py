# app/gradio_app.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rag.rag_retriever import retrieve_context


# ===== Model Loading =====
model_id = "meta-llama/Llama-3.2-3B-Instruct"
adapter_dir = "finetune/adapter"

tok = AutoTokenizer.from_pretrained(model_id)
base = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, adapter_dir)
model.eval()


# ===== Chat Function =====
def medcarebot(message, history):
    context = retrieve_context(message)

    messages = [
        {"role": "system", "content": "You are a helpful and safe medical assistant. Never provide a direct diagnosis or prescription."}
    ]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": f"{message}\n\nReference:\n{context}"})

    # 🟢 This returns a tensor — we must wrap it in a dict
    inputs = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # ✅ Wrap in dict for model.generate()
    input_dict = {"input_ids": inputs}

    outputs = model.generate(
        **input_dict,
        max_new_tokens=400,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )

    # 🟢 Only decode the generated part
    input_len = inputs.shape[-1]
    new_tokens = outputs[0, input_len:]
    reply = tok.decode(new_tokens, skip_special_tokens=True).strip()

    if "Reference:" in reply:
        reply = reply.split("Reference:")[0].strip()

    return reply




# ===== Gradio Chat Interface =====
chatbot = gr.ChatInterface(
    fn=medcarebot,
    title="🩺 MedCareBot - Medical Q&A Assistant",
    description=(
        "Ask about your health symptoms or medical topics.\n"
        "⚠️ Disclaimer: This system is for research and educational use only. "
        "It is not a substitute for professional medical advice."
    ),
    theme="soft",  # optional: "soft", "glass", "default"
    textbox=gr.Textbox(placeholder="Type your symptom or question here...", container=False, scale=7),
    retry_btn="🔄 Retry",
    undo_btn="↩️ Undo",
    clear_btn="🧹 Clear Chat",
)

if __name__ == "__main__":
    # chatbot.launch(server_name="0.0.0.0", server_port=7860)
    chatbot.launch(server_name="127.0.0.1", server_port=7860)

