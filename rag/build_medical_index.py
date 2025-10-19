# rag/build_medical_index.py
import os, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document



print(" Loading local medical dataset...")

local_path = r"data\train.jsonl"
if not os.path.exists(local_path):
    raise FileNotFoundError(f" Local dataset not found: {local_path}")


docs = []
with open(local_path, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        user, assistant = ex.get("user", ""), ex.get("assistant", "")
        if not user or not assistant:
            continue
        if len(assistant) < 50 or "sorry" in assistant.lower():
            continue
        text = f"{user.strip()}\n\n{assistant.strip()}"
        docs.append(Document(page_content=text))

print(f" Collected {len(docs)} clean medical QA pairs for indexing.")

# ===========================

# ===========================
print(" Initializing BioBERT embeddings...")
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print(" Building FAISS index...")
db = FAISS.from_documents(docs, emb)
db.save_local("rag/medical_index.faiss")

print(f" Medical FAISS index built successfully ({len(docs)} docs)")
print(" Saved to: rag/medical_index.faiss")
