# rag/rag_retriever.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings   # ✅ updated import

# Create embedding model
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the FAISS medical index
db = FAISS.load_local(
    r"D:\pycharm_project\pythonProject\LLM\LLM_Project_zx\rag\rag\medical_index.faiss",  # ✅ corrected path (no double 'rag\rag')
    emb,
    allow_dangerous_deserialization=True
)

def retrieve_context(query: str, k: int = 3) -> str:
    """Retrieve top-k relevant medical documents from FAISS index."""
    docs = db.similarity_search(query, k=k)
    return "\n".join([d.page_content for d in docs])
