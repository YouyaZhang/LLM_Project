# rag/rag_retriever.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db_path = r"rag\medical_index.faiss"
db = FAISS.load_local(db_path, emb, allow_dangerous_deserialization=True)


def retrieve_context(query: str, k: int = 6) -> str:
    """
    Retrieve top-k relevant medical documents from FAISS index.
    Returns a concatenated string of retrieved texts.
    """
    docs = db.similarity_search(query, k=k)
    context_text = "\n\n---\n\n".join([d.page_content for d in docs])
    return context_text
