# rag/build_medical_index.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import load_dataset
import json, os

print("Loading medical text...")

# 尝试加载公开数据集
try:
    dataset = load_dataset("bigbio/pubmed_qa_labeled_fold0", split="train[:1000]")
    texts = [x["question"] + " " + x["context"] for x in dataset if x.get("context")]
    print(f" have loaded PubMedQA : {len(texts)} ")
except Exception as e:
    print("Unable to access PubMed dataset, using local chatdoctor_200k_lora.jsonl data instead")
    local_path = r"D:\pycharm_project\pythonProject\LLM\LLM_Project_zx\data\chatdoctor_200k_lora.jsonl"
    texts = []
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                texts.append(ex["user"] + "\n" + ex["assistant"])
    else:
        raise RuntimeError(" No available dataset found, please check the network or local data directory.")

# 切分文本
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.create_documents(texts)

# 构建向量索引
print(" Computing embeddings and indexing...")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, emb)
db.save_local("rag/medical_index.faiss")

print(f" Medical index has been built, with a total of {len(docs)} documents")
print(" Save path: rag/medical_index.faiss")
