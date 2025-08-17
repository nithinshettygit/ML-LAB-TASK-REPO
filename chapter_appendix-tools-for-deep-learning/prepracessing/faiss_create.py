# !pip install langchain_community
# !pip install fuzzywuzzy[speedup]
# !pip install regex
# !pip install sentence-transformers faiss-cpu numpy
# !pip install tqdm
# !pip install langgraph langchain-google-genai chromadb faiss-cpu
# !pip install langchain langchain-community
# !pip install flask  # later for web UI

import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document

# --- Configuration & Initialization ---
VECTOR_DB_DIR = "data/faiss_vectorstore_final"
SUBCHAPTER_METADATA_FILE = "subchapter_metadata.json"
CHUNKS_FILE = "merged_chunks_with_figures.json"

# Initialize embeddings
embedding_model_name = "thenlper/gte-large"
embedding_model_kwargs = {"device": "cuda"}
embedding_encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=embedding_encode_kwargs
)

# --- Load and Process Data ---
try:
    with open(SUBCHAPTER_METADATA_FILE, "r", encoding="utf-8") as f:
        subchapter_metadata = json.load(f)
    print("✅ Subchapter metadata loaded.")
    # Convert the dictionary to a list of Documents for FAISS
    subchapter_docs = [
        Document(page_content=subchapter_metadata[doc_id], metadata={"doc_id": doc_id})
        for doc_id in subchapter_metadata
    ]
except FileNotFoundError:
    print(f"❌ Error: {SUBCHAPTER_METADATA_FILE} not found. Please ensure it's in the same directory.")
    exit()
except json.JSONDecodeError:
    print(f"❌ Error: {SUBCHAPTER_METADATA_FILE} is not a valid JSON file.")
    exit()

try:
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        original_chunks = json.load(f)
    print("✅ Original data loaded.")
    # Convert the list of dictionaries to a list of Documents for FAISS
    content_docs = [
        Document(page_content=chunk.get("content", ""), metadata=chunk)
        for chunk in original_chunks if chunk.get("content") and chunk.get("id")
    ]
    # Create a dictionary for quick lookup by ID
    content_lookup = {str(chunk.get("id")): chunk for chunk in original_chunks}
except FileNotFoundError:
    print(f"❌ Error: {CHUNKS_FILE} not found. Please ensure it's in the same directory.")
    exit()
except json.JSONDecodeError:
    print(f"❌ Error: {CHUNKS_FILE} is not a valid JSON file.")
    exit()

# --- Create Indexes ---
if not os.path.exists(VECTOR_DB_DIR):
    os.makedirs(VECTOR_DB_DIR)

# Create and Save Subchapter Index
if subchapter_docs:
    print("🚀 Creating subchapter FAISS index...")
    vectorstore_subchapters = FAISS.from_documents(subchapter_docs, embeddings)
    vectorstore_subchapters.save_local(os.path.join(VECTOR_DB_DIR, "subchapter_faiss_index"))
    print("✅ Subchapter FAISS index created and saved successfully.")
else:
    print("⚠️ No valid subchapters found. Skipping subchapter index creation.")

# Create and Save Full Content Index
if content_docs:
    print("🚀 Creating full content FAISS index...")
    vectorstore_full_content = FAISS.from_documents(content_docs, embeddings)
    vectorstore_full_content.save_local(os.path.join(VECTOR_DB_DIR, "content_faiss_index"))
    print("✅ Full content FAISS index created and saved successfully.")
else:
    print("❌ No valid content chunks found. Cannot create full content index.")
    exit()
