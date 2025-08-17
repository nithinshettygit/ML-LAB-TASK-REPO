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

# --- Functions to load indexes ---
def load_vectorstores(vector_dir, embeddings_model):
    """Loads existing FAISS vector stores from disk."""
    vectorstore_subchapters = None
    try:
        vectorstore_subchapters = FAISS.load_local(
            folder_path=os.path.join(vector_dir, "subchapter_faiss_index"),
            embeddings=embeddings_model,
            allow_dangerous_deserialization=True
        )
        print("✅ Subchapter FAISS vector store loaded successfully.")
    except Exception:
        print("⚠️ Subchapter FAISS store not found. Skipping.")

    vectorstore_full_content = None
    try:
        vectorstore_full_content = FAISS.load_local(
            folder_path=os.path.join(vector_dir, "content_faiss_index"),
            embeddings=embeddings_model,
            allow_dangerous_deserialization=True
        )
        print("✅ Full content FAISS vector store loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading full content FAISS vector store: {e}")
        return None, None
    
    return vectorstore_subchapters, vectorstore_full_content

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # Load the vector stores
    vectorstore_subchapters, vectorstore_full_content = load_vectorstores(VECTOR_DB_DIR, embeddings)
    if not vectorstore_subchapters or not vectorstore_full_content:
        print("Please run the create_faiss_indexes.py script first to generate the indexes.")
        exit()

    try:
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            original_chunks = json.load(f)
        
        # Create a direct lookup map: Subchapter Name -> Document Chunk
        subchapter_to_chunk_map = {chunk.get("subchapter", "").strip(): chunk for chunk in original_chunks}
        print("✅ Subchapter to content mapping created.")
    except FileNotFoundError:
        print(f"❌ Error: {CHUNKS_FILE} not found.")
        exit()
    except json.JSONDecodeError:
        print(f"❌ Error: {CHUNKS_FILE} is not a valid JSON file.")
        exit()

    # Define a sample query for testing
    query_to_test = "What is a double displacement Reaction?"
    
    print(f"\n🔍 Performing two-stage search for: '{query_to_test}'...")
    
    # Stage 1: Search the subchapter FAISS index
    subchapter_results = vectorstore_subchapters.similarity_search(query_to_test, k=1)
    
    if not subchapter_results:
        print("No matching subchapter found.")
    else:
        top_subchapter_doc = subchapter_results[0]
        subchapter_name = top_subchapter_doc.page_content.strip()
        
        print(f"✅ Found top-ranked subchapter: '{subchapter_name}'")
        
        # Stage 2: Use the exact subchapter name for a direct lookup
        print("\n🔍 Performing direct content lookup...")
        
        final_doc = subchapter_to_chunk_map.get(subchapter_name)
        
        if final_doc:
            print("\n--- Final Result ---")
            print(f"Subchapter: {final_doc.get('subchapter', 'N/A')}")
            content_snippet = final_doc.get('content', '').replace('\n', ' ').strip()
            print(f"Content (first 300 chars): {content_snippet[:300]}...")
            
            figures_str = final_doc.get("figures", "")
            if figures_str:
                print("Figures:")
                print(f" - {figures_str}")
            else:
                print("Figures: None")
        else:
            print(f"❌ Error: Could not retrieve content for subchapter '{subchapter_name}'.")
