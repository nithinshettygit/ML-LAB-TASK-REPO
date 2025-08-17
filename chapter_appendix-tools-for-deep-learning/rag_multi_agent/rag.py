import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ----------------------------
# CONFIGURATION
# ----------------------------
VECTOR_DB_DIR = "data/faiss_vectorstore_final"
CHUNKS_FILE = "merged_chunks_with_figures.json"
TOP_K = 5
EMBEDDING_MODEL = "thenlper/gte-large"
DEVICE = "cuda"
GOOGLE_API_KEY = "AIzaSyDwSeSRNqTrUbw10XzkW-xYIUEtK4vPVg8"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ----------------------------
# EMBEDDINGS & FAISS
# ----------------------------
embeddings = HuggingFaceBgeEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)

def load_faiss_indexes(vector_dir):
    """Load subchapter and content FAISS vector stores"""
    try:
        vectorstore_subchapters = FAISS.load_local(
            os.path.join(vector_dir, "subchapter_faiss_index"),
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ Subchapter FAISS loaded.")
    except Exception:
        vectorstore_subchapters = None
        print("⚠️ Subchapter FAISS not found.")

    try:
        vectorstore_content = FAISS.load_local(
            os.path.join(vector_dir, "content_faiss_index"),
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ Content FAISS loaded.")
    except Exception as e:
        vectorstore_content = None
        print(f"❌ Error loading content FAISS: {e}")

    return vectorstore_subchapters, vectorstore_content

vectorstore_subchapters, vectorstore_content = load_faiss_indexes(VECTOR_DB_DIR)
if not vectorstore_subchapters or not vectorstore_content:
    raise RuntimeError("FAISS indexes missing. Run index creation first.")

# ----------------------------
# LOAD JSON CHUNKS
# ----------------------------
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

subchapter_to_chunk = {c.get("subchapter", "").strip(): c for c in chunks}

# ----------------------------
# PROMPT TEMPLATES
# ----------------------------
lesson_prompt_template = """
You are an 8th-grade science teacher. Using the following content, generate a **detailed, student-facing lesson** in Markdown:
Content: {retrieved_content}
Figures: {figures}

Instructions:
- Present the lesson directly to students.
- Include a short funny/memorable introduction.
- Integrate figures with <img src='...' alt='Figure'> and their descriptions.
- Highlight important terms in bold (like CNS, PNS, hormones, reflex arc, forebrain, midbrain, hindbrain).
- Structure lesson: Introduction, Explanation with examples, Figure integration, Key takeaways, Summary, Questions/Activities.
"""

figure_prompt_template = """
You are a science teacher. For the following list of figures, provide a **brief, student-friendly explanation** of each figure:
Figures: {figures_json}

Return explanations in Markdown format.
"""

lesson_prompt = PromptTemplate(input_variables=["retrieved_content", "figures"], template=lesson_prompt_template)
figure_prompt = PromptTemplate(input_variables=["figures_json"], template=figure_prompt_template)

# ----------------------------
# LLMs (Gemini)
# ----------------------------
llm_main = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    temperature=0.7,
    max_tokens=3000,
    api_key=GOOGLE_API_KEY
)

llm_figure = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    temperature=0.7,
    max_tokens=1500,
    api_key=GOOGLE_API_KEY
)

# ----------------------------
# FUNCTION: Generate figure explanations
# ----------------------------
def generate_figure_explanations(figures_list):
    """Generates figure explanations via LLM"""
    if not figures_list:
        return ""
    figures_json = json.dumps(figures_list)
    chain = LLMChain(llm=llm_figure, prompt=figure_prompt)
    return chain.run(figures_json=figures_json)

# ----------------------------
# FUNCTION: Generate lesson
# ----------------------------
def generate_lesson(query: str, top_k: int = TOP_K):
    # 1️⃣ Retrieve top subchapter
    sub_results = vectorstore_subchapters.similarity_search(query, k=1)
    if not sub_results:
        print("⚠️ No matching subchapter found.")
        return None
    top_subchapter = sub_results[0].page_content.strip()
    print(f"Top Subchapter: {top_subchapter}")

    # 2️⃣ Retrieve content chunks
    chunk_doc = subchapter_to_chunk.get(top_subchapter)
    if not chunk_doc:
        print(f"❌ No chunk found for subchapter '{top_subchapter}'")
        return None

    content_results = vectorstore_content.similarity_search(query, k=top_k)
    combined_texts = [doc.page_content for doc in content_results]
    retrieved_content = "\n\n".join(combined_texts)

    # 3️⃣ Handle figures
    figures_list = chunk_doc.get("figures", [])
    figure_explanations = generate_figure_explanations(figures_list)

    # Build figures string
    figures_str = ""
    for fig in figures_list:
        figures_str += f"<img src='[image_url_placeholder]' alt='{fig['figure']}' />\n**Description:** {fig['desc']}\n\n"

    if figure_explanations:
        figures_str += figure_explanations + "\n"

    # 4️⃣ Generate lesson via main LLM
    chain = LLMChain(llm=llm_main, prompt=lesson_prompt)
    lesson = chain.run(retrieved_content=retrieved_content, figures=figures_str)
    return lesson

# ----------------------------
# MAIN AGENT LOOP
# ----------------------------
if __name__ == "__main__":
    user_query = input("Enter the subchapter name or topic: ").strip()
    lesson_output = generate_lesson(user_query, top_k=TOP_K)
    if lesson_output:
        print("\n\n--- Generated Lesson ---\n")
        print(lesson_output)
