#!/usr/bin/env python3
# AI Teacher agent — complete corrected version with robust image-finding
# Run in Colab/Jupyter for inline HTML rendering.

import os
import json
import glob
import re
from typing import TypedDict, List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# Optional CUDA detection (safe fallback)
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# ----------------------------
# CONFIG
# ----------------------------
VECTOR_DB_DIR = "data/faiss_vectorstore_final"
CHUNKS_FILE = "merged_chunks_with_figures.json"
IMAGE_DIR = "/content/data/images"   # relative path (works in Colab: /content/data/images)
TOP_K = 5
EMBEDDING_MODEL = "thenlper/gte-large"

# Read API key from environment
GOOGLE_API_KEY = "AIzaSyDwSeSRNqTrUbw10XzkW-xYIUEtK4vPVg8"  # <-- Replace with your actual key yes i have underscore

if not GOOGLE_API_KEY:
    print("⚠️ WARNING: GOOGLE_API_KEY not set in environment. Set os.environ['GOOGLE_API_KEY']='YOUR_KEY' before running.")

# ----------------------------
# STATE TYPE
# ----------------------------
class AgentState(TypedDict, total=False):
    query: str
    retrieved_content: str
    figures_list: List[dict]
    topic: str
    image_html: str
    video_html: str
    lesson_text: str
    flow_step: str

# ----------------------------
# EMBEDDINGS + FAISS LOADER
# ----------------------------
embeddings = HuggingFaceBgeEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)

def load_faiss_indexes(vector_dir: str):
    subch = cont = None
    try:
        subch = FAISS.load_local(
            os.path.join(vector_dir, "subchapter_faiss_index"),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("✅ Subchapter FAISS loaded.")
    except Exception as e:
        print(f"❌ Subchapter FAISS load error: {e}")

    try:
        cont = FAISS.load_local(
            os.path.join(vector_dir, "content_faiss_index"),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("✅ Content FAISS loaded.")
    except Exception as e:
        print(f"❌ Content FAISS load error: {e}")

    return subch, cont

vectorstore_subchapters, vectorstore_content = load_faiss_indexes(VECTOR_DB_DIR)

# Load JSON chunks
if not os.path.exists(CHUNKS_FILE):
    raise FileNotFoundError(f"{CHUNKS_FILE} not found. Upload your merged chunks JSON.")
with open(CHUNKS_FILE, "r", encoding="utf-8") as fh:
    chunks = json.load(fh)

# Build mapping; only include valid subchapter entries
subchapter_to_chunk = {
    c.get("subchapter").strip(): c
    for c in chunks
    if isinstance(c, dict) and c.get("subchapter")
}

# ----------------------------
# HELPERS: robust image finder + HTML builder
# ----------------------------
def normalize_tokens(name: str) -> List[str]:
    """Return list of alphanumeric tokens from the figure name (no punctuation)."""
    return re.findall(r"[A-Za-z0-9]+", name)

def find_image_file(figure_name: str) -> Optional[str]:
    """
    Robustly locate an image file for a given figure_name.
    Strategies:
     1) spaces->underscores exact matches (png/jpg)
     2) token-driven stems (underscore, dot numeric combos)
     3) fuzzy glob on important tokens
    Returns absolute or relative path if found, else None.
    """
    if not figure_name or not figure_name.strip():
        return None

    # trim and remove trailing punctuation
    plain = figure_name.strip()
    plain = re.sub(r"[.,:;!?\)\(]+$", "", plain).strip()
    cand = plain.replace(" ", "_")

    candidates = [
        f"{cand}.png",
        f"{cand}.jpg",
        f"{cand}.jpeg",
        f"{cand.lower()}.png",
        f"{cand.lower()}.jpg",
    ]
    for c in candidates:
        p = os.path.join(IMAGE_DIR, c)
        if os.path.exists(p):
            return p

    # token-driven stems
    tokens = normalize_tokens(figure_name)
    stems = []
    for n in range(1, min(6, len(tokens) + 1)):
        stems.append("_".join(tokens[:n]))
    # try numeric dot combos (e.g., 1.1)
    for i in range(len(tokens) - 1):
        if tokens[i].isdigit() and tokens[i + 1].isdigit():
            stems.append("_".join(tokens[:i] + [f"{tokens[i]}.{tokens[i + 1]}"] + tokens[i + 2: i + 2]))
    # dedupe
    seen = set()
    stems = [s for s in stems if not (s in seen or seen.add(s))]

    for s in stems:
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(IMAGE_DIR, s + ext)
            if os.path.exists(p):
                return p
            p2 = os.path.join(IMAGE_DIR, f"Figure_{s}{ext}")
            if os.path.exists(p2):
                return p2

    # fuzzy glob by tokens (first up to 4 tokens)
    if tokens:
        pattern_tokens = tokens[:4]
        glob_pattern = "*" + "*".join(pattern_tokens) + "*.png"
        matches = glob.glob(os.path.join(IMAGE_DIR, glob_pattern))
        if matches:
            return matches[0]
        glob_pattern_jpg = "*" + "*".join(pattern_tokens) + "*.jp*g"
        matches = glob.glob(os.path.join(IMAGE_DIR, glob_pattern_jpg))
        if matches:
            return matches[0]

    return None

def build_clickable_image_html(figure_name: str, caption: str = "", thumb_width: int = 360) -> str:
    """Return a clickable thumbnail HTML snippet for a figure name (or a red warning if not found)."""
    img_path = find_image_file(figure_name)
    if not img_path:
        return f"<p style='color:crimson;'>Image not found for '{figure_name}'. Checked '{IMAGE_DIR}'.</p>"

    # Use relative path (works in Colab). Wrap <img> in <a> so it opens full image in new tab.
    html = (
        "<div style='text-align:center; margin:16px 0;'>"
        f"<a href='{img_path}' target='_blank' rel='noopener noreferrer'>"
        f"<img src='{img_path}' alt='{caption or figure_name}' width='{thumb_width}' "
        "style='max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 6px rgba(0,0,0,0.12);'/>"
        "</a>"
        f"<div style='font-style:italic; font-size:0.95em; margin-top:6px;color:#333'>{caption or figure_name}</div>"
        "</div>"
    )
    return html

# ----------------------------
# HELPERS: optional youtube video fetcher
# ----------------------------
def _video_html_from_topic(topic: str) -> str:
    """Search YouTube with yt-dlp for a single relevant video and return iframe HTML."""
    try:
        import yt_dlp
    except Exception:
        return ""  # yt_dlp not available

    search_q = f"ytsearch1:{topic} animation explained"
    ydl_opts = {"quiet": True, "extract_flat": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_q, download=False)
            entries = info.get("entries") or []
            if not entries:
                return ""
            video_info = entries[0] or {}
            vid = video_info.get("id")
            title = video_info.get("title") or ""
            if not vid:
                return ""
            return (
                "<div style='margin-top:20px;'>"
                "<h3>🎥 Video Explanation</h3>"
                "<p>Watch this short animation to learn more:</p>"
                f"<iframe width='560' height='315' src='https://www.youtube.com/embed/{vid}' "
                "frameborder='0' allowfullscreen style='max-width:100%;'></iframe>"
                f"<p><em>{title}</em></p>"
                "</div>"
            )
    except Exception:
        return ""

# ----------------------------
# AGENT NODES
# ----------------------------
def retrieval_agent(state: AgentState) -> AgentState:
    print("--- Retrieval Agent ---")
    if vectorstore_subchapters is None or vectorstore_content is None:
        print("❌ FAISS stores missing.")
        return {"flow_step": "end_with_error"}

    query = state.get("query", "").strip()
    if not query:
        print("⚠️ Empty query.")
        return {"flow_step": "end_with_error"}

    sub_res = vectorstore_subchapters.similarity_search(query, k=1)
    if not sub_res:
        print("⚠️ No matching subchapter.")
        return {"flow_step": "end_with_error"}

    top_sub = sub_res[0].page_content.strip()
    print(f"Top subchapter: {top_sub}")

    chunk_doc = subchapter_to_chunk.get(top_sub)
    if not chunk_doc:
        print(f"❌ No chunk_doc for '{top_sub}'.")
        return {"flow_step": "end_with_error"}

    content_docs = vectorstore_content.similarity_search(query, k=TOP_K)
    combined = "\n\n".join([d.page_content for d in content_docs]) if content_docs else ""
    return {
        "retrieved_content": combined,
        "figures_list": chunk_doc.get("figures", []),
        "topic": top_sub,
        "flow_step": "multimedia_generation",
    }

def multimedia_agent(state: AgentState) -> AgentState:
    print("--- Multimedia Agent ---")
    figures = state.get("figures_list", [])
    # Build image HTML by calling the clickable builder for each figure
    image_html_parts = []
    for f in figures:
        fig_name = f.get("figure", "")
        desc = f.get("desc", "")
        image_html_parts.append(build_clickable_image_html(fig_name, desc))
    image_html = "\n".join(image_html_parts)
    video_html = _video_html_from_topic(state.get("topic", ""))
    return {"image_html": image_html, "video_html": video_html, "flow_step": "lesson_generation"}

def teacher_agent(state: AgentState) -> AgentState:
    print("--- Teacher Agent ---")
    prompt = PromptTemplate(
        input_variables=["retrieved_content", "image_html", "video_html"],
        template="""
You are an 8th-grade science teacher. Using the following content, generate a friendly, student-facing lesson in HTML.

Content:
{retrieved_content}

Insert the Figures HTML exactly where appropriate:
{image_html}

Insert the Video HTML exactly where appropriate:
{video_html}

Structure the lesson into:
1) Introduction (funny/memorable)
2) Explanation with examples
3) Visual aids area (use provided figure HTML)
4) Video explanation area (use provided video HTML)
5) Key takeaways
6) Summary

Please output valid HTML fragments (do not escape the provided figure_html or video_html).
""",
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-8b",
        temperature=0.7,
        max_tokens=3000,
        api_key=GOOGLE_API_KEY or None,
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        # LLMChain.run returns the text directly; using run is simpler and more consistent here
        lesson_text = chain.run(
            retrieved_content=state.get("retrieved_content", ""),
            image_html=state.get("image_html", ""),
            video_html=state.get("video_html", ""),
        )
    except Exception as e:
        # fallback: attempt invoke and tolerant extraction
        try:
            result = chain.invoke({
                "retrieved_content": state.get("retrieved_content", ""),
                "image_html": state.get("image_html", ""),
                "video_html": state.get("video_html", ""),
            })
            lesson_text = (
                (result.get("text") if isinstance(result, dict) else None)
                or (result.get("output_text") if isinstance(result, dict) else None)
                or (result[0] if isinstance(result, list) and result else None)
                or str(result)
            )
        except Exception as e2:
            lesson_text = f"<p style='color:crimson;'>LLM generation failed: {e}; fallback error: {e2}</p>"

    return {"lesson_text": lesson_text, "flow_step": "display_result"}

def display_agent(state: AgentState) -> AgentState:
    print("--- Display Agent ---")
    final_html = f"""
<html>
<head><meta charset="utf-8"><style>
body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
.lesson-container {{ background:#f9f9f9; padding:20px; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.06); }}
h1 {{ text-align:center; color:#2e6c80; }}
</style></head>
<body>
  <div class="lesson-container">
    <h1>🌟 AI Teacher Lesson</h1>
    {state.get('lesson_text', '')}
  </div>
</body>
</html>
"""
    try:
        from IPython.display import display, HTML
        display(HTML(final_html))
    except Exception:
        print(final_html)
    return state

def error_node(state: AgentState) -> AgentState:
    print("--- Error Node ---")
    fallback = "<h2>⚠️ Lesson generation failed — check data and FAISS indexes.</h2>"
    try:
        from IPython.display import display, HTML
        display(HTML(fallback))
    except Exception:
        print(fallback)
    return state

# ----------------------------
# LANGGRAPH BUILD
# ----------------------------
workflow = StateGraph(AgentState)
workflow.add_node("retrieval_agent", retrieval_agent)
workflow.add_node("multimedia_agent", multimedia_agent)
workflow.add_node("teacher_agent", teacher_agent)
workflow.add_node("display_agent", display_agent)
workflow.add_node("error_node", error_node)

workflow.set_entry_point("retrieval_agent")
workflow.add_edge("multimedia_agent", "teacher_agent")
workflow.add_edge("teacher_agent", "display_agent")
workflow.add_edge("display_agent", END)
workflow.add_conditional_edges(
    "retrieval_agent",
    lambda s: "error" if s.get("flow_step") == "end_with_error" else "success",
    {"success": "multimedia_agent", "error": "error_node"},
)
workflow.add_edge("error_node", END)

app = workflow.compile()

# ----------------------------
# RUN (example)
# ----------------------------
if __name__ == "__main__":
    if vectorstore_subchapters is None or vectorstore_content is None:
        print("❌ Please ensure FAISS indexes are present at the configured path.")
    else:
        user_query = input("Enter topic/subchapter (e.g., 'human brain'): ").strip() or "human brain"
        print(f"Starting generation for: {user_query}")
        final_state = app.invoke({"query": user_query, "flow_step": "start"})
