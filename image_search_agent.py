"""
Conversational agent: reuse the execution logic from LangGraph_Agent_Demo/local_agent.py,
but use Gemini as the LLM.

- You can extend tools here (e.g., local image search).
- The original clip_match-based implementation has been removed.
- The LLM uses Google Gemini 3.1 Flash-Lite (gemini-3.1-flash-lite-preview).
"""

from __future__ import annotations

import os
import csv
import warnings
from pathlib import Path
from typing import List, TypedDict

from urllib3.exceptions import NotOpenSSLWarning
from PIL import Image
import chromadb
import torch
import torch.nn.functional as F
import open_clip

from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Ignore urllib3 LibreSSL/NotOpenSSL warnings
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# Ignore google-related FutureWarnings about Python version
warnings.filterwarnings("ignore", category=FutureWarning, module="google")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core._python_version_support")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.auth")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.oauth2")


# =========================
# Agent State (same as LangGraph_Agent_Demo)
# =========================

class AgentState(TypedDict):
    messages: List[BaseMessage]


# =========================
# Courses dataset & tools (following local_agent.py)
# =========================

def load_courses(path: Path) -> List[dict]:
    """Load a CSV of courses if it exists; otherwise return empty list."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


COURSES_CSV_PATH = Path(__file__).with_name("courses.csv")
COURSES: List[dict] = load_courses(COURSES_CSV_PATH)


@tool
def search_courses(query: str) -> str:
    """Search the local course dataset for relevant entries."""
    if not COURSES:
        return "Course dataset not found. Please put a courses.csv next to image_search_agent.py."

    query_tokens = query.lower().split()
    scored = []

    for course in COURSES:
        text = " ".join(str(v) for v in course.values()).lower()
        score = sum(token in text for token in query_tokens)
        if score > 0:
            scored.append((score, course))

    if not scored:
        return "No relevant courses found in the dataset."

    scored.sort(key=lambda x: x[0], reverse=True)

    return "\n".join(
        f"{c.get('code','?')}: {c.get('title','(no title)')} ({c.get('level','?')}) - {c.get('description','')}"
        for _, c in scored
    )


@tool
def calc(expression: str) -> str:
    """Evaluate a simple arithmetic expression."""
    allowed = set("0123456789+-*/(). %")
    if any(ch not in allowed for ch in expression):
        return "Error: invalid characters"

    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


@tool
def write_text(path: str, content: str) -> str:
    """Write text content to a local file."""
    try:
        p = Path(path)
        p.write_text(content, encoding="utf-8")
        return f"Saved {len(content)} characters to {p}"
    except Exception as e:
        return f"Error: {e}"


# =========================
# Image Indexing (OpenCLIP + ChromaDB)
# Logic mirrors LangGraph_Agent_Demo/local_agent.py, but embeddings come from open_clip
# =========================

class ImageIndex:
    def __init__(self, image_dir: str = "images"):
        # Use an images subdirectory next to this file by default
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = os.path.join(base_dir, image_dir)

        # Initialize ChromaDB (persistent)
        chroma_path = os.path.join(base_dir, "chroma_db")
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name="local_images",
            metadata={"hnsw:space": "cosine"},
        )

        # Load OpenCLIP model
        print("Loading OpenCLIP model (ViT-B-32, laion2b_s34b_b79k)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        self._index_images()

    def _index_images(self) -> None:
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir, exist_ok=True)

        # Get list of images on disk
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        image_files = [
            f
            for f in os.listdir(self.image_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]

        if not image_files:
            print(f"No images found in {self.image_dir}")
            return

        # Check what's already indexed to avoid re-embedding
        existing_ids = set(self.collection.get()["ids"])

        new_images = []
        new_ids = []
        new_metadatas = []

        print("Checking for new images to index...")
        for f in image_files:
            file_id = f"img_{f}"  # Simple ID scheme
            if file_id not in existing_ids:
                try:
                    img_path = os.path.join(self.image_dir, f)
                    image = Image.open(img_path).convert("RGB")
                    new_images.append(image)
                    new_ids.append(file_id)
                    new_metadatas.append({"filename": f})
                except Exception as e:
                    print(f"Skipping {f}: {e}")

        if new_images:
            print(f"Embedding and indexing {len(new_images)} new images...")
            # Use OpenCLIP to encode images
            with torch.no_grad():
                image_tensors = [
                    self.preprocess(img).unsqueeze(0).to(self.device)
                    for img in new_images
                ]
                image_batch = torch.cat(image_tensors, dim=0)
                emb = self.model.encode_image(image_batch)
                # emb = F.normalize(emb, dim=-1)
                embeddings = emb.cpu().numpy()
            self.collection.add(
                embeddings=embeddings.tolist(),
                ids=new_ids,
                metadatas=new_metadatas,
            )
            print("Indexing complete.")
        else:
            print("Index is up to date.")

    def search(self, query: str, k: int = 3):
        # Embed query text with OpenCLIP
        with torch.no_grad():
            tokens = self.tokenizer([query]).to(self.device)
            text_emb = self.model.encode_text(tokens)
            # text_emb = F.normalize(text_emb, dim=-1)
            query_emb = text_emb.cpu().numpy()

        # Query Chroma
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=k,
        )

        # Parse results
        parsed = []
        if results and results.get("metadatas"):
            metas = results["metadatas"][0]
            dists = results["distances"][0]  # Chroma returns distances by default

            for i, meta in enumerate(metas):
                parsed.append((meta["filename"], dists[i]))

        return parsed


# Initialize global image index
image_index = ImageIndex()


@tool
def search_images(query: str) -> str:
    """Search for images/memes/photos locally using a text description."""
    results = image_index.search(query)
    if not results:
        return "No relevant images found in the local images directory."

    return "\n".join(
        f"{i+1}. {filename} (distance: {dist:.2f})"
        for i, (filename, dist) in enumerate(results)
    )


TOOLS = [search_courses, calc, write_text, search_images]


# =========================
# LLM (Google Gemini 3.1 Flash-Lite)
# =========================

# Model docs: https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite-preview
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"

# Paste your Gemini API key here for local use (do NOT commit real keys to Git)
GEMINI_API_KEY: str = "AIzaSyCoOmQlXcciTq6nzdoQTOLx2IUgZ1Y-GX8"  # e.g. "AIzaSyA......"


def set_api_key(key: str) -> None:
    global GEMINI_API_KEY
    GEMINI_API_KEY = key.strip()


def _get_llm():
    """Create a Gemini 3.1 Flash-Lite chat model bound with tools."""
    # Prefer GEMINI_API_KEY in this file; fall back to env vars
    api_key = GEMINI_API_KEY or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Gemini API key is not configured.\n"
            "Please either:\n"
            "1) Edit image_search_agent.py and set GEMINI_API_KEY, or\n"
            "2) Export GOOGLE_API_KEY / GEMINI_API_KEY in your shell."
        )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError("Please install: pip install langchain-google-genai")

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0,
        api_key=api_key,
        convert_system_message_to_human=True,
    ).bind_tools(TOOLS)


llm = None  # Lazy init so the module can be imported without an API key


def get_llm():
    global llm
    if llm is None:
        llm = _get_llm()
    return llm


# =========================
# System Prompt
# =========================

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a careful assistant with access to tools.\n\n"
        "Guidelines:\n"
        "- Use calc for ANY arithmetic, even if it seems simple.\n"
        "- Use write_text when the user asks to write text content to a local file.\n"
        "- Use search_courses when the user asks about courses, degrees, or programs.\n"
        "- Use search_images when the user asks to find/search an image by description or keyword.\n"
        "- When a tool returns results, summarize or filter them to answer the user's question.\n"
        "- Do NOT ask follow-up questions unless the request is ambiguous.\n"
        "- If you answer without tools and the answer could be wrong, that is a failure.\n"
        "- Prefer tools when accuracy matters."
    )
)


# =========================
# Graph Nodes (same structure as LangGraph_Agent_Demo)
# =========================

def agent_node(state: AgentState):
    messages = state["messages"]

    last_user_msg = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content
            break

    if messages and messages[-1].type == "tool":
        # Workaround: ToolMessage serialization is unstable in some Gemini integrations.
        # Convert the latest tool output into plain text messages before calling LLM again.
        tool_result = messages[-1].content
        non_tool_messages = [m for m in messages if getattr(m, "type", None) != "tool"]
        messages = non_tool_messages + [
            SystemMessage(
                content=(
                    "You have received tool results.\n"
                    f'The original user question was: "{last_user_msg}"\n'
                    "Answer the user directly using the tool results below.\n"
                    "Do NOT call any more tools."
                )
            ),
            HumanMessage(content=f"Tool results:\n{tool_result}"),
        ]
    else:
        messages = messages + [
            SystemMessage(
                content="Before answering, decide whether a tool would improve accuracy."
            )
        ]

    response = get_llm().invoke(messages)
    return {"messages": state["messages"] + [response]}


tool_node = ToolNode(TOOLS)


def route_after_agent(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# =========================
# Build LangGraph
# =========================

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile()


# =========================
# Run loop (same shape as LangGraph_Agent_Demo)
# =========================

def _render_assistant_content(content) -> str:
    """Render Gemini / LangChain structured content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t)
        if parts:
            return "\n".join(parts)
    return str(content)


if __name__ == "__main__":
    print("\nGemini Chat Agent ready. Type 'exit' to quit.\n")

    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        result = app.invoke(
            {
                "messages": [
                    SYSTEM_PROMPT,
                    HumanMessage(content=user),
                ]
            }
        )

        last = result["messages"][-1]
        print(f"\nAgent: {_render_assistant_content(last.content)}\n")
