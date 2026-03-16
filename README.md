# api_demo — LangGraph + Gemini Demo

This directory contains a minimal **LangGraph + Gemini** conversational agent, showing how to build a tool-using agent locally:

- Uses **Google Gemini 3.1 Flash-Lite** as the LLM.
- Uses LangGraph to wire together an `agent` node and a `tools` node.

## Environment

1. Install Python and dependencies:

   - Make sure you have **Python 3.10+** installed. You can download it from `https://www.python.org/downloads/`.
   - Create and activate a virtual environment in the `LangGraph_Agent_Demo_with_API` folder (recommended):

   ```bash
   python3 -m venv .venv          # or `python` if it is 3.10+
   source .venv/bin/activate      # on macOS / Linux
   # On Windows (PowerShell):
   # .venv\Scripts\Activate.ps1

   pip install -r requirements.txt
   ```

2. Get a Gemini API key (free):  
   Create a key in [Google AI Studio](https://aistudio.google.com/apikey).

3. Configure the Gemini API key in code (no env vars required):  
   Open `image_search_agent.py` and find:

   ```python
   GEMINI_API_KEY: str = ""  # e.g. "AIzaSyA......"
   ```

   Set the string value to your own Gemini API key.

## Run the conversational agent

From inside the `LangGraph_Agent_Demo_with_API` directory:

```bash
python image_search_agent.py
```

You will enter a simple REPL. You can ask questions in English or Chinese and the agent will respond using Gemini and tools.  
When you ask to “search for an image” or “find a picture” by description, the agent will call the `search_images` tool to search local images (in the `images/` folder) using CLIP + Chroma.
