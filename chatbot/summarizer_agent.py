from __future__ import annotations

"""ADK agent that summarizes the conversation and writes a Markdown file.

This agent is intended to be used as a sub-agent of the root chatbot. It
produces a concise, structured summary of the current chat turn/context and
persists it to `chatbot/downloads/` as a Markdown file with YAML frontmatter.

Notes:
- The agent will generate the summary text and then call the provided tool to
  save the Markdown into the downloads folder.
- The tool returns the final file path which the agent should include in its
  final message back to the root.
"""

import os
import re
import datetime as _dt
from typing import Dict

from google.adk.agents import Agent


DOWNLOADS_DIR = os.path.join(os.path.dirname(__file__), "downloads")


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\-\_\s]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text or "summary"


def save_markdown_to_downloads(title: str, content_markdown: str) -> Dict[str, str]:
    """Create a Markdown file under `chatbot/downloads/` with YAML frontmatter.

    Args:
        title: Title for the summary document.
        content_markdown: The body content in Markdown format. The agent should
            provide a structured markdown (headings like Overview, Key Findings,
            Citations, Action Items, Sources) as appropriate.

    Returns:
        dict: {"status": "success"|"error", "path": str, "filename": str, "error_message"?: str}

    Expected usage pattern by the agent:
        1) Generate the structured markdown content.
        2) Call this tool with the title and content.
        3) Include the returned file path in the final answer.
    """
    try:
        os.makedirs(DOWNLOADS_DIR, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = _slugify(title)
        filename = f"{slug}-{ts}.md"
        path = os.path.join(DOWNLOADS_DIR, filename)

        frontmatter = (
            f"---\n"
            f"title: \"{title}\"\n"
            f"created: {_dt.datetime.now().isoformat()}\n"
            f"source: doc_rag_chatbot\n"
            f"tags:\n  - summary\n  - chat\n---\n\n"
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write(frontmatter)
            f.write(content_markdown.strip() + "\n")

        return {"status": "success", "path": path, "filename": filename}
    except Exception as e:  # pragma: no cover
        return {"status": "error", "error_message": str(e), "path": "", "filename": ""}


_MODEL = os.getenv("ADK_MODEL", "gemini-2.5-flash")

summarizer_agent = Agent(
    name="conversation_summarizer_agent",
    model=_MODEL,
    description=(
        "Summarizes the chat into a structured Markdown document and saves it to the downloads folder."
    ),
    instruction=(
        "You are a summarization specialist.\n"
        "Task: Read the conversation context available to you and produce a concise, structured summary.\n"
        "Structure the markdown with clear sections such as: Overview, Key Findings, Citations (use [citation_key | doi] only), Action Items, Sources.\n"
        "Keep it factual and avoid speculation.\n\n"
        "After composing the markdown, CALL save_markdown_to_downloads(title, content_markdown) to persist it.\n"
        "Return a short confirmation including the saved file path."
    ),
)

# Register the file-saving tool
summarizer_agent.tools.append(save_markdown_to_downloads)
