from __future__ import annotations

"""ADK sub-agent for internet search via google_search tool.

This agent accepts general web queries from the root chatbot and returns a concise
answer summarizing the most relevant search results, including URLs.

It uses ADK's built-in `google_search` tool:
https://google.github.io/adk-docs/tools/built-in-tools/

It also leverages session state to remember prior search context within a session,
so follow-up questions can refine prior results.
"""

from typing import Any, Dict, List
from google.adk.agents import Agent
from google.adk.tools import google_search


# Note: Built-in tools (like google_search) cannot be used within sub-agents per ADK limitations.
# Use this agent as a root agent (separate app) or wrap it via agent_tool at the root level only.
internet_search_agent = Agent(
	name="internet_search_agent",
	model="gemini-2.0-flash",
	description=(
		"Web search agent that uses the google_search tool to answer questions from the internet."
	),
	instruction=(
		"You are an internet research assistant. Always call google_search with the user's query, "
		"review the top results, and synthesize a concise answer with 2-4 key findings. Include titles and URLs. "
		"Prefer reputable sources (e.g., journals, docs, standards).\n\n"
		"State and memory: maintain a brief note in session state under 'web_context' about what was searched "
		"and which sources looked promising. If the user asks a follow-up, consult 'web_context' to refine the search or reuse relevant links."
	),
)

# Register the built-in google_search tool
internet_search_agent.tools.append(google_search)
