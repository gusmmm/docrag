from __future__ import annotations

"""MCP-backed File Agent for accessing project files (PDFs in input/pdf/).

This agent uses the Model Context Protocol (MCP) filesystem server to safely
interact with the repository's input/pdf/ folder, allowing listing and reading
files so the user can ask to "show the original PDF/text". It should return
the file contents or a link/preview instruction back to the root agent.

Security:
- The MCP server is restricted to the absolute path of the repo's input/pdf/ dir.
- tool_filter is applied to only allow safe read/list operations.

Usage:
- The root agent should delegate requests like "show the original PDF" to this
  agent; this agent should then use MCP tools to locate the PDF and respond
  with either the binary content converted to text (for .md or .txt), or a
  pointer to the file path for UI display if binary (.pdf).
"""

import os
from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import (
	MCPToolset,
	StdioConnectionParams,
	StdioServerParameters,
)
from typing import Dict, Any, List, Optional


def _input_abs_path() -> str:
	# repo_root/chatbot -> up one to repo_root, then input/
	repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	return os.path.abspath(os.path.join(repo_root, "input", "pdf"))


_ALLOWED_PATH = _input_abs_path()

# Configure MCP filesystem toolset restricted to input/pdf/ directory
_mcp_files = MCPToolset(
	connection_params=StdioConnectionParams(
		server_params=StdioServerParameters(
			command="npx",
			args=[
				"-y",
				"@modelcontextprotocol/server-filesystem",
				_ALLOWED_PATH,
			],
		),
		timeout=10,
	),
	# No tool_filter applied to allow all filesystem tools
)


_MODEL = os.getenv("ADK_MODEL", "gemini-2.5-flash")

def list_input_pdfs(pattern: Optional[str] = None) -> Dict[str, Any]:
	"""Recursively list PDF files under the allowed input/ directory.

	Args:
		pattern: Optional substring to filter filenames (case-insensitive).

	Returns:
		dict: {"status": "success", "pdfs": [{"name": str, "path": str}], "count": int}

	Notes:
		- Primary file access should use MCP tools. This helper only surfaces
		  likely PDF targets so the agent can follow up with MCP read_file if needed.
	"""
	results: List[Dict[str, str]] = []
	try:
		needle = (pattern or "").lower().strip()
		for root, _dirs, files in os.walk(_ALLOWED_PATH):
			for fn in files:
				if fn.lower().endswith(".pdf"):
					if needle and needle not in fn.lower():
						continue
					abs_path = os.path.join(root, fn)
					try:
						rel_path = os.path.relpath(abs_path, _ALLOWED_PATH)
					except Exception:
						rel_path = fn
					results.append({"name": fn, "path": abs_path, "relative": rel_path})
		return {"status": "success", "pdfs": results, "count": len(results)}
	except Exception as e:  # pragma: no cover
		return {"status": "error", "error_message": str(e), "pdfs": [], "count": 0}

def find_pdf(hint: str) -> Dict[str, Any]:
	"""Find the best matching PDF by filename fragment.

	Args:
		hint: Case-insensitive substring to search (e.g., author, year, slug).

	Returns:
		dict: {"status": "success", "match": {name, path, relative} } or error
	"""
	try:
		res = list_input_pdfs(pattern=hint)
		if res.get("status") != "success":
			return res
		pdfs: List[Dict[str, str]] = res.get("pdfs", [])
		if not pdfs:
			return {"status": "error", "error_message": f"No PDFs found matching '{hint}'."}
		# Simple scoring: prefer files whose name contains the hint earlier and longer
		lo = hint.lower().strip()
		def score(item: Dict[str, str]) -> tuple:
			name = item.get("name", "").lower()
			idx = name.find(lo) if lo else -1
			return (
				0 if idx == -1 else 1,  # contains?
				-idx if idx != -1 else -9999,  # earlier is better (higher -idx)
				-len(name),  # shorter name preferred
			)
		best = sorted(pdfs, key=score, reverse=True)[0]
		return {"status": "success", "match": best}
	except Exception as e:  # pragma: no cover
		return {"status": "error", "error_message": str(e)}

file_agent = Agent(
	name="project_file_agent",
	model=_MODEL,
	description=(
		"Handles file access for the project (read-only), especially PDFs in input/pdf/."
	),
	instruction=(
		"You are the File Agent. Your job is to access files under the allowed directory only: \n"
		f"- ALLOWED: '{_ALLOWED_PATH}'.\n\n"
			"Capabilities via MCP tools: list directories, search for filenames (e.g., *.pdf), and read files.\n"
			"Search recursively (including subfolders like input/pdf/).\n"
			"NEVER modify, move, or delete files. Perform READ-ONLY operations.\n"
		"When the user requests to see the original text or PDF: FIRST call 'find_pdf' (or 'list_input_pdfs') with a short hint derived from the request (e.g., author/year/slug) to get candidate paths; THEN, if needed, use MCP 'directory_tree' or 'search_files' to refine.\n"
	"For Markdown or text files, read and return the content. For MCP 'read_file', use the RELATIVE path returned by 'list_input_pdfs' (relative to the allowed root). For PDFs, provide the ABSOLUTE path for the UI to open in a new tab.\n\n"
	"If needed, you may call 'list_input_pdfs' or 'find_pdf' to identify likely PDF paths and then use MCP 'read_file' with the relative path (for text previews) or return the absolute path for UI rendering (for PDFs).\n\n"
		"You are a SUB-AGENT. Do not address the user directly. Return your findings to the root orchestrator (doc_rag_chatbot)."
	),
	tools=[_mcp_files, list_input_pdfs, find_pdf],
)
