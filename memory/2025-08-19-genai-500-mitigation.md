# 2025-08-19 - Mitigation for google.genai 500 INTERNAL in ADK web

- Changed default ADK model from `gemini-2.5-flash-lite` to `gemini-2.5-flash` in both `chatbot/agent.py` and `chatbot/db_agent.py`.
- Rationale: Reduce transient 500 INTERNAL errors observed during `adk web` streaming. Users can still override via `ADK_MODEL` env.
- Next: Consider adding simple retry/backoff in tool calls and/or forcing `response_mime_type="application/json"` when tools are expected to be called.
