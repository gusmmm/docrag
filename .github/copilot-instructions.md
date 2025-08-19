# Copilot Instructions
- ALWAYS check the latest documentation online for any updates or changes. Your information is outdated.
- ALWAYS check the memory/ folder for the latest information about the project, what has been implemented, and any other relevant details.

# Memory
- ALWAYS check the memory/ folder for the latest information about the project, what has been implemented, and any other relevant details.
- After significant changes update the memory at the memory/ folder

# General Guidelines
- ALWAYS use uv to manage all python packages, dependencies, environments, and scripts

# using OpenRouter
- whenever openAI-style JSON schema is required, use the openrouter service

# using google genai agents
- follow the latest documentation for using Google GenAI agents. See the latest documentation at https://googleapis.github.io/python-genai/
- the llm agents and clients are from google genai. Use by default model="gemini-2.5-flash-lite". The other parameters can be found in the documentation. For more complex tasks use "gemini-2.5-flash", and for the most advanced tasks use "gemini-2.5-pro".
- the embedding model is from gemini. The model="gemini-embedding-001". Check the latest documentation for more details at https://ai.google.dev/gemini-api/docs/embeddings?hl=en

# docling
- this project uses docling to create a multi-modal RAG. The online documentation is at https://docling-project.github.io/docling/
- first we will process pdf files into md format. The pdf files will be in the input/ folder
- the vector database is milvus. Check the latest documentation for more details at https://milvus.io/docs/pt/build_RAG_with_milvus_and_docling.md
