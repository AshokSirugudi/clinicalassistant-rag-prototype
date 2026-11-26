## Project objective

Build a Clinical RAG Assistant using fictional patient records and shared clinical documents.

## Architecture and technology

- Modular monolith architecture.
- Python 3.12, Streamlit, Pandas, Pydantic, ChromaDB, OpenAI API and Pytest.
- Explicit RAG pipeline without LangChain or LlamaIndex.
- Cloud-based development and execution only.

## Data and safety

- Use fictional patient data exclusively.
- Enforce patient-specific metadata filtering.
- Prevent cross-patient data leakage.
- Provide source citations for generated answers.
- Reject questions when supporting evidence is insufficient.
- Never commit API keys or credentials.

## Development methodology

- Follow incremental, milestone-based development.
- Explain concepts and design alternatives before implementation.
- Obtain user approval before implementing each step.
- Keep code modular, readable and maintainable.
- Verify functionality through appropriate tests.
- Do not introduce unnecessary dependencies or architectural complexity.
- Do not change the agreed scope without approval.
