# Project decision log

## Development environment and architecture

Develop, run, and test in Codex Cloud only. Keep the project in the private GitHub repository `AshokSirugudi/clinicalassistant-rag-prototype`.

Use Python 3.12.14 and a modular monolith. Keep the Streamlit interface, ingestion, retrieval, and answer generation in separate modules within one application. Use an explicit RAG pipeline that ingests records and documents, creates embeddings, retrieves evidence, and generates answers with source citations. The pipeline will use ChromaDB and the OpenAI SDK directly, without LangChain or LlamaIndex. Reject questions when the retrieved evidence is insufficient.

## Selected technology and direct dependencies

Keep the approved direct versions pinned in [requirements.txt](../requirements.txt). Pip resolves transitive dependencies at installation time.

| Dependency | Pin | Role |
| --- | --- | --- |
| Streamlit | `streamlit==1.67.0` | Web interface for questions, answers, and citations |
| Pandas | `pandas==3.1.1` | Load and transform tabular patient records |
| Pydantic | `pydantic==2.14.4` | Validate records, metadata, and structured outputs |
| ChromaDB | `chromadb==1.5.9` | Store embeddings and retrieve evidence with metadata filters |
| OpenAI | `openai==3.60.0` | Client for embeddings and answer generation |
| Pytest | `pytest==9.2.0` | Run automated tests |
| python-dotenv | `python-dotenv==1.2.4` | Load environment configuration from a local `.env` file in the cloud workspace |

## Data and patient isolation

Use exclusively fictional patient records in `data/patients/` and shared clinical documents in `data/shared/`. Give each patient-scoped item patient identifier metadata. Require retrieval to apply the selected patient's filter to patient-scoped evidence. Include eligible shared documents while excluding every other patient's evidence from context, answers, and citations.

## Automated cloud setup

Configure Codex Cloud's **Setup script** for the repository. The agreed script creates `.venv` with Python 3.12 when absent, verifies the interpreter version before reuse, installs from `requirements.txt`, and runs `pip check`. Installation and dependency verification must succeed on repeated runs. The repository branch selected in the environment must include `requirements.txt`.

The install commands have been verified in the existing workspace. The environment draft currently has no setup script saved. Automatic execution in a fresh Codex Cloud session remains unverified.

## Verification completed

- Python 3.12.14 ran in `.venv`. All seven installed versions matched the direct pins, all seven packages imported, and `pip check` reported no broken requirements.
- Pytest ran six smoke tests successfully. Pandas filtered and aggregated fictional records. Pydantic accepted valid input and rejected invalid input.
- ChromaDB stored synthetic embeddings in a temporary database and retrieved the selected patient's vector through a metadata filter even when another patient's vector was closer.
- Streamlit's `AppTest` executed a temporary script and verified its rendered output. The OpenAI client imported without API calls. Temporary smoke-test files were removed.

## OpenAI credentials and billing

Defer provisioning OpenAI API credentials and enabling billing until the embeddings milestone. Keep `OPENAI_API_KEY` empty in the tracked template, and keep real credentials out of Git.

## Milestone 8 — Stale-chunk reconciliation

The ingestion command removes stale IDs after upserting the current chunks and verifying that every current ID is present. Synthetic ingestion tests verified that editing one section and deleting a different section removes both stale IDs, retains the replacement and its metadata, and leaves exactly 71 current chunks from the original 72.
