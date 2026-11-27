# Clinical RAG Assistant

I built this prototype around two questions I want to be able to answer in a demo. Which records was the search allowed to see? Where did the excerpt come from? Pick a fictional patient, review their appointment history, and ask a question. The app shows an exact paragraph with a citation when it finds supporting evidence. When it finds none, it says so.

The corpus has 10 fictional patients, 40 appointments, 20 visit notes, and four shared guidance documents. It contains no real patient data. The prototype is for demonstration and technical review only. Do not use it for clinical decisions.

## Run it in Codex Cloud

From the repository root, use Python 3.12. The clean installation was verified with Python 3.12.14.

```bash
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: --index-url https://pypi.org/simple -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Open the app through an authenticated Codex Cloud preview for port `8501`, or use the Cloud desktop browser if your task offers it. **Offline Keyword Search** is the default. It needs no OpenAI key, billing, or ChromaDB index. Start with the [six-scenario demo guide](docs/demo-guide.md).

## How the pieces fit together

This is a modular monolith built as one Python application. Streamlit handles the interface, and the retrieval pipeline is explicit Python.

| Component | Responsibility |
| --- | --- |
| `src/config.py` and `src/models.py` | Resolve workspace paths, load configuration, and define validated records and citation metadata. |
| `src/csv_loader.py` and `src/dataset_validation.py` | Load identifiers as strings, normalize approved fields, and validate records with Pandas and Pydantic. Report duplicates, invalid references, and appointments that require review. |
| `src/document_loader.py` and `src/chunking.py` | Match each visit note to an accepted completed appointment. Assign shared guidance its own scope. Create one chunk per Markdown section with a deterministic ID and citation metadata. |
| `src/ingest.py`, `src/embeddings.py`, and `src/vector_store.py` | Build the optional ChromaDB indexes. Upsert by chunk ID, then reconcile stale IDs after a successful write. |
| `src/keyword_search.py` and `src/retrieval.py` | Rank eligible chunks using literal word overlap or cosine distance, depending on the selected mode. Validate source scope and citations before returning evidence. |
| `src/answering.py` | Accept only complete, verbatim source paragraphs with locally resolved citations. Return an unable-to-answer message when evidence is insufficient. |
| `streamlit_app.py` and `src/streamlit_app.py` | Launch the app, show accepted appointments in date order, and display excerpts with citations. Clear the question and answer when the patient or retrieval mode changes. |

The flow starts with the accepted CSV records. The document loader checks visit-note metadata against those records, and the chunker carries the patient, source path, title, and section forward. Retrieval limits eligibility to the selected patient and `SHARED`. The answer layer checks the scope again and attaches citations from the validated metadata. The optional vector ingestion command prints a data-quality summary before it writes an index.

### Patient isolation and credentials

- Keyword search excludes other patients before scoring. Vector retrieval applies the allowed patient IDs inside the ChromaDB query before top‑k selection.
- Citation validation checks that visit notes point to the selected patient's appointment file and shared guidance points to the shared directory. The UI labels shared guidance separately.
- The patient selector scopes the demo. A real deployment would need authentication and patient-level authorization before retrieval.
- Keep API keys in server-side environment settings. The tracked `.env.example` contains an empty placeholder. Git ignores local `.env` files, virtual environments, caches, and the default generated ChromaDB directories.

## Retrieval modes

| Mode | What to expect |
| --- | --- |
| **Offline Keyword Search** | Ranks validated chunks by literal word overlap and selects a matching source paragraph locally. Synonyms and alternate wording can be missed. |
| **Synthetic Vector Test** | Uses deterministic synthetic vectors to exercise ChromaDB storage, cosine queries, filtering, and citations. The vector rankings carry no semantic meaning. |
| **OpenAI** | Prepared for real embeddings and extractive answer selection. The UI requires explicit server opt-in with `CLINICAL_RAG_ENABLE_OPENAI=1` and a configured `OPENAI_API_KEY`. Live API behavior has not been validated. |

Select **Synthetic Vector Test** and submit a question to build its separate index on first use. The following questions reuse the index. The default output is under `chroma/SYNTHETIC/`. You can also build or refresh it manually.

```bash
.venv/bin/python -m src.ingest --embedding-mode synthetic
```

The OpenAI mode stays disabled for the interview demo. It would require billing, credentials, and a separately built OpenAI index. The optional cosine-distance cutoff applies to vector retrieval.

## Repository layout

```text
streamlit_app.py      Repository-root launch entry point
src/                 Validation, chunking, retrieval, answering, and UI
data/patients/       Fictional patient and appointment CSVs and visit notes
data/shared/         Shared fictional guidance
tests/               Unit tests, Streamlit AppTest, and invalid CSV fixtures
docs/                Decisions, demo questions, and offline evaluation
requirements.txt     Pinned direct dependencies
```

## Run the tests

```bash
.venv/bin/python -m pytest -q
```

The suite uses temporary indexes, synthetic vectors, and fake OpenAI clients. The four-case offline evaluation exercises the actual keyword UI and reports retrieval, citations, isolation, and refusal separately. [Offline evaluation](docs/offline-evaluation.md)

## Streamlit Community Cloud

Choose the GitHub repository and branch in Streamlit Community Cloud. In the app settings, set **Main file path** to **`streamlit_app.py`** at the repository root and select Python 3.12. Community Cloud installs the pinned direct dependencies from `requirements.txt`. Keep the app private for a review and leave OpenAI mode disabled. Offline Keyword Search starts using the checked-in fictional data.

Community Cloud can discard local files when a deployment restarts. The first Synthetic Vector Test question then rebuilds the index. Set `CHROMA_DIR` to a writable local directory if the default path is unavailable. Initialization is coordinated across sessions in one Streamlit worker. Manually refresh a retained index when the source corpus changes. Synthetic rankings can miss passages containing the requested phrase, so use Offline Keyword Search for the demo guide.

## What I would validate next

I would expand the evaluation with alternate phrasings, synonyms, and questions that look plausible but have no supporting passage. I would validate real embeddings separately once live API use is in scope. A verbatim citation establishes that the words appear in the cited passage. It cannot establish that the passage is clinically correct or sufficient for a decision.

For the choices behind the prototype, see the [decision log](docs/decision-log.md).
