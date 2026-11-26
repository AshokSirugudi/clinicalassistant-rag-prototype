# Offline evaluation | Phase 1, Step 14.2

- Evaluated GitHub `main` at `2b295e21d3b2dda8ac600423414fb18c3662391e` with the local evaluation tests.
- Mode: **Offline Keyword Search**, with `top_k=5`.
- Corpus: 10 fictional patients, 40 appointments, 20 visit notes, and four shared guidance documents.
- Execution: Streamlit AppTest submitted each exact question once. The test captured the lexical retrieval result separately from the saved answer and rendered UI. It verified that no vector index was created. No OpenAI API calls were made.

## Results

| Case | Patient and question | Retrieval quality | Citation accuracy | Patient isolation | Answer behavior | Observed output |
| --- | --- | --- | --- | --- | --- | --- |
| 1. Knee symptoms | `P001` · What does the note say about knee discomfort and limits on usual activities? | PASS | PASS | PASS | PASS | `A001.md` · Symptoms · “Activity-related knee discomfort with some limits on usual activities.” |
| 2. Shared headache diary | `P003` · For a headache diary, what should I note about how long each headache lasted, how intense it felt, and which activities it interrupted? | PASS | PASS | PASS | PASS | `headache-diary.md` · What to Record · “For each headache, note when it started, how long it lasted, how intense it felt, and which activities it interrupted.” |
| 3. Footwear match | `P010` · What about footwear? | PASS | PASS | PASS | PASS | `A037.md` · Symptoms · “The visit was a routine foot examination. The patient raised footwear concerns.” |
| 4. Footwear isolation control | `P001` · What about footwear? | PASS | PASS | PASS | PASS | `unable_to_answer` with reason `no_matches`. No excerpts or citations were returned or displayed. |

**Overall result: 4 cases passed, 0 failed. All 16 dimension checks passed.** The full Pytest suite passed 257 tests. It emitted 108 ChromaDB embedding-function deprecation warnings from the existing vector tests. There were no evaluation failures.

Each positive case had the expected section in the top five, and the exact displayed excerpt matched its source paragraph. The displayed title, path, section, and applicable date, clinician, and appointment ID matched the source metadata. Every retrieved or cited item belonged to the selected patient or `SHARED`. The identical footwear question returned no evidence for `P001` and cited the expected visit for `P010`.

## Reproduce

Run from the repository root in the existing Codex Cloud virtual environment. The evaluation fixture configures the offline mode and captures both retrieval and answer outcomes. [Evaluation tests](../tests/test_offline_evaluation.py)

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -v -p no:cacheprovider tests/test_offline_evaluation.py
```

These four cases verify the approved excerpts, citations, and refusal on a fixed fictional dataset. Literal word matching can miss synonyms or retrieve weak matches. This evaluation does not establish semantic retrieval quality, evidence sufficiency for other questions, or clinical correctness.
