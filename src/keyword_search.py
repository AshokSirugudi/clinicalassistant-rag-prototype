"""Rank validated patient-scoped chunks by literal word overlap."""

import re
from pathlib import Path

from .chunking import chunk_document
from .config import PROJECT_ROOT, AppConfig
from .dataset_validation import ValidationReport
from .document_loader import load_documents
from .models import SHARED_PATIENT_ID
from .retrieval import (
    KeywordResult,
    RelevanceResult,
    RetrievalError,
    source_directories,
    validate_source_classification,
)

STOP_WORDS = frozenset(
    "a about an and are as at be by can containing did do does excerpts find for from had has have how "
    "i in is it me of on or please show tell that the their this to was were what when where which who "
    "with would you".split()
)


def keyword_terms(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", text.casefold())) - STOP_WORDS


def retrieve_keywords(
    patient_id: str,
    question: str,
    *,
    config: AppConfig,
    report: ValidationReport,
    project_root: Path = PROJECT_ROOT,
) -> RelevanceResult:
    """Search local validated documents without opening an embedding index."""
    patient_id, question = patient_id.strip(), question.strip()
    if not question:
        raise RetrievalError("Question must not be blank")
    dataset = report.dataset_ready
    if dataset is None:
        raise RetrievalError("Dataset validation is incomplete")
    if patient_id == SHARED_PATIENT_ID or not any(
        row.record.patient_id == patient_id for row in dataset.patients
    ):
        raise RetrievalError("Select a known patient")
    terms = keyword_terms(question)
    if not terms:
        return RelevanceResult("no_matches", (), 0, 0)

    notes, shared = source_directories(config, project_root)
    loaded = load_documents(config, report, project_root=project_root)
    if loaded.errors:
        raise RetrievalError("Document validation failed during keyword search")
    allowed = (patient_id, SHARED_PATIENT_ID)
    matches = []
    try:
        for document in loaded.documents:
            if document.patient_id not in allowed:
                continue
            for chunk in chunk_document(document):
                if chunk.patient_id not in allowed or chunk.patient_id != document.patient_id:
                    raise ValueError
                validate_source_classification(
                    chunk.patient_id, chunk.source_id, chunk.source_metadata, notes, shared,
                )
                score = len(terms & keyword_terms(chunk.text))
                if score:
                    matches.append(KeywordResult(
                        chunk.text, chunk.chunk_id, score, chunk.patient_id, chunk.source_id,
                        chunk.chunk_position, chunk.source_metadata,
                    ))
    except ValueError:
        raise RetrievalError("Keyword search encountered malformed evidence") from None

    matches.sort(key=lambda hit: (-hit.keyword_score, hit.source_id, hit.chunk_position, hit.chunk_id))
    accepted = tuple(matches[:config.top_k])
    return RelevanceResult("potentially_relevant" if accepted else "no_matches", accepted, len(accepted), 0)
