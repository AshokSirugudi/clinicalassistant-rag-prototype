"""Retrieve citation-ready evidence from a patient-scoped ChromaDB query."""

from collections.abc import Collection
from dataclasses import dataclass
from math import isfinite
from pathlib import PurePosixPath

from .config import AppConfig
from .embeddings import EmbeddingError, embed_question, synthetic_vector
from .models import SHARED_PATIENT_ID, DocumentSourceMetadata
from .vector_store import EmbeddingMode, VectorStoreError, index_config, open_chunk_collection


class RetrievalError(RuntimeError):
    """A request or retrieval response is invalid."""


class RetrievalIndexError(RetrievalError):
    """The selected index is unavailable or incompatible."""


class RetrievalEmbeddingError(RetrievalError):
    """The question could not be embedded."""


@dataclass(frozen=True)
class RetrievalResult:
    text: str
    chunk_id: str
    distance: float
    patient_id: str
    source_id: str
    chunk_position: int
    source_metadata: DocumentSourceMetadata


def retrieve(
    patient_id: str,
    question: str,
    *,
    config: AppConfig,
    mode: EmbeddingMode,
    accepted_patient_ids: Collection[str],
) -> list[RetrievalResult]:
    """Return scoped evidence in ascending distance order, or an empty list for no matches."""
    patient_id, question = patient_id.strip(), question.strip()
    if not question:
        raise RetrievalError("Question must not be blank")
    if not patient_id or patient_id == SHARED_PATIENT_ID or patient_id not in accepted_patient_ids:
        raise RetrievalError("Select a known patient")
    if mode not in ("synthetic", "openai"):
        raise RetrievalError("Unknown embedding mode")

    try:
        collection, _ = open_chunk_collection(index_config(config, mode), create=False)
    except VectorStoreError as error:
        raise RetrievalIndexError(str(error)) from None
    try:
        vector = synthetic_vector(question) if mode == "synthetic" else embed_question(question, config)
    except EmbeddingError as error:
        raise RetrievalEmbeddingError(str(error)) from None

    allowed = [patient_id, SHARED_PATIENT_ID]
    try:
        response = collection.query(
            query_embeddings=[vector],
            where={"patient_id": {"$in": allowed}},
            n_results=config.top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        raise RetrievalIndexError("Unable to query the selected ChromaDB index") from None

    try:
        ids, = response["ids"]
        texts, = response["documents"]
        distances, = response["distances"]
        metadatas, = response["metadatas"]
        if len(set(ids)) != len(ids) or len(ids) > config.top_k:
            raise ValueError
        results = [
            _checked_result(chunk_id, text, distance, metadata, allowed)
            for chunk_id, text, distance, metadata in zip(ids, texts, distances, metadatas, strict=True)
        ]
    except (KeyError, TypeError, ValueError, OverflowError):
        raise RetrievalError("ChromaDB returned malformed evidence or citation metadata") from None
    return sorted(results, key=lambda result: result.distance)


def _checked_result(chunk_id, text, distance, metadata, allowed: list[str]) -> RetrievalResult:
    patient_id = metadata["patient_id"]
    if patient_id not in allowed:
        raise RetrievalError("ChromaDB returned evidence outside the selected patient scope")
    if (
        not isinstance(chunk_id, str) or not chunk_id.strip() or chunk_id != metadata["chunk_id"]
        or not isinstance(text, str) or not text.strip()
        or type(distance) not in (float, int) or not isfinite(distance)
        or type(metadata["chunk_position"]) is not int or metadata["chunk_position"] < 0
    ):
        raise ValueError

    citation = DocumentSourceMetadata.model_validate({
        key: value for key, value in metadata.items() if key in DocumentSourceMetadata.model_fields
    })
    source_id = metadata["source_id"]
    path = PurePosixPath(citation.source_path)
    if path.is_absolute() or ".." in path.parts or "\\" in citation.source_path or path.suffix != ".md":
        raise ValueError
    shared = patient_id == SHARED_PATIENT_ID
    if shared != (citation.source_type == "shared_guidance"):
        raise ValueError
    if shared:
        if source_id != citation.source_path:
            raise ValueError
    elif source_id != citation.appointment_id or path.parts[-2:] != (patient_id, f"{source_id}.md"):
        raise ValueError
    return RetrievalResult(text, chunk_id, float(distance), patient_id, source_id, metadata["chunk_position"], citation)
