"""Build and reconcile one local Clinical RAG index."""

import argparse
import sys
from dataclasses import dataclass
from hashlib import shake_256
from pathlib import Path
from typing import Literal

from .chunking import chunk_document
from .config import PROJECT_ROOT, AppConfig, load_config
from .csv_loader import load_appointments, load_patients
from .data_quality import format_data_quality_summary
from .dataset_validation import validate_dataset
from .document_loader import load_documents
from .embeddings import EmbeddingError, embed_chunks
from .models import DocumentChunk
from .vector_store import (
    SYNTHETIC_DIMENSIONS,
    SYNTHETIC_MODEL,
    VectorStoreError,
    open_chunk_collection,
    upsert_chunks,
)

type EmbeddingMode = Literal["synthetic", "openai"]


class IngestionError(RuntimeError):
    """An ingestion stage failed and the command cannot report success."""


@dataclass(frozen=True)
class IngestionResult:
    documents: int
    chunks: int
    indexed: int
    stale_deleted: int


def _index_config(config: AppConfig, mode: EmbeddingMode) -> AppConfig:
    if mode == "synthetic":
        return config.model_copy(update={
            "chroma_dir": config.chroma_dir / "SYNTHETIC",
            "embedding_model": SYNTHETIC_MODEL,
            "openai_api_key": None,
        })
    if mode != "openai":
        raise IngestionError("Unknown embedding mode")
    if config.openai_api_key is None:
        raise IngestionError("OpenAI mode requires OPENAI_API_KEY")
    return config.model_copy(update={"chroma_dir": config.chroma_dir / "openai"})


def _synthetic_embeddings(chunks: list[DocumentChunk]) -> dict[str, list[float]]:
    return {
        chunk.chunk_id: [
            (byte - 127.5) / 127.5
            for byte in shake_256(chunk.text.encode("utf-8")).digest(SYNTHETIC_DIMENSIONS)
        ]
        for chunk in chunks
    }


def _reconcile_ids(config: AppConfig, expected: set[str]) -> int:
    collection, batch_size = open_chunk_collection(config)
    try:
        stored = set(collection.get(include=[])["ids"])
    except Exception:
        raise VectorStoreError("Unable to read indexed IDs. No stale IDs were deleted.") from None
    if not expected <= stored:
        raise VectorStoreError("Some current chunk IDs are missing. No stale IDs were deleted.")

    stale = sorted(stored - expected)
    try:
        for start in range(0, len(stale), batch_size):
            collection.delete(ids=stale[start:start + batch_size])
        verified = set(collection.get(include=[])["ids"])
    except Exception:
        raise VectorStoreError("Unable to complete stale-ID reconciliation. The index may be partial or stale.") from None
    if verified != expected:
        raise VectorStoreError("Stored IDs do not match the current corpus after reconciliation")
    return len(stale)


def ingest(
    config: AppConfig, mode: EmbeddingMode, *, project_root: Path = PROJECT_ROOT
) -> IngestionResult:
    """Run the pipeline and reconcile the selected mode's index after successful writes."""
    index_config = _index_config(config, mode)
    if mode == "synthetic":
        print("Embedding mode: SYNTHETIC (offline vectors for testing)")
    else:
        print("Embedding mode: OPENAI")

    patients = load_patients(config.patients_csv)
    appointments = load_appointments(config.appointments_csv)
    report = validate_dataset(patients, appointments)
    print(format_data_quality_summary(patients, appointments, report))
    if report.incomplete:
        for issue in report.file_errors:
            location = f"{issue.filename}:{issue.line_number}" if issue.line_number else issue.filename
            print(f"{location}: {issue.message}", file=sys.stderr)
        raise IngestionError("CSV validation is incomplete. The index was not opened.")

    loaded = load_documents(config, report, project_root=project_root)
    if loaded.errors:
        for issue in loaded.errors:
            print(f"{issue.source_path or 'Documents'}: {issue.message}", file=sys.stderr)
        raise IngestionError("Document validation failed. The index was not opened.")

    chunks = []
    for document in loaded.documents:
        try:
            chunks.extend(chunk_document(document))
        except ValueError:
            raise IngestionError(f"Chunking failed for {document.source_path}") from None
    print(f"Documents: {len(loaded.documents)} | Chunks: {len(chunks)}")
    if not chunks:
        raise IngestionError("No chunks were produced. The index was not opened.")

    try:
        vectors = _synthetic_embeddings(chunks) if mode == "synthetic" else embed_chunks(chunks, config)
    except EmbeddingError as error:
        raise IngestionError(f"Embedding failed. {error}") from None

    try:
        indexed = upsert_chunks(index_config, chunks, vectors)
        stale_deleted = _reconcile_ids(index_config, {chunk.chunk_id for chunk in chunks})
    except VectorStoreError as error:
        raise IngestionError(f"Indexing failed. {error}") from None

    print(f"Indexed: {indexed} | Stale deletions: {stale_deleted} | Errors: 0")
    return IngestionResult(len(loaded.documents), len(chunks), indexed, stale_deleted)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-mode", choices=("synthetic", "openai"), required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config()
    except (OSError, ValueError):
        print("Ingestion failed. Check application configuration.", file=sys.stderr)
        return 1
    try:
        ingest(config, args.embedding_mode)
    except IngestionError as error:
        print(f"Ingestion failed. {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
