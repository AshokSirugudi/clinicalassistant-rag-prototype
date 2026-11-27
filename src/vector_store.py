"""Persist explicitly embedded chunks in the local ChromaDB collection.

Upsert replaces rows with the same chunk IDs. The ingestion pipeline owns
reconciling stale IDs when source documents change or are removed.
"""

from collections.abc import Mapping, Sequence
from math import isfinite
from struct import Struct, error as StructError
from typing import Literal

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from .config import AppConfig
from .embeddings import MODEL_DIMENSIONS, SYNTHETIC_DIMENSIONS, SYNTHETIC_MODEL
from .models import DocumentChunk

COLLECTION_NAME = "clinical_chunks_v1"
type EmbeddingMode = Literal["synthetic", "openai"]


class VectorStoreError(RuntimeError):
    """Vector store inputs, configuration, or writes failed."""


class MissingVectorIndexError(VectorStoreError):
    """The configured index or collection has not been created."""


def index_config(config: AppConfig, mode: EmbeddingMode) -> AppConfig:
    """Resolve the index location and model identity for an embedding mode."""
    if mode == "synthetic":
        return config.model_copy(update={
            "chroma_dir": config.chroma_dir / "SYNTHETIC",
            "embedding_model": SYNTHETIC_MODEL,
            "openai_api_key": None,
        })
    if mode == "openai":
        return config.model_copy(update={"chroma_dir": config.chroma_dir / "openai"})
    raise VectorStoreError("Unknown embedding mode")


def _vector_dimensions(model: str) -> int:
    if model == SYNTHETIC_MODEL:
        return SYNTHETIC_DIMENSIONS
    dimensions = MODEL_DIMENSIONS.get(model)
    if dimensions is None:
        raise VectorStoreError("Configured embedding model has no known vector dimensions")
    return dimensions


def open_chunk_collection(config: AppConfig, *, create: bool = True):
    """Open the configured collection and verify its compatibility before use."""
    expected_metadata = {
        "embedding_model": config.embedding_model,
        "embedding_dimensions": _vector_dimensions(config.embedding_model),
    }
    if not create and not (config.chroma_dir / "chroma.sqlite3").is_file():
        raise MissingVectorIndexError("ChromaDB index does not exist for the selected mode")
    try:
        client = chromadb.PersistentClient(
            path=config.chroma_dir, settings=Settings(anonymized_telemetry=False)
        )
        if create:
            collection = client.get_or_create_collection(
                COLLECTION_NAME,
                configuration={"hnsw": {"space": "cosine"}},
                metadata=expected_metadata,
                embedding_function=None,
            )
        else:
            collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
        settings = collection.configuration
        metadata = collection.metadata or {}
        max_batch_size = client.get_max_batch_size()
    except NotFoundError:
        raise MissingVectorIndexError("ChromaDB collection does not exist for the selected mode") from None
    except Exception:
        raise VectorStoreError("Unable to open the ChromaDB collection") from None

    if (settings.get("hnsw") or {}).get("space") != "cosine":
        raise VectorStoreError("ChromaDB collection must use HNSW cosine distance")
    if settings.get("embedding_function") is not None:
        raise VectorStoreError("ChromaDB collection must have automatic embeddings disabled")
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise VectorStoreError("ChromaDB collection embedding model or dimensions do not match")
    return collection, max_batch_size


def upsert_chunks(
    config: AppConfig,
    chunks: Sequence[DocumentChunk],
    embeddings: Mapping[str, Sequence[float]],
) -> int:
    """Upsert new or changed chunks, returning the count successfully processed."""
    dimensions = _vector_dimensions(config.embedding_model)

    ids = [chunk.chunk_id for chunk in chunks]
    unique_ids = set(ids)
    if len(unique_ids) != len(ids):
        raise VectorStoreError("Duplicate chunk IDs in storage input")
    if embeddings.keys() != unique_ids:
        raise VectorStoreError("Embedding mapping must contain exactly the supplied chunk IDs")

    vectors = []
    packed_vectors = []
    float32 = Struct(f"<{dimensions}f")
    for chunk_id in ids:
        vector = embeddings[chunk_id]
        if len(vector) != dimensions:
            raise VectorStoreError(f"Expected {dimensions} vector dimensions")
        if not all(type(value) in (int, float) for value in vector):
            raise VectorStoreError("Embedding vectors must contain only finite numbers")
        try:
            packed = float32.pack(*vector)
        except (OverflowError, StructError):
            raise VectorStoreError("Embedding values must fit in ChromaDB's float32 format") from None
        stored_values = float32.unpack(packed)
        if not all(isfinite(value) for value in stored_values):
            raise VectorStoreError("Embedding vectors must contain only finite numbers")
        if not any(stored_values):
            raise VectorStoreError("Zero vectors cannot be stored for cosine distance")
        vectors.append(list(stored_values))
        packed_vectors.append(packed)
    if not ids:
        return 0

    collection, max_batch_size = open_chunk_collection(config)

    documents = [chunk.text for chunk in chunks]
    metadatas = [chunk.to_chroma_metadata() for chunk in chunks]
    written = 0
    for start in range(0, len(ids), max_batch_size):
        end = min(start + max_batch_size, len(ids))
        try:
            saved = collection.get(
                ids=ids[start:end], include=["documents", "embeddings", "metadatas"]
            )
            existing = {
                chunk_id: (text, float32.pack(*vector), metadata)
                for chunk_id, text, vector, metadata in zip(
                    saved["ids"], saved["documents"], saved["embeddings"],
                    saved["metadatas"], strict=True,
                )
            } if saved["ids"] else {}
            changed = [
                i for i in range(start, end)
                if existing.get(ids[i]) != (documents[i], packed_vectors[i], metadatas[i])
            ]
            if changed:
                collection.upsert(
                    ids=[ids[i] for i in changed],
                    documents=[documents[i] for i in changed],
                    embeddings=[vectors[i] for i in changed],
                    metadatas=[metadatas[i] for i in changed],
                )
                written += len(changed)
        except Exception:
            raise VectorStoreError(
                f"ChromaDB storage operation failed after {written} confirmed writes. The write may be partial."
            ) from None
    return len(ids)
