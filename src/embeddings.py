"""Embed validated chunks while keeping their identities and metadata local."""

from collections.abc import Sequence
from hashlib import shake_256
from math import isfinite

from openai import OpenAI, OpenAIError

from .config import AppConfig
from .models import DocumentChunk

BATCH_SIZE = 32
REQUEST_TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 2
SYNTHETIC_MODEL = "synthetic-shake256-v1"
SYNTHETIC_DIMENSIONS = 1536
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class EmbeddingError(RuntimeError):
    """An embedding request or response failed without a usable result."""


def synthetic_vector(text: str) -> list[float]:
    """Generate the same offline vector for a text in ingestion and retrieval."""
    return [
        (byte - 127.5) / 127.5
        for byte in shake_256(text.encode("utf-8")).digest(SYNTHETIC_DIMENSIONS)
    ]


def embed_question(
    question: str, config: AppConfig, *, client: OpenAI | None = None
) -> list[float]:
    """Embed one question using the same SDK settings and validation as chunks."""
    vector = _embed_records([("question", question)], config, client)["question"]
    if not any(vector):
        raise EmbeddingError("Question embedding has zero magnitude")
    return vector


def embed_chunks(
    chunks: Sequence[DocumentChunk],
    config: AppConfig,
    *,
    client: OpenAI | None = None,
) -> dict[str, list[float]]:
    """Return a complete mapping from chunk IDs to embeddings."""
    records = [(chunk.chunk_id, chunk.text) for chunk in chunks]
    if not records:
        return {}
    if len({chunk_id for chunk_id, _ in records}) != len(records):
        raise EmbeddingError("Duplicate chunk IDs in embedding input")
    return _embed_records(records, config, client)


def _embed_records(
    records: list[tuple[str, str]], config: AppConfig, client: OpenAI | None
) -> dict[str, list[float]]:
    dimensions = MODEL_DIMENSIONS.get(config.embedding_model)
    if dimensions is None:
        raise EmbeddingError("Configured embedding model has no known vector dimensions")

    if client is not None:
        return _embed_batches(records, config.embedding_model, dimensions, client)
    if config.openai_api_key is None:
        raise EmbeddingError("OPENAI_API_KEY is required to generate embeddings")
    try:
        with OpenAI(
            api_key=config.openai_api_key.get_secret_value(),
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
        ) as client:
            return _embed_batches(records, config.embedding_model, dimensions, client)
    except OpenAIError:
        raise EmbeddingError("Unable to initialize or close the OpenAI client") from None


def _embed_batches(
    records: list[tuple[str, str]], model: str, dimensions: int, client: OpenAI
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        label = f"Embedding batch {start // BATCH_SIZE + 1}"
        try:
            response = client.embeddings.create(
                model=model, input=[text for _, text in batch], encoding_format="float"
            )
        except OpenAIError as error:
            raise EmbeddingError(f"{label} failed ({type(error).__name__})") from None

        vectors: dict[int, list[float]] = {}
        data = getattr(response, "data", None)
        if not isinstance(data, list):
            raise EmbeddingError(f"{label} returned invalid response data")
        for item in data:
            index = getattr(item, "index", None)
            if type(index) is not int or not 0 <= index < len(batch):
                raise EmbeddingError(f"{label} returned an invalid response index")
            if index in vectors:
                raise EmbeddingError(f"{label} returned a duplicate response index")
            vector = getattr(item, "embedding", None)
            if not isinstance(vector, list) or len(vector) != dimensions:
                raise EmbeddingError(f"{label} expected {dimensions} vector dimensions")
            if not all(type(value) in (float, int) and isfinite(value) for value in vector):
                raise EmbeddingError(f"{label} returned non-numeric or non-finite vector values")
            vectors[index] = vector
        if len(vectors) != len(batch):
            raise EmbeddingError(f"{label} is missing response indices")
        result.update({chunk_id: vectors[index] for index, (chunk_id, _) in enumerate(batch)})
    return result
