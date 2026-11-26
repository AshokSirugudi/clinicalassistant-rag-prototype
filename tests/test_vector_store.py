import json
import subprocess
import sys

import chromadb
import pytest
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection

from src.chunking import chunk_document
from src.config import AppConfig
from src.csv_loader import load_appointments, load_patients
from src.dataset_validation import validate_dataset
from src.document_loader import load_documents
from src.vector_store import COLLECTION_NAME, VectorStoreError, upsert_chunks


def open_client(path):
    return chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))


def vector(x, y, z):
    return [x, y, z] + [0.0] * 1533


@pytest.fixture(scope="module")
def chunks():
    config = AppConfig()
    report = validate_dataset(load_patients(config.patients_csv), load_appointments(config.appointments_csv))
    result = load_documents(config, report)
    assert not result.errors
    return [chunk for doc in result.documents for chunk in chunk_document(doc)]


def test_persistence_idempotence_and_patient_filtering(tmp_path, chunks):
    config = AppConfig(chroma_dir=tmp_path / "chroma")
    by_section = {(c.source_id, c.source_metadata.section_heading): c for c in chunks}
    knee = by_section["A001", "Recommendations"]
    shared = by_section["data/shared/activity-related-discomfort.md", "Tracking Activity and Function"]
    embeddings = {
        chunk.chunk_id: vector(0.0, 0.0, 1.0) if chunk.patient_id in {"P001", "SHARED"}
        else vector(1.0, 0.0, 0.0)
        for chunk in chunks
    }
    embeddings[knee.chunk_id] = vector(0.8, 0.6, 0.0)
    embeddings[shared.chunk_id] = vector(0.6, 0.8, 0.0)

    assert upsert_chunks(config, chunks, embeddings) == 72
    assert upsert_chunks(config, chunks, embeddings) == 72
    collection = open_client(config.chroma_dir).get_collection(COLLECTION_NAME, embedding_function=None)
    assert collection.count() == 72
    reopened = subprocess.run(
        [sys.executable, "-c", """
import json
import sys

import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(sys.argv[1], settings=Settings(anonymized_telemetry=False))
collection = client.get_collection(sys.argv[2], embedding_function=None)
query = [[1.0, 0.0, 0.0] + [0.0] * 1533]
scope = {"patient_id": {"$in": ["P001", "SHARED"]}}
print(json.dumps({
    "count": collection.count(),
    "unfiltered": collection.query(query_embeddings=query, n_results=1)["ids"],
    "filtered": collection.query(query_embeddings=query, where=scope, n_results=2)["ids"],
    "eligible": collection.get(where=scope, include=["metadatas"])["metadatas"],
}))
""",
         str(config.chroma_dir), COLLECTION_NAME],
        check=True, capture_output=True, text=True,
    )
    persisted = json.loads(reopened.stdout)
    assert persisted["count"] == 72
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    assert by_id[persisted["unfiltered"][0][0]].patient_id not in {"P001", "SHARED"}
    filtered, = persisted["filtered"]
    assert len(filtered) == 2
    assert all(by_id[chunk_id].patient_id in {"P001", "SHARED"} for chunk_id in filtered)
    assert len(persisted["eligible"]) == 18
    assert {metadata["patient_id"] for metadata in persisted["eligible"]} == {"P001", "SHARED"}

    assert collection.configuration["hnsw"]["space"] == "cosine"
    assert collection.configuration["embedding_function"] is None
    saved = collection.get(include=["documents", "embeddings", "metadatas"])
    for chunk_id, text, embedding, metadata in zip(
        saved["ids"], saved["documents"], saved["embeddings"], saved["metadatas"], strict=True
    ):
        assert text == by_id[chunk_id].text
        assert metadata == by_id[chunk_id].to_chroma_metadata()
        assert embedding.tolist() == pytest.approx(embeddings[chunk_id])


def test_synthetic_cosine_ranking_with_patient_filter(tmp_path, chunks):
    config = AppConfig(chroma_dir=tmp_path / "chroma")
    by_section = {(chunk.source_id, chunk.source_metadata.section_heading): chunk for chunk in chunks}
    other_patient = by_section["A034", "Recommendations"]
    knee = by_section["A001", "Recommendations"]
    activity = by_section["data/shared/activity-related-discomfort.md", "Tracking Activity and Function"]
    blood_pressure = by_section["data/shared/home-blood-pressure-log.md", "What to Record"]
    ranked = [other_patient, knee, activity, blood_pressure]
    embeddings = {
        other_patient.chunk_id: vector(1.0, 0.0, 0.0),
        knee.chunk_id: vector(0.8, 0.6, 0.0),
        activity.chunk_id: vector(0.6, 0.8, 0.0),
        blood_pressure.chunk_id: vector(0.0, 0.0, 1.0),
    }
    assert upsert_chunks(config, list(reversed(ranked)), embeddings) == 4
    collection = open_client(config.chroma_dir).get_collection(COLLECTION_NAME, embedding_function=None)
    query = [vector(1.0, 0.0, 0.0)]

    unfiltered = collection.query(query_embeddings=query, n_results=4)
    assert unfiltered["ids"] == [[chunk.chunk_id for chunk in ranked]]
    assert unfiltered["distances"][0] == pytest.approx([0.0, 0.2, 0.4, 1.0])

    scope = {"patient_id": {"$in": ["P001", "SHARED"]}}
    eligible = collection.get(where=scope, include=["metadatas"])
    assert set(eligible["ids"]) == {knee.chunk_id, activity.chunk_id, blood_pressure.chunk_id}
    filtered = collection.query(
        query_embeddings=query, where=scope, n_results=2, include=["distances", "metadatas"]
    )
    assert filtered["ids"] == [[knee.chunk_id, activity.chunk_id]]
    assert other_patient.chunk_id not in filtered["ids"][0]
    assert filtered["distances"][0] == pytest.approx([0.2, 0.4])
    assert filtered["metadatas"] == [[knee.to_chroma_metadata(), activity.to_chroma_metadata()]]


def test_only_new_or_changed_records_are_upserted(tmp_path, chunks, monkeypatch):
    config = AppConfig(chroma_dir=tmp_path / "chroma")
    chunk = chunks[0]
    upsert = Collection.upsert
    submitted = []

    def track(collection, **kwargs):
        submitted.append(kwargs["ids"])
        return upsert(collection, **kwargs)

    monkeypatch.setattr(Collection, "upsert", track)
    original = {chunk.chunk_id: vector(1.0, 0.0, 0.0)}
    changed = {chunk.chunk_id: vector(0.0, 1.0, 0.0)}
    for embeddings in (original, original, changed):
        assert upsert_chunks(config, [chunk], embeddings) == 1
    assert submitted == [[chunk.chunk_id], [chunk.chunk_id]]
    collection = open_client(config.chroma_dir).get_collection(COLLECTION_NAME, embedding_function=None)
    assert collection.count() == 1
    saved = collection.get(ids=[chunk.chunk_id], include=["embeddings", "metadatas"])
    assert saved["ids"] == [chunk.chunk_id]
    assert saved["embeddings"][0].tolist() == changed[chunk.chunk_id]
    assert saved["metadatas"] == [chunk.to_chroma_metadata()]


@pytest.mark.parametrize(("case", "message"), [
    ("missing", "exactly the supplied chunk IDs"),
    ("extra", "exactly the supplied chunk IDs"),
    ("duplicate", "Duplicate chunk IDs"),
    ("dimensions", "Expected 1536 vector dimensions"),
    ("nonfinite", "only finite numbers"),
    ("zero", "Zero vectors"),
])
def test_invalid_input_does_not_open_storage(tmp_path, chunks, case, message):
    config = AppConfig(chroma_dir=tmp_path / "chroma")
    chunk = chunks[0]
    batch = [chunk, chunk] if case == "duplicate" else [chunk]
    embeddings = {chunk.chunk_id: vector(1.0, 0.0, 0.0)}
    if case == "missing":
        embeddings.clear()
    elif case == "extra":
        embeddings["extra"] = vector(1.0, 0.0, 0.0)
    elif case == "dimensions":
        embeddings[chunk.chunk_id] = [1.0, 0.0]
    elif case == "nonfinite":
        embeddings[chunk.chunk_id] = vector(float("nan"), 0.0, 0.0)
    elif case == "zero":
        embeddings[chunk.chunk_id] = vector(0.0, 0.0, 0.0)
    with pytest.raises(VectorStoreError, match=message):
        upsert_chunks(config, batch, embeddings)
    assert not config.chroma_dir.exists()


@pytest.mark.parametrize(("metric", "model", "dimensions", "message"), [
    ("l2", "text-embedding-3-small", 1536, "cosine distance"),
    ("cosine", "text-embedding-ada-002", 1536, "model or dimensions"),
    ("cosine", "text-embedding-3-small", 3, "model or dimensions"),
])
def test_existing_collection_configuration_is_verified(
    tmp_path, chunks, metric, model, dimensions, message
):
    config = AppConfig(chroma_dir=tmp_path / "chroma")
    client = open_client(config.chroma_dir)
    collection = client.create_collection(
        COLLECTION_NAME, configuration={"hnsw": {"space": metric}},
        metadata={"embedding_model": model, "embedding_dimensions": dimensions},
        embedding_function=None,
    )
    chunk = chunks[0]
    with pytest.raises(VectorStoreError, match=message):
        upsert_chunks(config, [chunk], {chunk.chunk_id: vector(1.0, 0.0, 0.0)})
    assert collection.count() == 0


def test_storage_error_omits_underlying_details(tmp_path, chunks, monkeypatch):
    config = AppConfig(chroma_dir=tmp_path / "chroma")
    chunk = chunks[0]

    def fail(**kwargs):
        raise OSError("dummy-key " + chunk.text)

    monkeypatch.setattr("src.vector_store.chromadb.PersistentClient", fail)
    with pytest.raises(VectorStoreError, match="Unable to open") as error:
        upsert_chunks(config, [chunk], {chunk.chunk_id: vector(1.0, 0.0, 0.0)})
    assert "dummy-key" not in str(error.value)
    assert chunk.text not in str(error.value)
    assert error.value.__suppress_context__
