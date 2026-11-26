from types import SimpleNamespace

import chromadb
import pytest
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from openai import OpenAIError

from src.chunking import chunk_document
from src.config import AppConfig
from src.csv_loader import load_appointments, load_patients
from src.dataset_validation import validate_dataset
from src.document_loader import load_documents
from src.embeddings import SYNTHETIC_MODEL, synthetic_vector
from src.ingest import _synthetic_embeddings
from src.retrieval import RetrievalEmbeddingError, RetrievalError, RetrievalIndexError, retrieve
from src.vector_store import COLLECTION_NAME, index_config, open_chunk_collection, upsert_chunks

PATIENT_IDS = {"P001", "P002", "P009"}
QUESTION = "What was recommended for knee discomfort?"


@pytest.fixture(autouse=True)
def forbid_live_openai(monkeypatch):
    def unexpected_client(**kwargs):
        pytest.fail("Retrieval tests must use a fake OpenAI client")

    monkeypatch.setattr("src.embeddings.OpenAI", unexpected_client)


@pytest.fixture
def config(tmp_path):
    return AppConfig(chroma_dir=tmp_path / "chroma", top_k=2)


@pytest.fixture(scope="module")
def evidence():
    config = AppConfig()
    report = validate_dataset(load_patients(config.patients_csv), load_appointments(config.appointments_csv))
    documents = load_documents(config, report)
    assert not documents.errors
    chunks = {
        (chunk.source_id, chunk.source_metadata.section_heading): chunk
        for document in documents.documents for chunk in chunk_document(document)
    }
    return (
        chunks["A001", "Recommendations"],
        chunks["data/shared/activity-related-discomfort.md", "Tracking Activity and Function"],
        chunks["A034", "Recommendations"],
    )


def vector(x, y, z):
    return [x, y, z] + [0.0] * 1533


def search(config, patient_id="P001", question=QUESTION, mode="synthetic"):
    return retrieve(patient_id, question, config=config, mode=mode, accepted_patient_ids=PATIENT_IDS)


def test_query_filters_before_top_k_and_returns_citations(config, evidence, monkeypatch):
    own, shared, other = evidence
    index = index_config(config, "synthetic")
    upsert_chunks(index, evidence, {
        own.chunk_id: vector(0.8, 0.6, 0.0),
        shared.chunk_id: vector(0.6, 0.8, 0.0),
        other.chunk_id: vector(1.0, 0.0, 0.0),
    })
    query_vector = vector(1.0, 0.0, 0.0)
    collection, _ = open_chunk_collection(index, create=False)
    assert collection.query(query_embeddings=[query_vector], n_results=1)["ids"] == [[other.chunk_id]]
    monkeypatch.setattr("src.retrieval.synthetic_vector", lambda question: query_vector)
    query = Collection.query
    calls = []

    def record_query(self, **kwargs):
        calls.append(kwargs)
        return query(self, **kwargs)

    monkeypatch.setattr(Collection, "query", record_query)
    results = search(config, patient_id=" P001 ", question=f" {QUESTION} ")
    assert calls == [{
        "query_embeddings": [query_vector],
        "where": {"patient_id": {"$in": ["P001", "SHARED"]}},
        "n_results": 2,
        "include": ["documents", "metadatas", "distances"],
    }]
    assert [hit.patient_id for hit in results] == ["P001", "SHARED"]
    assert [hit.distance for hit in results] == pytest.approx([0.2, 0.4])
    for hit, chunk in zip(results, (own, shared), strict=True):
        assert (hit.chunk_id, hit.text, hit.source_id, hit.chunk_position) == (
            chunk.chunk_id, chunk.text, chunk.source_id, chunk.chunk_position
        )
        assert hit.source_metadata == chunk.source_metadata
    assert [hit.patient_id for hit in search(config, patient_id="P002")] == ["SHARED"]


def test_synthetic_query_matches_ingestion_algorithm(config, evidence):
    chunk = evidence[0]
    vectors = _synthetic_embeddings([chunk])
    assert vectors[chunk.chunk_id] == synthetic_vector(chunk.text)
    upsert_chunks(index_config(config, "synthetic"), [chunk], vectors)
    hit, = search(config, question=chunk.text.strip())
    assert hit.chunk_id == chunk.chunk_id


@pytest.mark.parametrize(("patient_id", "question", "message"), [
    ("P001", " \t", "Question must not be blank"),
    ("P999", QUESTION, "known patient"),
    ("SHARED", QUESTION, "known patient"),
    ("p001", QUESTION, "known patient"),
])
def test_invalid_inputs_stop_before_index_access(config, patient_id, question, message):
    with pytest.raises(RetrievalError, match=message) as error:
        retrieve(patient_id, question, config=config, mode="synthetic",
                 accepted_patient_ids=PATIENT_IDS | {"SHARED"})
    assert type(error.value) is RetrievalError
    assert not config.chroma_dir.exists()


def test_no_matches_is_an_empty_list(config, evidence):
    other = evidence[2]
    upsert_chunks(index_config(config, "synthetic"), [other], {other.chunk_id: vector(1.0, 0.0, 0.0)})
    assert search(config) == []


def test_missing_index_is_not_created(config):
    with pytest.raises(RetrievalIndexError, match="index does not exist"):
        search(config)
    assert not config.chroma_dir.exists()


def test_missing_collection_is_not_created(config):
    index = index_config(config, "synthetic")
    client = chromadb.PersistentClient(index.chroma_dir, settings=Settings(anonymized_telemetry=False))
    with pytest.raises(RetrievalIndexError, match="collection does not exist"):
        search(config)
    assert client.list_collections() == []


@pytest.mark.parametrize(("metric", "model", "dimensions", "message"), [
    ("l2", SYNTHETIC_MODEL, 1536, "cosine distance"),
    ("cosine", "text-embedding-3-small", 1536, "model or dimensions"),
    ("cosine", SYNTHETIC_MODEL, 3, "model or dimensions"),
])
def test_incompatible_collection(config, metric, model, dimensions, message):
    index = index_config(config, "synthetic")
    client = chromadb.PersistentClient(index.chroma_dir, settings=Settings(anonymized_telemetry=False))
    client.create_collection(
        COLLECTION_NAME,
        configuration={"hnsw": {"space": metric}},
        metadata={"embedding_model": model, "embedding_dimensions": dimensions},
        embedding_function=None,
    )
    with pytest.raises(RetrievalIndexError, match=message):
        search(config)


def test_automatic_embedding_collection_is_rejected(config):
    index = index_config(config, "synthetic")
    client = chromadb.PersistentClient(index.chroma_dir, settings=Settings(anonymized_telemetry=False))
    client.create_collection(
        COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": SYNTHETIC_MODEL, "embedding_dimensions": 1536},
    )
    with pytest.raises(RetrievalIndexError, match="automatic embeddings disabled"):
        search(config)


@pytest.mark.parametrize(("field", "value", "message"), [
    ("patient_id", "P009", "outside the selected patient scope"),
    ("source_path", "data/patients/visit_notes/P009/A001.md", "malformed evidence"),
    ("section_heading", "", "malformed evidence"),
    ("clinician", None, "malformed evidence"),
    ("chunk_id", "different-id", "malformed evidence"),
])
def test_malformed_or_out_of_scope_response_fails_closed(config, evidence, monkeypatch, field, value, message):
    own, shared, _ = evidence
    open_chunk_collection(index_config(config, "synthetic"))
    response = {
        "ids": [[shared.chunk_id, own.chunk_id]],
        "documents": [[shared.text, own.text]],
        "distances": [[0.1, 0.2]],
        "metadatas": [[shared.to_chroma_metadata(), own.to_chroma_metadata() | {field: value}]],
    }
    monkeypatch.setattr(Collection, "query", lambda self, **kwargs: response)
    with pytest.raises(RetrievalError, match=message):
        search(config)


def test_mocked_openai_query_uses_existing_embedding_service(config, evidence, monkeypatch):
    own = evidence[0]
    query_vector = vector(1.0, 0.0, 0.0)
    upsert_chunks(index_config(config, "openai"), [own], {own.chunk_id: query_vector})

    class FakeClient:
        embeddings = None

        def __init__(self, *, api_key, timeout, max_retries):
            assert (api_key, timeout, max_retries) == ("dummy-key", 30.0, 2)
            self.embeddings = self

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def create(self, **kwargs):
            assert kwargs == {
                "model": "text-embedding-3-small", "input": [QUESTION], "encoding_format": "float"
            }
            return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=query_vector)])

    monkeypatch.setattr("src.embeddings.OpenAI", FakeClient)
    with pytest.raises(RetrievalEmbeddingError, match="OPENAI_API_KEY"):
        search(config, mode="openai")
    authenticated = config.model_copy(update={"openai_api_key": AppConfig(openai_api_key="dummy-key").openai_api_key})
    hit, = search(authenticated, mode="openai")
    assert hit.chunk_id == own.chunk_id and hit.distance == pytest.approx(0)

    def fail(*args, **kwargs):
        raise OpenAIError("dummy-key must not appear")

    monkeypatch.setattr(FakeClient, "create", fail)
    with pytest.raises(RetrievalEmbeddingError, match="failed") as error:
        search(authenticated, mode="openai")
    assert "dummy-key" not in str(error.value)
