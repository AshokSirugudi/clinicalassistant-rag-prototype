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
from src.retrieval import RetrievalEmbeddingError, RetrievalError, RetrievalIndexError, retrieve, retrieve_relevant
from src.vector_store import COLLECTION_NAME, index_config, open_chunk_collection, upsert_chunks

PATIENT_IDS = {"P001", "P002", "P009"}
ALL_PATIENT_IDS = tuple(f"P{number:03d}" for number in range(1, 11))
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
def corpus_chunks():
    config = AppConfig()
    report = validate_dataset(load_patients(config.patients_csv), load_appointments(config.appointments_csv))
    documents = load_documents(config, report)
    assert not documents.errors
    return {
        (chunk.source_id, chunk.source_metadata.section_heading): chunk
        for document in documents.documents for chunk in chunk_document(document)
    }


@pytest.fixture(scope="module")
def evidence(corpus_chunks):
    return (
        corpus_chunks["A001", "Recommendations"],
        corpus_chunks["data/shared/activity-related-discomfort.md", "Tracking Activity and Function"],
        corpus_chunks["A034", "Recommendations"],
    )


def vector(x, y, z):
    return [x, y, z] + [0.0] * 1533


def search(config, patient_id="P001", question=QUESTION, mode="synthetic"):
    return retrieve(patient_id, question, config=config, mode=mode, accepted_patient_ids=PATIENT_IDS)


def search_relevant(config, patient_id="P001"):
    return retrieve_relevant(
        patient_id, QUESTION, config=config, mode="synthetic", accepted_patient_ids=PATIENT_IDS
    )


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


@pytest.fixture(scope="module")
def all_patient_index(tmp_path_factory, corpus_chunks, evidence):
    config = AppConfig(chroma_dir=tmp_path_factory.mktemp("all-patient-index"), top_k=2)
    by_patient = {}
    for chunk in corpus_chunks.values():
        if chunk.patient_id in ALL_PATIENT_IDS:
            by_patient.setdefault(chunk.patient_id, chunk)
    assert set(by_patient) == set(ALL_PATIENT_IDS)
    shared = evidence[1]
    # For query axis i, every other patient's vector has a stronger projection.
    embeddings = {
        by_patient[patient_id].chunk_id: [
            float(axis != position) for axis in range(len(ALL_PATIENT_IDS))
        ] + [0.0] * 1526
        for position, patient_id in enumerate(ALL_PATIENT_IDS)
    }
    embeddings[shared.chunk_id] = [0.1] * 10 + [1.0] + [0.0] * 1525
    index = index_config(config, "synthetic")
    upsert_chunks(index, [*by_patient.values(), shared], embeddings)
    return config, index, by_patient, shared


@pytest.mark.parametrize("patient_id", ALL_PATIENT_IDS)
def test_every_patient_excludes_closer_other_patients(patient_id, all_patient_index, monkeypatch):
    config, index, by_patient, shared = all_patient_index
    query_vector = [0.0] * 1536
    query_vector[ALL_PATIENT_IDS.index(patient_id)] = 1.0
    expected = {by_patient[patient_id].chunk_id, shared.chunk_id}
    collection, _ = open_chunk_collection(index, create=False)
    baseline = collection.query(query_embeddings=[query_vector], n_results=11, include=["distances"])
    ids, = baseline["ids"]
    distances, = baseline["distances"]
    assert set(ids[-2:]) == expected
    assert distances[8] < distances[9]

    monkeypatch.setattr("src.retrieval.synthetic_vector", lambda question: query_vector)
    query = Collection.query
    calls = []

    def record_query(self, **kwargs):
        calls.append(kwargs)
        return query(self, **kwargs)

    monkeypatch.setattr(Collection, "query", record_query)
    hits = retrieve(patient_id, QUESTION, config=config, mode="synthetic", accepted_patient_ids=ALL_PATIENT_IDS)
    assert len(calls) == 1
    assert calls[0]["where"] == {"patient_id": {"$in": [patient_id, "SHARED"]}}
    assert calls[0]["n_results"] == config.top_k
    assert len(hits) == 2 and {hit.chunk_id for hit in hits} == expected
    assert {hit.patient_id for hit in hits} == {patient_id, "SHARED"}


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
    result = search_relevant(config)
    assert (result.status, result.accepted_results, result.retrieved_count, result.rejected_count) == (
        "no_matches", (), 0, 0,
    )


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
    ("source_path", "data/shared/P001/A001.md", "malformed evidence"),
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


def test_visit_note_relabeled_as_shared_rejects_the_whole_response(config, evidence, monkeypatch):
    _, shared, other = evidence
    open_chunk_collection(index_config(config, "synthetic"))
    visit_path = other.source_metadata.source_path
    spoofed = shared.to_chroma_metadata() | {
        "chunk_id": other.chunk_id,
        "source_id": visit_path,
        "source_path": visit_path,
    }
    response = {
        "ids": [[shared.chunk_id, other.chunk_id]],
        "documents": [[shared.text, other.text]],
        "distances": [[0.1, 0.2]],
        "metadatas": [[shared.to_chroma_metadata(), spoofed]],
    }
    monkeypatch.setattr(Collection, "query", lambda self, **kwargs: response)
    with pytest.raises(RetrievalError, match="malformed evidence"):
        search(config)
    with pytest.raises(RetrievalError, match="malformed evidence"):
        search_relevant(config.model_copy(update={"max_cosine_distance": 0.15}))


@pytest.fixture
def relevance_index(config, evidence, corpus_chunks, monkeypatch):
    own, shared, other = evidence
    far = corpus_chunks["data/shared/home-blood-pressure-log.md", "What to Record"]
    config = config.model_copy(update={"top_k": 3})
    query_vector = vector(1.0, 0.0, 0.0)
    upsert_chunks(index_config(config, "synthetic"), [own, shared, far, other], {
        own.chunk_id: query_vector,
        shared.chunk_id: vector(0.0, 1.0, 0.0),
        far.chunk_id: vector(-1.0, 0.0, 0.0),
        other.chunk_id: query_vector,
    })
    monkeypatch.setattr("src.retrieval.synthetic_vector", lambda question: query_vector)
    return config, (own, shared, far)


@pytest.mark.parametrize(("threshold", "kept"), [(None, 3), (0.0, 1), (1.0, 2), (2.0, 3)])
def test_relevance_threshold_preserves_scoped_ranking_and_citations(relevance_index, monkeypatch, threshold, kept):
    config, expected = relevance_index
    config = config.model_copy(update={"max_cosine_distance": threshold})
    query = Collection.query
    calls = []

    def record_query(self, **kwargs):
        calls.append(kwargs)
        return query(self, **kwargs)

    monkeypatch.setattr(Collection, "query", record_query)
    result = search_relevant(config)
    assert len(calls) == 1
    assert calls[0]["where"] == {"patient_id": {"$in": ["P001", "SHARED"]}}
    assert calls[0]["n_results"] == 3
    assert result.status == "potentially_relevant"
    assert (result.retrieved_count, result.rejected_count) == (3, 3 - kept)
    assert [hit.distance for hit in result.accepted_results] == [0.0, 1.0, 2.0][:kept]
    for hit, chunk in zip(result.accepted_results, expected[:kept], strict=True):
        assert (hit.patient_id, hit.chunk_id, hit.text) == (chunk.patient_id, chunk.chunk_id, chunk.text)
        assert hit.source_metadata == chunk.source_metadata


def test_retrieved_but_all_below_relevance(relevance_index):
    config, _ = relevance_index
    result = search_relevant(config.model_copy(update={"max_cosine_distance": 0.0}), patient_id="P002")
    assert (result.status, result.accepted_results, result.retrieved_count, result.rejected_count) == (
        "all_below_relevance", (), 2, 2,
    )


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
