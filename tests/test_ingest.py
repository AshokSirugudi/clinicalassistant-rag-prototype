from pathlib import Path
from shutil import copytree

import pytest

from src import ingest as ingestion
from src.chunking import chunk_document
from src.config import AppConfig, load_config
from src.csv_loader import load_appointments, load_patients
from src.dataset_validation import validate_dataset
from src.document_loader import load_documents
from src.embeddings import EmbeddingError
from src.vector_store import SYNTHETIC_MODEL, VectorStoreError, open_chunk_collection

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sources(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    copytree(ROOT / "data", tmp_path / "data")
    config = load_config(tmp_path).model_copy(update={"chroma_dir": tmp_path / "chroma"})
    return tmp_path, config


def expected_chunks(config, root):
    report = validate_dataset(load_patients(config.patients_csv), load_appointments(config.appointments_csv))
    loaded = load_documents(config, report, project_root=root)
    assert not loaded.errors
    return {chunk.chunk_id: chunk for doc in loaded.documents for chunk in chunk_document(doc)}


def edit_sources(config):
    edited = config.shared_docs_dir / "headache-diary.md"
    original = edited.read_text(encoding="utf-8")
    edited.write_text(original.replace("how intense it felt", "how intense the headache felt"), encoding="utf-8")
    shortened = config.shared_docs_dir / "home-blood-pressure-log.md"
    original = shortened.read_text(encoding="utf-8")
    shortened.write_text(original.split("## Preparing for Follow-Up")[0], encoding="utf-8")


def test_synthetic_ingestion_isolated_and_idempotent(sources, capsys, monkeypatch):
    root, config = sources

    def forbid_openai(*args, **kwargs):
        pytest.fail("Synthetic ingestion must not invoke the OpenAI embedding service")

    monkeypatch.setattr(ingestion, "embed_chunks", forbid_openai)
    with config.appointments_csv.open("a", encoding="utf-8") as source:
        source.write(
            "A901,P999,2026-10-10,Fictional Clinician,scheduled,Review,\n"
            "A902,P001,2026-10-10,Fictional Clinician,scheduled,Review,\n"
            "A903,P001,2026-10-10,Fictional Clinician,cancelled,Review,\n"
        )
    expected = expected_chunks(config, root)
    assert config.openai_api_key is None
    for _ in range(2):
        assert ingestion.ingest(config, "synthetic", project_root=root) == ingestion.IngestionResult(24, 72, 72, 0)
    collection, _ = open_chunk_collection(ingestion._index_config(config, "synthetic"))
    assert collection.metadata["embedding_model"] == SYNTHETIC_MODEL
    assert collection.metadata["embedding_dimensions"] == 1536
    stored = collection.get(include=["metadatas"])
    assert collection.count() == 72 and set(stored["ids"]) == set(expected)
    assert dict(zip(stored["ids"], stored["metadatas"], strict=True)) == {
        key: chunk.to_chroma_metadata() for key, chunk in expected.items()
    }
    assert (config.chroma_dir / "SYNTHETIC").is_dir() and not (config.chroma_dir / "openai").exists()
    output = capsys.readouterr().out
    assert "Embedding mode: SYNTHETIC" in output
    assert "Rows read: patients=10, appointments=43" in output
    assert "Accepted: 50 | Rejected: 1 | Review required: 2" in output
    assert "Indexed: 72 | Stale deletions: 0 | Errors: 0" in output


def test_changed_and_deleted_chunks_are_reconciled(sources):
    root, config = sources
    original = expected_chunks(config, root)
    ingestion.ingest(config, "synthetic", project_root=root)
    edit_sources(config)
    expected = expected_chunks(config, root)
    assert len(original.keys() - expected.keys()) == 2
    replacement_id, = expected.keys() - original.keys()
    assert ingestion.ingest(config, "synthetic", project_root=root) == ingestion.IngestionResult(24, 71, 71, 2)
    collection, _ = open_chunk_collection(ingestion._index_config(config, "synthetic"))
    assert set(collection.get(include=[])["ids"]) == set(expected)
    replacement = collection.get(ids=[replacement_id], include=["metadatas"])
    assert replacement["metadatas"] == [expected[replacement_id].to_chroma_metadata()]


@pytest.mark.parametrize("failure", ["csv", "document", "embedding", "storage"])
def test_failures_leave_stale_ids_for_a_safe_retry(sources, monkeypatch, failure):
    root, config = sources
    original = expected_chunks(config, root)
    ingestion.ingest(config, "synthetic", project_root=root)
    edit_sources(config)
    with monkeypatch.context() as patch:
        if failure == "csv":
            missing = config.model_copy(update={"patients_csv": root / "missing.csv"})
            run_config = missing
            message = "CSV validation is incomplete"
        else:
            run_config = config
            if failure == "document":
                path = config.visit_notes_dir / "P001" / "A001.md"
                original_note = path.read_bytes()
                path.unlink()
                message = "Document validation failed"
            elif failure == "embedding":
                def fail_embeddings(chunks):
                    raise EmbeddingError("Simulated embedding failure")
                patch.setattr(ingestion, "_synthetic_embeddings", fail_embeddings)
                message = "Embedding failed"
            else:
                def fail_storage(config, chunks, vectors):
                    raise VectorStoreError("Simulated write failure")
                patch.setattr(ingestion, "upsert_chunks", fail_storage)
                message = "Indexing failed"
        with pytest.raises(ingestion.IngestionError, match=message):
            ingestion.ingest(run_config, "synthetic", project_root=root)
        if failure == "document":
            path.write_bytes(original_note)
    collection, _ = open_chunk_collection(ingestion._index_config(config, "synthetic"))
    assert set(collection.get(include=[])["ids"]) == set(original)
    assert ingestion.ingest(config, "synthetic", project_root=root).stale_deleted == 2


def test_openai_preflight_rejects_missing_credentials(tmp_path):
    config = AppConfig(chroma_dir=tmp_path / "chroma")
    with pytest.raises(ingestion.IngestionError, match="requires OPENAI_API_KEY"):
        ingestion._index_config(config, "openai")
    assert not config.chroma_dir.exists()
