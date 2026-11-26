from collections import Counter
from dataclasses import replace

import pytest
from pydantic import ValidationError

from src.chunking import chunk_document
from src.config import AppConfig
from src.csv_loader import load_appointments, load_patients
from src.dataset_validation import validate_dataset
from src.document_loader import MarkdownDocument, load_documents


@pytest.fixture
def guidance():
    path = "data/shared/example.md"
    return MarkdownDocument(
        source_id=path,
        source_path=path,
        source_type="shared_guidance",
        patient_id="SHARED",
        title="Fictional guidance",
        body="# Fictional guidance\n\nIntro.\n\n## First\n\n  Line one.\nLine two.  \n\n## Second\n\nAnother paragraph.\n",
        section_headings=("First", "Second"),
    )


def test_chunk_text_order_and_citations(guidance):
    first, second = chunk_document(guidance)
    assert [first.chunk_position, second.chunk_position] == [0, 1]
    assert first.text == "# Fictional guidance\n\n## First\n\n  Line one.\nLine two.  \n\n"
    assert second.text == "# Fictional guidance\n\n## Second\n\nAnother paragraph.\n"
    assert first.to_chroma_metadata() == {
        "patient_id": "SHARED",
        "source_id": guidance.source_id,
        "source_path": guidance.source_path,
        "source_type": "shared_guidance",
        "title": "Fictional guidance",
        "section_heading": "First",
        "chunk_position": 0,
        "chunk_id": first.chunk_id,
    }


def test_chunk_ids_repeat_and_change_with_section_text(guidance):
    first, second = chunk_document(guidance)
    assert [chunk.chunk_id for chunk in chunk_document(guidance)] == [first.chunk_id, second.chunk_id]
    changed = chunk_document(replace(guidance, body=guidance.body.replace("Line two.", "Updated text.")))
    assert changed[0].chunk_id != first.chunk_id
    assert changed[1].chunk_id == second.chunk_id


@pytest.mark.parametrize(("old", "new", "message"), [
    ("## First", "## Renamed", "malformed Markdown"),
    ("## First", "## ", "malformed Markdown"),
    ("## Second", "## First", "malformed Markdown"),
    ("  Line one.\nLine two.  \n", " \n", "empty section 'First'"),
])
def test_bad_sections_include_source_path(guidance, old, new, message):
    with pytest.raises(ValueError, match=message) as error:
        chunk_document(replace(guidance, body=guidance.body.replace(old, new)))
    assert guidance.source_path in str(error.value)


def test_shared_guidance_cannot_be_scoped_to_a_patient(guidance):
    with pytest.raises(ValidationError, match="only shared guidance uses patient_id SHARED"):
        chunk_document(replace(guidance, patient_id="P001"))


def test_clean_corpus_counts_metadata_and_patient_isolation():
    config = AppConfig()
    dataset = validate_dataset(load_patients(config.patients_csv), load_appointments(config.appointments_csv))
    result = load_documents(config, dataset)
    assert not result.errors
    patient_chunks = Counter()
    source_counts = Counter()
    chunk_ids = set()
    for document in result.documents:
        chunks = chunk_document(document)
        assert [chunk.chunk_position for chunk in chunks] == [0, 1, 2]
        for chunk in chunks:
            assert chunk.patient_id == document.patient_id
            assert chunk.source_id == document.source_id
            assert chunk.source_metadata.source_path == document.source_path
            assert chunk.source_metadata.title == document.title
            assert chunk.source_metadata.section_heading == document.section_headings[chunk.chunk_position]
            assert chunk.source_metadata.appointment_id == document.appointment_id
            assert chunk.source_metadata.visit_date == document.visit_date
            assert chunk.source_metadata.clinician == document.clinician
            patient_chunks[chunk.patient_id] += 1
            source_counts[chunk.source_metadata.source_type] += 1
            chunk_ids.add(chunk.chunk_id)
    assert source_counts == {"visit_note": 60, "shared_guidance": 12}
    assert len(chunk_ids) == 72
    assert patient_chunks == {"SHARED": 12, **{f"P{i:03d}": 6 for i in range(1, 11)}}
