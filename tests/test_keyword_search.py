from dataclasses import replace
from unittest.mock import Mock

import pytest

from src import keyword_search
from src.config import AppConfig
from src.csv_loader import CsvIssue, load_appointments, load_patients
from src.dataset_validation import ValidationReport, validate_dataset
from src.document_loader import DocumentLoadResult, load_documents
from src.retrieval import KeywordResult, RetrievalError


@pytest.fixture(scope="module")
def report():
    config = AppConfig()
    return validate_dataset(load_patients(config.patients_csv), load_appointments(config.appointments_csv))


def test_corpus_keyword_ranking_is_deterministic_and_preserves_citations(report, tmp_path):
    config = AppConfig(chroma_dir=tmp_path / "unused", top_k=3, max_cosine_distance=0.0)
    result = keyword_search.retrieve_keywords(
        " P001 ", 'Find excerpts containing "KNEE discomfort"', config=config, report=report,
    )
    assert result.status == "potentially_relevant"
    assert result.retrieved_count == 3 and result.rejected_count == 0
    assert result == keyword_search.retrieve_keywords("P001", "knee discomfort", config=config, report=report)
    first, second, _ = result.accepted_results
    assert isinstance(first, KeywordResult) and not hasattr(first, "distance")
    assert first.keyword_score == second.keyword_score == 2
    assert first.source_id == second.source_id == "A001"
    assert first.source_metadata.section_heading == "Symptoms"
    assert second.source_metadata.section_heading == "Assessment"
    assert first.source_metadata.source_path == "data/patients/visit_notes/P001/A001.md"
    assert first.source_metadata.appointment_id == "A001"
    assert first.source_metadata.visit_date.isoformat() == "2026-10-10"
    assert first.source_metadata.clinician == "Iris Mosswick"
    assert {hit.patient_id for hit in result.accepted_results} <= {"P001", "SHARED"}
    assert not config.chroma_dir.exists()


def test_other_patient_is_not_scored_and_shared_is_searchable(report, monkeypatch):
    config = AppConfig(top_k=1)
    tokenize = Mock(wraps=keyword_search.keyword_terms)
    monkeypatch.setattr(keyword_search, "keyword_terms", tokenize)
    result = keyword_search.retrieve_keywords("P001", "footwear", config=config, report=report)
    assert (result.status, result.accepted_results, result.retrieved_count) == ("no_matches", (), 0)
    assert all("footwear" not in call.args[0].casefold() for call in tokenize.call_args_list[1:])
    owner, = keyword_search.retrieve_keywords("P010", "footwear", config=config, report=report).accepted_results
    assert owner.patient_id == "P010" and owner.source_id == "A037"

    shared, = keyword_search.retrieve_keywords("P001", "headache diary", config=config, report=report).accepted_results
    assert shared.patient_id == "SHARED" and shared.keyword_score == 2
    assert shared.source_metadata.source_type == "shared_guidance"
    assert shared.source_metadata.source_path == "data/shared/headache-diary.md"
    assert shared.source_metadata.appointment_id is None


@pytest.mark.parametrize(("patient", "question", "message"), [
    ("P001", " \t ", "Question must not be blank"),
    ("P999", "knee", "Select a known patient"),
    ("p001", "knee", "Select a known patient"),
    ("SHARED", "knee", "Select a known patient"),
])
def test_invalid_inputs_fail_before_loading_documents(report, monkeypatch, patient, question, message):
    load = Mock()
    monkeypatch.setattr(keyword_search, "load_documents", load)
    with pytest.raises(RetrievalError, match=message):
        keyword_search.retrieve_keywords(patient, question, config=AppConfig(), report=report)
    load.assert_not_called()


def test_no_terms_or_matching_words(report):
    for question in ("What about it?", "unmatched termxyz"):
        result = keyword_search.retrieve_keywords("P001", question, config=AppConfig(), report=report)
        assert (result.status, result.accepted_results, result.retrieved_count, result.rejected_count) == (
            "no_matches", (), 0, 0,
        )


def test_incomplete_dataset_blocks_keyword_search():
    report = ValidationReport(file_errors=[CsvIssue("patients.csv", None, None, "incomplete")])
    with pytest.raises(RetrievalError, match="Dataset validation is incomplete"):
        keyword_search.retrieve_keywords("P001", "knee", config=AppConfig(), report=report)


def test_relabeled_visit_note_fails_closed_even_after_a_valid_match(report, monkeypatch):
    config = AppConfig()
    loaded = load_documents(config, report)
    assert not loaded.errors
    shared = next(doc for doc in loaded.documents if doc.patient_id == "SHARED")
    other = next(doc for doc in loaded.documents if doc.patient_id == "P010")
    relabeled = replace(
        other, patient_id="SHARED", source_type="shared_guidance", source_id=other.source_path,
        appointment_id=None, visit_date=None, clinician=None,
    )
    monkeypatch.setattr(keyword_search, "load_documents", Mock(
        return_value=DocumentLoadResult(documents=[shared, relabeled]),
    ))
    with pytest.raises(RetrievalError, match="malformed evidence"):
        keyword_search.retrieve_keywords("P001", "activity", config=config, report=report)
