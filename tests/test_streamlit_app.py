from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from src.answering import ANSWER_MESSAGE, UNABLE_MESSAGES, AnswerCitation, AnswerExcerpt, AnswerResult
from src.config import load_config
from src.csv_loader import CsvIssue, load_appointments, load_patients
from src.dataset_validation import ValidationReport, validate_dataset
from src.ingest import ingest
from src.keyword_search import retrieve_keywords
from src.models import DocumentSourceMetadata
from src.retrieval import RetrievalIndexError

APP = Path(__file__).resolve().parents[1] / "src" / "streamlit_app.py"
QUESTION = 'Find excerpts containing "knee discomfort"'


@pytest.fixture(autouse=True)
def offline_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("CLINICAL_RAG_ENABLE_OPENAI", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RETRIEVAL_MAX_COSINE_DISTANCE", "off")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "5")

    def forbid_api(**kwargs):
        pytest.fail("Streamlit tests must not construct an OpenAI client")

    monkeypatch.setattr("src.answering.OpenAI", forbid_api)
    monkeypatch.setattr("src.embeddings.OpenAI", forbid_api)


def open_app(patient_id=None, mode="keyword"):
    app = AppTest.from_file(str(APP)).run()
    if mode != "keyword":
        app.selectbox(key="embedding_mode").select(mode).run()
    if patient_id:
        app.selectbox(key="patient_id").select(patient_id).run()
    assert not app.exception
    return app


def submit(app, question=QUESTION):
    app.text_area(key="question").input(question)
    app.button[0].click().run()
    assert not app.exception


@pytest.fixture
def cited_answer():
    note = DocumentSourceMetadata(
        source_type="visit_note", source_path="data/patients/visit_notes/P001/A001.md",
        title="Fictional clinical visit note", section_heading="Assessment",
        appointment_id="A001", visit_date="2026-10-10", clinician="Iris Mosswick",
    )
    guidance = DocumentSourceMetadata(
        source_type="shared_guidance", source_path="data/shared/headache-diary.md",
        title="Keeping a Headache Diary", section_heading="What to Record",
    )
    return AnswerResult("answered", ANSWER_MESSAGE, excerpts=(
        AnswerExcerpt("Fictional visit excerpt.", "patient_visit_note",
                      AnswerCitation("E1", "chunk1", "P001", "A001", note)),
        AnswerExcerpt("Fictional shared excerpt.", "shared_general_guidance",
                      AnswerCitation("E2", "chunk2", "SHARED", guidance.source_path, guidance)),
    ))


def test_patient_selector_and_chronological_appointments():
    app = open_app()
    assert len(app.selectbox(key="patient_id").options) == 10
    assert "Mira Alderwick (P001)" in app.selectbox(key="patient_id").options
    assert not app.text_area

    app.selectbox(key="patient_id").select("P001").run()
    assert not app.exception
    assert any("Mira Alderwick (P001)" in item.value and "1988-04-17" in item.value for item in app.text)
    records = app.dataframe[0].value
    dates_and_ids = list(zip(records["Date"], records["Appointment ID"], strict=True))
    assert dates_and_ids == [
        ("2026-05-10", "A002"), ("2026-08-10", "A004"),
        ("2026-10-10", "A001"), ("2026-12-10", "A003"),
    ]


def test_appointment_id_breaks_date_ties(monkeypatch):
    from dataclasses import replace

    config = load_config()
    report = validate_dataset(load_patients(config.patients_csv), load_appointments(config.appointments_csv))
    appointments = report.accepted.appointments
    date = next(row.record.appointment_date for row in appointments if row.record.appointment_id == "A001")
    report.accepted.appointments = [
        replace(row, record=row.record.model_copy(update={"appointment_date": date}))
        if row.record.appointment_id == "A002" else row
        for row in reversed(appointments)
    ]
    monkeypatch.setattr("src.dataset_validation.validate_dataset", Mock(return_value=report))
    app = open_app("P001")
    records = app.dataframe[0].value
    assert list(records["Date"]) == sorted(records["Date"])
    assert list(records.loc[records["Date"] == date.isoformat(), "Appointment ID"]) == ["A001", "A002"]


def test_submit_once_render_citations_and_clear_on_patient_switch(monkeypatch, cited_answer):
    answer = Mock(return_value=cited_answer)
    monkeypatch.setattr("src.answering.answer_question", answer)
    app = open_app("P001", mode="synthetic")
    app.text_area(key="question").input(QUESTION).run()
    answer.assert_not_called()
    submit(app)
    answer.assert_called_once()
    assert answer.call_args.args == ("P001", QUESTION)
    assert answer.call_args.kwargs["mode"] == "synthetic"
    assert len(answer.call_args.kwargs["accepted_patient_ids"]) == 10
    assert {item.value for item in app.markdown} >= {"**Visit note**", "**General guidance**"}
    citations = "\n".join(item.value for item in app.text)
    for value in (
        "Fictional visit excerpt.", "Fictional shared excerpt.", "A001.md",
        "Section: Assessment", "Visit date: 2026-10-10", "Clinician: Iris Mosswick",
        "Appointment ID: A001", "headache-diary.md", "Section: What to Record",
    ):
        assert value in citations

    app.run()
    answer.assert_called_once()
    app.selectbox(key="patient_id").select("P002").run()
    assert not app.exception
    assert app.text_area(key="question").value == ""
    assert not any("Fictional visit excerpt." in item.value for item in app.text)
    assert "submitted_answer" not in app.session_state
    answer.assert_called_once()


def test_patient_no_longer_accepted_clears_selection_and_answer(monkeypatch, cited_answer):
    answer = Mock(return_value=cited_answer)
    monkeypatch.setattr("src.answering.answer_question", answer)
    app = open_app("P001", mode="synthetic")
    submit(app)
    config = load_config()
    report = validate_dataset(load_patients(config.patients_csv), load_appointments(config.appointments_csv))
    report.accepted.patients = [row for row in report.accepted.patients if row.record.patient_id != "P001"]
    monkeypatch.setattr("src.dataset_validation.validate_dataset", Mock(return_value=report))

    app.run()
    assert not app.exception
    assert app.selectbox(key="patient_id").value is None
    assert not app.text_area
    assert "submitted_answer" not in app.session_state
    answer.assert_called_once()


@pytest.mark.parametrize("reason", ["no_matches", "all_below_relevance", "insufficient_evidence"])
def test_unable_to_answer_messages(monkeypatch, reason):
    monkeypatch.setattr("src.answering.answer_question", Mock(
        return_value=AnswerResult("unable_to_answer", UNABLE_MESSAGES[reason], reason)
    ))
    app = open_app("P001", mode="synthetic")
    submit(app)
    assert UNABLE_MESSAGES[reason] in [item.value for item in app.info]


def test_missing_synthetic_index_shows_ingestion_command():
    app = open_app("P001", mode="synthetic")
    submit(app)
    assert any("synthetic index is missing" in item.value for item in app.error)
    assert app.code[0].value == "python -m src.ingest --embedding-mode synthetic"


def test_storage_errors_do_not_expose_details(monkeypatch):
    monkeypatch.setattr("src.answering.answer_question", Mock(
        side_effect=RetrievalIndexError("private index detail dummy-key")
    ))
    app = open_app("P001", mode="synthetic")
    submit(app)
    assert [item.value for item in app.error] == ["The selected evidence index is unavailable or incompatible."]
    assert not app.code


def test_dataset_validation_failure_disables_questions(monkeypatch):
    monkeypatch.setattr("src.dataset_validation.validate_dataset", Mock(
        return_value=ValidationReport(file_errors=[CsvIssue("private.csv", None, None, "private detail")])
    ))
    app = open_app()
    assert not app.selectbox and not app.text_area
    assert [item.value for item in app.error] == ["Dataset validation is incomplete. Questions are unavailable."]


def test_openai_is_disabled_without_server_opt_in_even_with_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    app = open_app()
    assert app.selectbox(key="embedding_mode").value == "keyword"
    assert app.selectbox(key="embedding_mode").options == ["Offline Keyword Search", "Synthetic Vector Test"]
    assert any("OpenAI mode is disabled" in item.value for item in app.caption)
    assert "dummy-key" not in str(app)


def test_mode_switch_clears_question_and_answer_without_calling_openai(monkeypatch, cited_answer):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    monkeypatch.setenv("CLINICAL_RAG_ENABLE_OPENAI", "1")
    answer = Mock(return_value=cited_answer)
    monkeypatch.setattr("src.answering.answer_question", answer)
    app = open_app("P001", mode="synthetic")
    assert app.selectbox(key="embedding_mode").value == "synthetic"
    submit(app)
    app.selectbox(key="embedding_mode").select("openai").run()
    assert not app.exception
    assert app.text_area(key="question").value == ""
    assert "submitted_answer" not in app.session_state
    answer.assert_called_once()
    adapter = Mock()
    monkeypatch.setattr("src.answering.make_openai_generator", Mock(return_value=adapter))
    submit(app)
    assert answer.call_count == 2
    assert answer.call_args.kwargs["mode"] == "openai"
    assert answer.call_args.kwargs["generator"] is adapter


def test_synthetic_end_to_end_with_temporary_index(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_TOP_K", "18")
    assert ingest(load_config(), "synthetic").indexed == 72
    app = open_app("P001", mode="synthetic")
    assert any("do not provide meaningful semantic relevance" in item.value for item in app.warning)
    submit(app)
    text = "\n".join(item.value for item in app.text)
    assert not app.error
    assert "data/patients/visit_notes/P001/A001.md" in text
    assert "data/patients/visit_notes/P002/" not in text
    assert "Section:" in text and "Appointment ID: A001" in text


def test_default_keyword_search_answers_without_index_and_clears_on_strategy_switch(monkeypatch):
    retrieve = Mock(wraps=retrieve_keywords)
    monkeypatch.setattr("src.keyword_search.retrieve_keywords", retrieve)
    vector_answer = Mock(side_effect=AssertionError("Keyword search must bypass vector retrieval"))
    monkeypatch.setattr("src.answering.answer_question", vector_answer)
    app = open_app("P001")
    assert app.selectbox(key="embedding_mode").value == "keyword"
    app.text_area(key="question").input(QUESTION).run()
    retrieve.assert_not_called()
    submit(app)
    assert not app.error
    text = "\n".join(item.value for item in app.text)
    assert "data/patients/visit_notes/P001/A001.md" in text and "Section: Symptoms" in text
    assert "Appointment ID: A001" in text
    assert "data/patients/visit_notes/P002/" not in text
    assert not load_config().chroma_dir.exists()
    app.run()
    retrieve.assert_called_once()

    app.selectbox(key="embedding_mode").select("synthetic").run()
    assert app.text_area(key="question").value == ""
    assert "submitted_answer" not in app.session_state
    assert not any("A001.md" in item.value for item in app.text)
    retrieve.assert_called_once()
    vector_answer.assert_not_called()


def test_keyword_search_with_no_scoped_support_returns_unable_to_answer():
    app = open_app("P001")
    submit(app, "footwear")
    assert UNABLE_MESSAGES["no_matches"] in [item.value for item in app.info]
    assert not app.error
    assert not any("A037.md" in item.value for item in app.text)
