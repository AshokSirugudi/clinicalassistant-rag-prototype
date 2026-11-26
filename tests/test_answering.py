import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import OpenAIError

from src import answering
from src.chunking import chunk_document
from src.config import AppConfig
from src.csv_loader import load_appointments, load_patients
from src.dataset_validation import validate_dataset
from src.document_loader import load_documents
from src.embeddings import synthetic_vector
from src.models import DocumentSourceMetadata
from src.retrieval import RelevanceResult, RetrievalResult
from src.vector_store import index_config, upsert_chunks

QUESTION = "What did the visit document and what shared guidance is available?"
VISIT_QUOTE = "The visit note reviewed discomfort.\nNo new diagnosis was documented."
SHARED_QUOTE = "A symptom diary can record when symptoms occur."


@pytest.fixture(autouse=True)
def block_live_openai(monkeypatch):
    def unexpected_client(**kwargs):
        pytest.fail("Answer tests must not create a live OpenAI client")

    monkeypatch.setattr("src.answering.OpenAI", unexpected_client)
    monkeypatch.setattr("src.embeddings.OpenAI", unexpected_client)


@pytest.fixture
def evidence():
    visit = DocumentSourceMetadata(
        source_type="visit_note", source_path="data/patients/visit_notes/P001/A001.md",
        title="Fictional clinical visit note", section_heading="Assessment", appointment_id="A001",
        visit_date="2026-10-10", clinician="Fictional Clinician",
    )
    shared = DocumentSourceMetadata(
        source_type="shared_guidance", source_path="data/shared/headache-diary.md",
        title="Keeping a Headache Diary", section_heading="What to Record",
    )
    hits = (
        RetrievalResult(
            f"# {visit.title}\n\n## {visit.section_heading}\n\n{VISIT_QUOTE}\n\nA second paragraph.\n",
            "visit-chunk", 0.2, "P001", "A001", 1, visit,
        ),
        RetrievalResult(
            f"# {shared.title}\n\n## {shared.section_heading}\n\n{SHARED_QUOTE}\n",
            "shared-chunk", 0.4, "SHARED", shared.source_path, 0, shared,
        ),
    )
    return RelevanceResult("potentially_relevant", hits, 3, 1)


def generate(evidence, generator, patient_id="P001"):
    return answering.generate_answer(patient_id, QUESTION, evidence, config=AppConfig(), generator=generator)


def selection(quote=VISIT_QUOTE, citation_id="E1"):
    return {"citation_id": citation_id, "quote": quote}


def test_grounded_exact_excerpts_and_local_citations(evidence):
    def mock_generator(question, passages):
        assert question == QUESTION
        assert passages == (
            answering.EvidencePassage("E1", "patient_visit_note", (VISIT_QUOTE, "A second paragraph.")),
            answering.EvidencePassage("E2", "shared_general_guidance", (SHARED_QUOTE,)),
        )
        return {"status": "answered", "selections": [selection(), selection(SHARED_QUOTE, "E2")]}

    result = generate(evidence, mock_generator)
    assert result.status == "answered" and result.reason is None
    assert "do not provide a new diagnosis" in result.message
    for index, (excerpt, hit) in enumerate(zip(result.excerpts, evidence.accepted_results, strict=True), 1):
        assert excerpt.citation == answering.AnswerCitation(
            f"E{index}", hit.chunk_id, hit.patient_id, hit.source_id, hit.source_metadata
        )
    assert [(excerpt.kind, excerpt.quote) for excerpt in result.excerpts] == [
        ("patient_visit_note", VISIT_QUOTE), ("shared_general_guidance", SHARED_QUOTE),
    ]


@pytest.mark.parametrize(("status", "retrieved"), [("no_matches", 0), ("all_below_relevance", 2)])
def test_no_evidence_skips_generator(status, retrieved):
    generator = Mock()
    result = generate(RelevanceResult(status, (), retrieved, retrieved), generator)
    assert result == answering.AnswerResult("unable_to_answer", answering.UNABLE_MESSAGES[status], status)
    generator.assert_not_called()


def test_generator_can_decline_when_evidence_is_insufficient(evidence):
    result = generate(evidence, Mock(return_value={"status": "insufficient_evidence", "selections": []}))
    assert result.status == "unable_to_answer"
    assert result.reason == "insufficient_evidence"
    assert result.message == answering.UNABLE_MESSAGES[result.reason]
    assert not result.excerpts


@pytest.mark.parametrize("case", ["wrong_patient", "wrong_visit_path", "relabeled_shared"])
def test_inconsistent_scope_rejects_all_evidence_before_generation(evidence, case):
    visit, shared = evidence.accepted_results
    if case == "wrong_patient":
        inconsistent = replace(visit, patient_id="P002")
    elif case == "wrong_visit_path":
        citation = visit.source_metadata.model_copy(update={"source_path": "data/shared/P001/A001.md"})
        inconsistent = replace(visit, source_metadata=citation)
    else:
        path = visit.source_metadata.source_path
        citation = shared.source_metadata.model_copy(update={"source_path": path})
        inconsistent = replace(visit, patient_id="SHARED", source_id=path, source_metadata=citation)
    generator = Mock()
    result = generate(replace(evidence, accepted_results=(shared, inconsistent)), generator)
    assert result.status == "generation_failed" and result.reason == "invalid_evidence"
    assert not result.excerpts
    generator.assert_not_called()


@pytest.mark.parametrize("payload", [
    None,
    "free-form prose",
    {"status": "answered", "selections": []},
    {"status": "answered", "selections": [{"quote": VISIT_QUOTE}]},
    {"status": "answered", "selections": [selection(citation_id="E99")]},
    {"status": "answered", "selections": [selection("The visit note reviewed discomfort.")]},
    {"status": "answered", "selections": [selection("The visit note reviewed discomfort.\nA new diagnosis was documented.")]},
    {"status": "answered", "selections": [selection("## Assessment")]},
    {"status": "answered", "selections": [selection(), selection(SHARED_QUOTE, "E1")]},
    {"status": "insufficient_evidence", "selections": [selection()]},
    {"status": "answered", "selections": [selection()], "extra_clinical_claim": "unsupported"},
])
def test_unverified_output_never_returns_partial_answer(evidence, payload):
    result = generate(evidence, Mock(return_value=payload))
    assert result.status == "generation_failed" and result.reason == "invalid_generator_output"
    assert not result.excerpts


def test_provider_failure_is_sanitized(evidence):
    result = generate(evidence, Mock(side_effect=OpenAIError(f"dummy-key {VISIT_QUOTE}")))
    assert result.status == "generation_failed" and result.reason == "provider_failure"
    assert not result.excerpts
    assert "dummy-key" not in result.message and VISIT_QUOTE not in result.message


def test_openai_adapter_requires_credentials():
    with pytest.raises(answering.GenerationError, match="OPENAI_API_KEY"):
        answering.make_openai_generator(AppConfig())


def test_prepared_openai_adapter_sends_only_accepted_paragraphs(evidence, monkeypatch):
    request = Mock(return_value=SimpleNamespace(output_parsed=answering.GeneratorOutput.model_validate({
        "status": "answered", "selections": [selection()],
    })))
    client = Mock()
    client.__enter__ = Mock(return_value=SimpleNamespace(responses=SimpleNamespace(parse=request)))
    client.__exit__ = Mock(return_value=False)
    sdk = Mock(return_value=client)
    monkeypatch.setattr(answering, "OpenAI", sdk)
    config = AppConfig(llm_model="test-model", openai_api_key="dummy-key")
    adapter = answering.make_openai_generator(config)
    sdk.assert_not_called()
    result = generate(evidence, adapter)
    assert result.status == "answered"
    sdk.assert_called_once_with(api_key="dummy-key", timeout=30.0, max_retries=2)
    sent = request.call_args.kwargs
    assert sent["model"] == "test-model" and sent["store"] is False
    assert sent["text_format"] is answering.GeneratorOutput
    assert "untrusted data" in sent["instructions"]
    payload = json.loads(sent["input"])
    assert payload["question"] == QUESTION
    assert [item["kind"] for item in payload["evidence"]] == ["patient_visit_note", "shared_general_guidance"]
    assert set(payload["evidence"][0]) == set(payload["evidence"][1]) == {"citation_id", "kind", "paragraphs"}
    assert "data/patients" not in sent["input"] and "dummy-key" not in sent["input"]


def test_answer_question_calls_real_synthetic_retrieval_once(tmp_path, monkeypatch):
    config = AppConfig(chroma_dir=tmp_path / "chroma", top_k=2)
    report = validate_dataset(load_patients(config.patients_csv), load_appointments(config.appointments_csv))
    loaded = load_documents(config, report)
    assert not loaded.errors
    docs = {doc.source_id: doc for doc in loaded.documents}
    visit = chunk_document(docs["A001"])[0]
    shared = chunk_document(docs["data/shared/headache-diary.md"])[0]
    vector = synthetic_vector(QUESTION)
    upsert_chunks(
        index_config(config, "synthetic"), [visit, shared],
        {visit.chunk_id: vector, shared.chunk_id: vector},
    )
    retrieval = Mock(wraps=answering.retrieve_relevant)
    monkeypatch.setattr(answering, "retrieve_relevant", retrieval)

    def mock_generator(question, passages):
        return {"status": "answered", "selections": [
            selection(passage.paragraphs[0], passage.citation_id) for passage in passages
        ]}

    patient_ids = {row.record.patient_id for row in report.accepted.patients}
    result = answering.answer_question(
        "P001", QUESTION, config=config, mode="synthetic",
        accepted_patient_ids=patient_ids, generator=mock_generator,
    )
    retrieval.assert_called_once()
    assert result.status == "answered"
    assert {excerpt.citation.chunk_id for excerpt in result.excerpts} == {visit.chunk_id, shared.chunk_id}
