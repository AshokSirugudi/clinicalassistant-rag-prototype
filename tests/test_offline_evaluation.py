"""Reproduce the four approved offline evaluation cases through Streamlit."""

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pytest
from streamlit.testing.v1 import AppTest

from src.answering import UNABLE_MESSAGES, AnswerResult
from src.keyword_search import retrieve_keywords
from src.retrieval import RelevanceResult

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    name: str
    patient_id: str
    question: str
    source_path: str | None = None
    title: str | None = None
    section: str | None = None
    quote: str | None = None
    visit: tuple[str, str, str] | None = None


CASES = (
    Case(
        "case-1-knee", "P001",
        "What does the note say about knee discomfort and limits on usual activities?",
        "data/patients/visit_notes/P001/A001.md", "Fictional clinical visit note", "Symptoms",
        "Activity-related knee discomfort with some limits on usual activities.",
        ("A001", "2026-10-10", "Iris Mosswick"),
    ),
    Case(
        "case-2-shared-diary", "P003",
        "For a headache diary, what should I note about how long each headache lasted, how intense it felt, and which activities it interrupted?",
        "data/shared/headache-diary.md", "Keeping a Headache Diary", "What to Record",
        "For each headache, note when it started, how long it lasted, how intense it felt, and which activities it interrupted.",
    ),
    Case(
        "case-3-footwear", "P010", "What about footwear?",
        "data/patients/visit_notes/P010/A037.md", "Fictional clinical visit note", "Symptoms",
        "The visit was a routine foot examination. The patient raised footwear concerns.",
        ("A037", "2026-10-19", "Rhea Crestwell"),
    ),
    Case("case-4-isolation-refusal", "P001", "What about footwear?"),
)


@dataclass(frozen=True)
class Observation:
    case: Case
    retrieval: RelevanceResult
    answer: AnswerResult
    app: AppTest


@pytest.fixture(scope="module", params=CASES, ids=lambda case: case.name)
def observed(request, tmp_path_factory):
    case = request.param
    index_path = tmp_path_factory.mktemp(case.name) / "unused-index"
    retrievals = []

    def capture(patient_id, question, **kwargs):
        assert (patient_id, question) == (case.patient_id, case.question)
        assert kwargs["config"].top_k == 5
        result = retrieve_keywords(patient_id, question, **kwargs)
        retrievals.append(result)
        return result

    def forbid_openai(**kwargs):
        pytest.fail("Offline evaluation must never construct an OpenAI client")

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("OPENAI_API_KEY", "")
        patch.delenv("CLINICAL_RAG_ENABLE_OPENAI", raising=False)
        patch.setenv("CHROMA_DIR", str(index_path))
        patch.setenv("RETRIEVAL_TOP_K", "5")
        patch.setenv("RETRIEVAL_MAX_COSINE_DISTANCE", "off")
        patch.setattr("src.keyword_search.retrieve_keywords", capture)
        patch.setattr("src.answering.OpenAI", forbid_openai)
        patch.setattr("src.embeddings.OpenAI", forbid_openai)

        app = AppTest.from_file(str(ROOT / "src/streamlit_app.py")).run()
        assert app.selectbox(key="embedding_mode").value == "keyword"
        app.selectbox(key="patient_id").select(case.patient_id).run()
        app.text_area(key="question").input(case.question)
        app.button[0].click().run()
        assert not app.exception
        assert not index_path.exists()
        retrieval, = retrievals
        context, question, answer = app.session_state["submitted_answer"]
        assert context == (case.patient_id, "keyword") and question == case.question
    return Observation(case, retrieval, answer, app)


def test_retrieval_quality(observed):
    case, retrieval = observed.case, observed.retrieval
    if case.source_path is None:
        assert retrieval == RelevanceResult("no_matches", (), 0, 0)
        return
    assert retrieval.status == "potentially_relevant"
    assert 1 <= len(retrieval.accepted_results) == retrieval.retrieved_count <= 5
    assert (case.source_path, case.section) in {
        (hit.source_metadata.source_path, hit.source_metadata.section_heading)
        for hit in retrieval.accepted_results
    }


def test_citation_accuracy(observed):
    case, answer = observed.case, observed.answer
    rendered = [block.value for block in observed.app.text]
    if case.source_path is None:
        assert not answer.excerpts
        assert not any("Source document:" in text for text in rendered)
        return

    excerpt, = answer.excerpts
    citation = excerpt.citation
    metadata = citation.source_metadata
    assert (metadata.source_path, metadata.title, metadata.section_heading) == (
        case.source_path, case.title, case.section,
    )
    source = (ROOT / case.source_path).read_text(encoding="utf-8")
    assert f"# {case.title}" in source.splitlines()
    section = source.split(f"\n## {case.section}\n", 1)[1].split("\n## ", 1)[0]
    assert excerpt.quote in re.split(r"\n[ \t]*\n", section.strip("\n"))
    hit, = [hit for hit in observed.retrieval.accepted_results if hit.chunk_id == citation.chunk_id]
    assert (citation.patient_id, citation.source_id, metadata) == (
        hit.patient_id, hit.source_id, hit.source_metadata,
    )
    details = [f"Source document: {case.title} ({case.source_path})", f"Section: {case.section}"]
    if case.visit:
        appointment_id, date, clinician = case.visit
        assert (metadata.appointment_id, metadata.visit_date.isoformat(), metadata.clinician) == case.visit
        details.extend([f"Visit date: {date}", f"Clinician: {clinician}", f"Appointment ID: {appointment_id}"])
    else:
        assert metadata.appointment_id is metadata.visit_date is metadata.clinician is None
    assert "\n".join(details) in rendered


def test_patient_isolation(observed):
    patient_id = observed.case.patient_id
    scopes_and_paths = [
        (hit.patient_id, hit.source_metadata.source_path) for hit in observed.retrieval.accepted_results
    ] + [
        (excerpt.citation.patient_id, excerpt.citation.source_metadata.source_path)
        for excerpt in observed.answer.excerpts
    ]
    for scope, path in scopes_and_paths:
        assert scope in {patient_id, "SHARED"}
        approved_dir = "data/shared" if scope == "SHARED" else f"data/patients/visit_notes/{patient_id}"
        assert PurePosixPath(path).is_relative_to(approved_dir)
    citations = [block.value for block in observed.app.text if "Source document:" in block.value]
    for other in range(1, 11):
        other_id = f"P{other:03d}"
        if other_id != patient_id:
            assert all(f"/visit_notes/{other_id}/" not in citation for citation in citations)
    if observed.case.source_path is None:
        assert not scopes_and_paths and not citations


def test_answer_behavior(observed):
    case, answer, app = observed.case, observed.answer, observed.app
    assert not app.error
    if case.source_path is None:
        assert (answer.status, answer.reason, answer.excerpts) == ("unable_to_answer", "no_matches", ())
        assert UNABLE_MESSAGES["no_matches"] in [item.value for item in app.info]
        return
    assert answer.status == "answered"
    excerpt, = answer.excerpts
    assert excerpt.quote == case.quote
    assert case.quote in [item.value for item in app.text]
    label = "**Visit note**" if case.visit else "**General guidance**"
    assert label in [item.value for item in app.markdown]
