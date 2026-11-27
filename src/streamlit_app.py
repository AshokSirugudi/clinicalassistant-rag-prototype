"""Single-page interface for the fictional Clinical RAG prototype."""

import os
import re
from collections.abc import Collection
from threading import Lock
from typing import Literal

import streamlit as st

from src.answering import AnswerResult, EvidencePassage, answer_question, generate_answer, make_openai_generator
from src.config import AppConfig, load_config
from src.csv_loader import load_appointments, load_patients
from src.dataset_validation import ValidationReport, validate_dataset
from src.ingest import IngestionError, ingest
from src.keyword_search import keyword_terms, retrieve_keywords
from src.retrieval import RetrievalEmbeddingError, RetrievalError, RetrievalIndexError
from src.vector_store import MissingVectorIndexError, VectorStoreError, index_config, open_chunk_collection

MODE_LABELS = {"keyword": "Offline Keyword Search", "synthetic": "Synthetic Vector Test", "openai": "OpenAI"}
SYNTHETIC_INGEST_COMMAND = "python -m src.ingest --embedding-mode synthetic"
type RetrievalMode = Literal["keyword", "synthetic", "openai"]


class SyntheticInitializationError(RuntimeError):
    """Synthetic ingestion could not finish."""


@st.cache_resource
def _synthetic_build_lock() -> Lock:
    """Share a lock across Streamlit sessions and reruns in this worker."""
    return Lock()


def _ensure_synthetic_index(config: AppConfig) -> None:
    index = index_config(config, "synthetic")
    pending = index.chroma_dir / ".initialization-pending"
    with _synthetic_build_lock():
        try:
            if not pending.exists():
                try:
                    open_chunk_collection(index, create=False)
                except MissingVectorIndexError:
                    pass
                else:
                    return
            # Mark the build before writing so an interrupted build can be retried.
            with st.spinner("Preparing the synthetic test index…"):
                pending.parent.mkdir(parents=True, exist_ok=True)
                pending.touch()
                ingest(config, "synthetic")
                pending.unlink()
        except (OSError, IngestionError):
            raise SyntheticInitializationError from None


def _keyword_generator(question: str, passages: tuple[EvidencePassage, ...]) -> object:
    """Select the full evidence paragraph with the strongest keyword overlap."""
    terms = keyword_terms(question)
    candidates = [
        (len(terms & keyword_terms(paragraph)), passage.citation_id, paragraph)
        for passage in passages for paragraph in passage.paragraphs
    ]
    score, citation_id, paragraph = max(candidates, key=lambda item: item[0], default=(0, "", ""))
    if not score:
        return {"status": "insufficient_evidence", "selections": []}
    return {"status": "answered", "selections": [{"citation_id": citation_id, "quote": paragraph}]}


def _synthetic_generator(question: str, passages: tuple[EvidencePassage, ...]) -> object:
    """Choose one complete paragraph containing the demo's literal search phrase."""
    match = re.fullmatch(r'Find excerpts containing "([^"]+)"', question, re.IGNORECASE)
    phrase = (match[1] if match else question).strip().casefold()
    for passage in passages:
        for paragraph in passage.paragraphs:
            if phrase and phrase in paragraph.casefold():
                return {
                    "status": "answered",
                    "selections": [{"citation_id": passage.citation_id, "quote": paragraph}],
                }
    return {"status": "insufficient_evidence", "selections": []}


def _clear_question() -> None:
    st.session_state["question"] = ""
    st.session_state.pop("submitted_answer", None)


def _submit(
    patient_id: str, question: str, config: AppConfig, mode: RetrievalMode,
    accepted_patient_ids: Collection[str], report: ValidationReport,
) -> AnswerResult:
    if not question.strip():
        return AnswerResult("generation_failed", "Enter a question before submitting.")
    try:
        if mode == "keyword":
            evidence = retrieve_keywords(patient_id, question, config=config, report=report)
            return generate_answer(
                patient_id, question, evidence, config=config, generator=_keyword_generator,
            )
        if mode == "synthetic":
            _ensure_synthetic_index(config)
            generator = _synthetic_generator
        else:
            generator = make_openai_generator(config)
        return answer_question(
            patient_id, question, config=config, mode=mode,
            accepted_patient_ids=accepted_patient_ids, generator=generator,
        )
    except SyntheticInitializationError:
        return AnswerResult(
            "generation_failed", "The synthetic index could not be prepared. Check source validation and storage access.",
            "synthetic_initialization_failed",
        )
    except (RetrievalIndexError, VectorStoreError):
        return AnswerResult("generation_failed", "The selected evidence index is unavailable or incompatible.")
    except RetrievalEmbeddingError:
        return AnswerResult("generation_failed", "The question could not be embedded.")
    except RetrievalError:
        return AnswerResult("generation_failed", "Evidence could not be validated for this question.")
    except Exception:
        return AnswerResult("generation_failed", "A grounded answer could not be produced.")


def _show_answer(question: str, result: AnswerResult) -> None:
    st.subheader("Answer")
    st.text(f"Submitted question: {question}")
    if result.status == "unable_to_answer":
        st.info(result.message)
        return
    if result.status == "generation_failed":
        st.error(result.message)
        if result.reason == "synthetic_initialization_failed":
            st.code(SYNTHETIC_INGEST_COMMAND, language="bash")
        return

    st.caption(result.message)
    for excerpt in result.excerpts:
        citation = excerpt.citation
        metadata = citation.source_metadata
        label = "General guidance" if excerpt.kind == "shared_general_guidance" else "Visit note"
        st.markdown(f"**{label}**")
        st.text(excerpt.quote)
        details = [
            f"Source document: {metadata.title} ({metadata.source_path})",
            f"Section: {metadata.section_heading}",
        ]
        if excerpt.kind == "patient_visit_note":
            details.extend([
                f"Visit date: {metadata.visit_date}",
                f"Clinician: {metadata.clinician}",
                f"Appointment ID: {metadata.appointment_id}",
            ])
        st.text("\n".join(details))


def main() -> None:
    st.set_page_config(page_title="Clinical RAG Assistant")
    st.title("Clinical RAG Assistant")

    patient_panel = st.container()
    # Replace the old form and answer before loading the new patient details.
    question_panel = st.empty()
    with patient_panel:
        _show_patient_workspace(question_panel)


def _show_patient_workspace(question_panel) -> None:
    try:
        config = load_config()
    except (OSError, ValueError):
        _clear_question()
        st.error("Application configuration is invalid. Check cloud environment settings.")
        return

    report = validate_dataset(load_patients(config.patients_csv), load_appointments(config.appointments_csv))
    dataset = report.dataset_ready
    if dataset is None:
        _clear_question()
        st.error("Dataset validation is incomplete. Questions are unavailable.")
        return

    patients = {row.record.patient_id: row.record for row in dataset.patients}
    modes = ["keyword", "synthetic"]
    if os.getenv("CLINICAL_RAG_ENABLE_OPENAI") == "1" and config.openai_api_key is not None:
        modes.append("openai")
    if st.session_state.get("embedding_mode") not in modes:
        st.session_state["embedding_mode"] = "keyword"
    mode = st.selectbox(
        "Retrieval strategy", modes, format_func=MODE_LABELS.__getitem__,
        key="embedding_mode", on_change=_clear_question,
    )
    if mode == "keyword":
        st.caption("Offline keyword matching over fictional documents. Excerpts are for demonstration only.")
    if mode == "synthetic":
        st.warning("Synthetic embeddings are for offline testing and do not provide meaningful semantic relevance or clinical advice.")
    if "openai" not in modes:
        st.caption("OpenAI mode is disabled. An operator must enable it after configuring credentials and billing.")

    if st.session_state.get("patient_id") not in patients:
        st.session_state["patient_id"] = None
    patient_id = st.selectbox(
        "Fictional patient", sorted(patients), index=None, key="patient_id",
        format_func=lambda value: f"{patients[value].display_name} ({value})",
        placeholder="Select a patient", on_change=_clear_question,
    )
    context = (patient_id, mode)
    context_changed = st.session_state.get("answer_context") != context
    if context_changed:
        _clear_question()
        st.session_state["answer_context"] = context
    if patient_id not in patients:
        _clear_question()
        st.info("Select a fictional patient to view appointments and ask a question.")
        return

    patient = patients[patient_id]
    st.subheader("Patient details")
    st.text(f"{patient.display_name} ({patient.patient_id})\nDate of birth: {patient.date_of_birth.isoformat()}")
    st.subheader("Appointments")
    appointments = sorted(
        (row.record for row in dataset.appointments if row.record.patient_id == patient_id),
        key=lambda record: (record.appointment_date, record.appointment_id),
    )
    if appointments:
        st.dataframe([
            {
                "Date": record.appointment_date.isoformat(),
                "Appointment ID": record.appointment_id,
                "Clinician": record.clinician,
                "Status": record.status.value,
                "Visit reason": record.visit_reason,
                "Visit summary": record.visit_summary or "",
            }
            for record in appointments
        ], hide_index=True)
    else:
        st.info("No accepted appointments for this patient.")

    with question_panel.container():
        with st.form("question_form"):
            question = st.text_area(
                "Question", key="question",
                help='Offline example: Find excerpts containing "knee discomfort"',
            )
            submitted = st.form_submit_button("Ask")
        if submitted and not context_changed:
            result = _submit(patient_id, question, config, mode, patients.keys(), report)
            st.session_state["submitted_answer"] = (context, question, result)
        saved = st.session_state.get("submitted_answer")
        if saved is not None and saved[0] == context:
            _show_answer(saved[1], saved[2])


if __name__ == "__main__":
    main()
