from datetime import date, datetime

import pytest
from pydantic import ValidationError

from src.models import Appointment, AppointmentStatus, DocumentChunk, Patient


PATIENT = {"patient_id": "0001", "display_name": "Aster Quillfen", "date_of_birth": "1992-02-29"}
APPOINTMENT = {
    "appointment_id": "A901",
    "patient_id": "0001",
    "appointment_date": "2026-10-10",
    "clinician": "Ellis Briarfen",
    "status": "completed",
    "visit_reason": "Wrist discomfort",
    "visit_summary": "Reviewed the effect of typing breaks.",
}
VISIT_METADATA = {
    "source_type": "visit_note",
    "source_path": "data/patients/visit_notes/0001/A901.md",
    "title": "Fictional clinical visit note",
    "section_heading": "Assessment",
    "appointment_id": "A901",
    "visit_date": "2026-10-10",
    "clinician": "Ellis Briarfen",
}
SHARED_METADATA = {
    "source_type": "shared_guidance",
    "source_path": "data/shared/headache-diary.md",
    "title": "Keeping a Headache Diary",
    "section_heading": "What to Record",
}


def test_patient_parses_csv_values():
    patient = Patient.model_validate(PATIENT)
    assert patient.patient_id == "0001"
    assert patient.date_of_birth == date(1992, 2, 29)


@pytest.mark.parametrize("value", ["1992/02/29", "1992-02-30", "1992-2-29", "19920229",
                                  "1992-02-29T00:00:00", datetime(1992, 2, 29), 699321600])
def test_date_requires_exact_calendar_format(value):
    with pytest.raises(ValidationError, match="date_of_birth"):
        Patient.model_validate(PATIENT | {"date_of_birth": value})
    with pytest.raises(ValidationError, match="appointment_date"):
        Appointment.model_validate(APPOINTMENT | {"appointment_date": value})


@pytest.mark.parametrize("changes", [{"display_name": " "}, {"patient_id": "SHARED"}, {"extra": "x"}])
def test_invalid_patient(changes):
    with pytest.raises(ValidationError):
        Patient.model_validate(PATIENT | changes)


@pytest.mark.parametrize("status", list(AppointmentStatus))
def test_appointment_parses_all_approved_statuses(status):
    summary = APPOINTMENT["visit_summary"] if status == AppointmentStatus.COMPLETED else ""
    appointment = Appointment.model_validate(
        APPOINTMENT | {"status": status.value, "visit_summary": summary}
    )
    assert appointment.status is status
    assert appointment.appointment_date == date(2026, 10, 10)
    assert appointment.visit_summary == (summary or None)


@pytest.mark.parametrize("summary", ["", "  ", None])
def test_completed_appointment_requires_summary(summary):
    with pytest.raises(ValidationError, match="visit_summary"):
        Appointment.model_validate(APPOINTMENT | {"visit_summary": summary})


@pytest.mark.parametrize("changes", [{"status": "rescheduled"}, {"clinician": ""}, {"visit_reason": None}])
def test_invalid_appointment(changes):
    with pytest.raises(ValidationError):
        Appointment.model_validate(APPOINTMENT | changes)


def visit_chunk(**changes):
    return DocumentChunk.model_validate({
        "patient_id": "0001",
        "source_id": "A901",
        "chunk_position": 0,
        "text": "Reviewed typing breaks.",
        "source_metadata": VISIT_METADATA,
    } | changes)


def test_chunk_id_depends_on_source_position_and_exact_text():
    original = visit_chunk()
    assert original.chunk_id == visit_chunk().chunk_id
    assert original.chunk_id == original.model_dump()["chunk_id"]
    changed_source = visit_chunk(
        source_id="A902", source_metadata=VISIT_METADATA | {"appointment_id": "A902"}
    )
    assert len({
        original.chunk_id, changed_source.chunk_id, visit_chunk(chunk_position=1).chunk_id,
        visit_chunk(text=original.text + " ").chunk_id,
    }) == 4


def test_chroma_metadata_for_visit_and_shared_sources():
    visit = visit_chunk()
    assert visit.to_chroma_metadata() == {
        "patient_id": "0001", "source_id": "A901", "chunk_position": 0,
        "chunk_id": visit.chunk_id, **VISIT_METADATA,
    }
    shared = visit_chunk(
        patient_id="SHARED", source_id=SHARED_METADATA["source_path"], source_metadata=SHARED_METADATA
    )
    assert shared.to_chroma_metadata() == {
        "patient_id": "SHARED", "source_id": SHARED_METADATA["source_path"], "chunk_position": 0,
        "chunk_id": shared.chunk_id, **SHARED_METADATA,
    }


@pytest.mark.parametrize("changes", [
    {"patient_id": "SHARED"},
    {"source_id": "A902"},
    {"chunk_position": -1},
    {"chunk_position": "0"},
    {"text": " "},
    {"source_metadata": VISIT_METADATA | {"visit_date": None}},
    {"source_metadata": SHARED_METADATA, "source_id": SHARED_METADATA["source_path"]},
    {"source_metadata": SHARED_METADATA | {"appointment_id": "A901"}, "patient_id": "SHARED"},
])
def test_invalid_document_chunk(changes):
    with pytest.raises(ValidationError):
        visit_chunk(**changes)
