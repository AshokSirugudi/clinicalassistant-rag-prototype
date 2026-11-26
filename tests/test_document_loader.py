from pathlib import Path

import pytest

from src.config import AppConfig
from src.csv_loader import CsvIssue, load_appointments, load_patients
from src.dataset_validation import ValidationReport, validate_dataset
from src.document_loader import load_documents

ROOT = Path(__file__).resolve().parents[1]
CONFIG = AppConfig()


@pytest.fixture
def report():
    return validate_dataset(load_patients(CONFIG.patients_csv), load_appointments(CONFIG.appointments_csv))


@pytest.fixture
def document_dirs(tmp_path):
    notes = tmp_path / "visit_notes"
    shared = tmp_path / "shared"
    (notes / "P001").mkdir(parents=True)
    shared.mkdir()
    config = AppConfig(visit_notes_dir=notes, shared_docs_dir=shared)
    note = notes / "P001" / "A001.md"
    text = (CONFIG.visit_notes_dir / "P001" / "A001.md").read_text(encoding="utf-8")
    return config, note, text


def test_clean_documents_have_validated_patient_scope_and_citation_fields(report):
    result = load_documents(CONFIG, report)
    assert not result.errors
    notes = [doc for doc in result.documents if doc.source_type == "visit_note"]
    shared = [doc for doc in result.documents if doc.source_type == "shared_guidance"]
    assert (len(notes), len(shared)) == (20, 4)
    assert {doc.source_id: doc.patient_id for doc in notes} == {
        row.record.appointment_id: row.record.patient_id
        for row in report.accepted.appointments if row.record.status == "completed"
    }
    assert all(doc.patient_id == "SHARED" and doc.source_id == doc.source_path for doc in shared)
    assert all(doc.title and doc.body and doc.section_headings for doc in result.documents)
    assert all((ROOT / doc.source_path).is_file() for doc in result.documents)
    assert notes[0].source_path == "data/patients/visit_notes/P001/A001.md"
    assert notes[0].section_headings == ("Symptoms", "Assessment", "Recommendations")


@pytest.mark.parametrize(("original", "replacement", "message"), [
    ('patient_id: "P001"', 'patient_id: "P002"', "patient_id does not match"),
    ('visit_date: "2026-10-10"', 'visit_date: "2026-10-11"', "visit_date does not match"),
    ('clinician: "Iris Mosswick"', 'clinician: "Different Clinician"', "clinician does not match"),
    ('appointment_id: "A001"', 'appointment_id: "A003"', "accepted, completed appointment"),
    ('patient_id: "P001"', 'patient_id: P001', "double-quoted string"),
    ('patient_id: "P001"', 'patient_id: "P001"\npatient_id: "P002"', "duplicate"),
    ('patient_id: "P001"\n', "", "Missing front matter fields"),
    ("## Symptoms", "## Observations", "Expected Symptoms"),
])
def test_rejects_malformed_or_mismatched_notes(report, document_dirs, tmp_path, original, replacement, message):
    config, note, text = document_dirs
    assert original in text
    note.write_text(text.replace(original, replacement), encoding="utf-8")
    result = load_documents(config, report, project_root=tmp_path)
    assert not result.documents
    errors = [error.message for error in result.errors if error.source_path == "visit_notes/P001/A001.md"]
    assert any(message in error for error in errors)
    assert "Missing valid note for completed appointment" in errors


def test_scope_rejects_wrong_folders_and_shared_visit_notes(report, document_dirs, tmp_path):
    config, note, text = document_dirs
    wrong_folder = config.visit_notes_dir / "P002"
    wrong_folder.mkdir()
    (wrong_folder / note.name).write_text(text, encoding="utf-8")
    (config.shared_docs_dir / "misplaced.md").write_text(text, encoding="utf-8")
    body_only = text.split("---\n", 2)[2]
    (config.shared_docs_dir / "body-only.md").write_text(body_only, encoding="utf-8")
    (config.shared_docs_dir / "guidance.md").write_text("# General guidance\n\n## Recording\n\nKeep a log.\n")
    result = load_documents(config, report, project_root=tmp_path)
    document, = result.documents
    assert (document.patient_id, document.source_id) == ("SHARED", "shared/guidance.md")
    errors = {issue.source_path: issue.message for issue in result.errors}
    assert "folder or filename" in errors["visit_notes/P002/A001.md"]
    assert "front matter" in errors["shared/misplaced.md"]
    assert "shared guidance" in errors["shared/body-only.md"]


def test_incomplete_validation_blocks_all_document_loading():
    report = ValidationReport(file_errors=[CsvIssue("appointments.csv", 1, None, "Missing header")])
    result = load_documents(CONFIG, report)
    assert not result.documents
    issue, = result.errors
    assert "incomplete" in issue.message
