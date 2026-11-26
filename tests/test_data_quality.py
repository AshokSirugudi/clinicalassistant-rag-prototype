from pathlib import Path

from src.csv_loader import load_appointments, load_patients
from src.data_quality import format_data_quality_summary
from src.dataset_validation import validate_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "patients"
FIXTURES = ROOT / "tests" / "fixtures" / "invalid"


def summarize(patients_path, appointments_path):
    patients = load_patients(patients_path)
    appointments = load_appointments(appointments_path)
    return format_data_quality_summary(patients, appointments, validate_dataset(patients, appointments))


def test_clean_summary():
    assert summarize(DATA / "patients.csv", DATA / "appointments.csv").splitlines() == [
        "Data quality summary",
        "Rows read: patients=10, appointments=40",
        "Accepted: 50 | Rejected: 0 | Review required: 0",
        "Dataset ready for ingestion: YES",
    ]


def test_review_records_are_excluded_from_ready_subset():
    fixture = FIXTURES / "conflicting-appointment-distinct-ids"
    output = summarize(fixture / "patients.csv", fixture / "appointments.csv")
    assert "Accepted: 1 | Rejected: 0 | Review required: 2" in output
    assert "Rejection reasons" not in output
    assert "Dataset ready for ingestion: YES (accepted records only)" in output


def test_row_with_multiple_rejection_reasons_is_counted_once(tmp_path):
    fixture = FIXTURES / "duplicate-appointment-id"
    path = tmp_path / "appointments.csv"
    header, valid, _ = (fixture / "appointments.csv").read_text().splitlines()
    invalid = valid.replace(",completed,", ",invalid,")
    path.write_text("\n".join((header, valid, invalid)) + "\n", encoding="utf-8")
    output = summarize(fixture / "patients.csv", path)
    assert "Rows read: patients=1, appointments=2" in output
    assert "Accepted: 1 | Rejected: 2 | Review required: 0" in output
    assert "Rejection reasons: conflicting_id (2), invalid_csv_row (1)" in output
    assert "Dataset ready for ingestion: YES (accepted records only)" in output


def test_incomplete_load_is_not_ready(tmp_path):
    output = summarize(DATA / "patients.csv", tmp_path / "missing.csv")
    assert "Rows read: patients=10, appointments=0" in output
    assert "Rejection reasons" not in output
    assert "Dataset ready for ingestion: NO (NOT READY, incomplete load, file errors: 1)" in output
