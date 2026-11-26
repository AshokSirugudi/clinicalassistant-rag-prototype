from pathlib import Path

import pytest

from src.csv_loader import LoadResult, load_appointments, load_patients
from src.dataset_validation import ReasonCode, validate_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/patients"
FIXTURES = ROOT / "tests/fixtures/invalid"
PATIENT_HEADER = "patient_id,display_name,date_of_birth\n"
APPOINTMENT_HEADER = (
    "appointment_id,patient_id,appointment_date,clinician,status,visit_reason,visit_summary\n"
)


def test_clean_dataset():
    report = validate_dataset(load_patients(DATA / "patients.csv"), load_appointments(DATA / "appointments.csv"))
    assert report.dataset_ready is report.accepted
    assert len(report.accepted.patients) == 10
    assert len(report.accepted.appointments) == 40
    assert not (report.rejected or report.review_required or report.incomplete)
    assert report.accepted.patients[0].source.filename == str(DATA / "patients.csv")
    assert report.accepted.appointments[-1].source.line_number == 41


@pytest.mark.parametrize(("name", "code", "lines", "field"), [
    ("duplicate-patient-id", ReasonCode.DUPLICATE_ID, (2, 3), None),
    ("duplicate-appointment-id", ReasonCode.DUPLICATE_ID, (2, 3), None),
    ("missing-display-name", ReasonCode.INVALID_CSV_ROW, (2,), "display_name"),
    ("missing-clinician", ReasonCode.INVALID_CSV_ROW, (2,), "clinician"),
    ("invalid-birth-date", ReasonCode.INVALID_CSV_ROW, (2,), "date_of_birth"),
    ("invalid-appointment-date", ReasonCode.INVALID_CSV_ROW, (2,), "appointment_date"),
    ("unknown-patient-reference", ReasonCode.UNKNOWN_PATIENT, (2,), None),
    ("invalid-appointment-status", ReasonCode.INVALID_CSV_ROW, (2,), "status"),
    ("conflicting-appointment-same-id", ReasonCode.CONFLICTING_ID, (2, 3), None),
    ("conflicting-appointment-distinct-ids", ReasonCode.AMBIGUOUS_APPOINTMENT, (2, 3), None),
])
def test_invalid_fixtures(name, code, lines, field):
    fixture = FIXTURES / name
    patient_load = load_patients(fixture / "patients.csv")
    path = fixture / "appointments.csv"
    appointment_load = load_appointments(path) if path.exists() else LoadResult()
    report = validate_dataset(patient_load, appointment_load)
    assert not report.incomplete
    assert not report.accepted.appointments
    assert len(report.accepted.patients) == int(path.exists())
    if code == ReasonCode.AMBIGUOUS_APPOINTMENT:
        assert not report.rejected
        finding, = report.review_required
    else:
        assert not report.review_required
        finding, = report.rejected
    assert finding.code == code
    assert tuple(source.line_number for source in finding.sources) == lines
    assert all(source.filename == str(path if path.exists() else fixture / "patients.csv")
               for source in finding.sources)
    if field:
        assert finding.records[0].errors[0].field == field


@pytest.mark.parametrize("invalid_name", ["Other Fictional Name", ""])
@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_patient_and_linked_appointment(tmp_path, invalid_name, reverse):
    patients = tmp_path / "patients.csv"
    appointments = tmp_path / "appointments.csv"
    patient_rows = ['P1,"Fictional\nName",1990-01-01\n', f"P1,{invalid_name},1990-01-01\n"]
    if reverse:
        patient_rows.reverse()
    patients.write_text(PATIENT_HEADER + "".join(patient_rows), encoding="utf-8")
    appointments.write_text(
        APPOINTMENT_HEADER + "A1,P1,2026-10-01,Fictional Clinician,scheduled,Review,\n",
        encoding="utf-8",
    )
    report = validate_dataset(load_patients(patients), load_appointments(appointments))
    assert not report.accepted.patients and not report.accepted.appointments
    findings = {finding.code: finding for finding in report.rejected}
    conflict = findings[ReasonCode.CONFLICTING_ID]
    assert sorted(source.line_number for source in conflict.sources) == ([2, 3] if reverse else [2, 4])
    assert findings[ReasonCode.REJECTED_PATIENT].related_sources == conflict.sources
    if invalid_name:
        assert "display_name" in conflict.message
    else:
        assert ReasonCode.INVALID_CSV_ROW in findings


def test_same_day_review_is_order_independent_and_other_visits_are_accepted(tmp_path):
    patients = tmp_path / "patients.csv"
    appointments = tmp_path / "appointments.csv"
    patients.write_text(PATIENT_HEADER + "P1,Fictional Name,1990-01-01\n", encoding="utf-8")
    rows = [
        "A1,P1,2026-10-01,Fictional Clinician,completed,Review,Summary one\n",
        "A2,P1,2026-10-01,Fictional Clinician,completed,Review,Summary two\n",
        "A3,P1,2026-10-01,Fictional Clinician,scheduled,Separate visit,\n",
        "A4,P1,2026-10-01,Another Clinician,scheduled,Review,\n",
        "A5,P1,2026-10-01,Fictional Clinician,scheduled,Separate visit,\n",
    ]
    for ordered in (rows, rows[::-1]):
        appointments.write_text(APPOINTMENT_HEADER + "".join(ordered), encoding="utf-8")
        report = validate_dataset(load_patients(patients), load_appointments(appointments))
        assert not report.rejected
        finding, = report.review_required
        assert finding.code == ReasonCode.AMBIGUOUS_APPOINTMENT
        assert finding.record_ids == ("A1", "A2")
        assert {row.record.appointment_id for row in report.dataset_ready.appointments} == {"A3", "A4", "A5"}


@pytest.mark.parametrize("broken_file", ["patients", "appointments"])
def test_incomplete_load_withholds_ready_dataset(tmp_path, broken_file):
    patients = tmp_path / "patients.csv"
    appointments = tmp_path / "appointments.csv"
    patients.write_text(
        PATIENT_HEADER + "P1,Fictional Name,1990-01-01\n"
        + ('P2,"unfinished\n' if broken_file == "patients" else ""),
        encoding="utf-8",
    )
    appointments.write_text(
        APPOINTMENT_HEADER + "A1,P2,2026-10-01,Fictional Clinician,scheduled,Review,\n"
        + ('A2,"unfinished\n' if broken_file == "appointments" else ""),
        encoding="utf-8",
    )
    report = validate_dataset(load_patients(patients), load_appointments(appointments))
    assert report.incomplete and report.dataset_ready is None
    issue, = report.file_errors
    assert issue.filename == str(tmp_path / f"{broken_file}.csv")
    assert issue.line_number == 3
    if broken_file == "patients":
        assert not report.rejected
        finding, = report.review_required
        assert finding.code == ReasonCode.PATIENT_REFERENCE_UNVERIFIED
    else:
        finding, = report.rejected
        assert finding.code == ReasonCode.UNKNOWN_PATIENT


def test_normalized_ids_drive_duplicate_and_reference_checks(tmp_path):
    patients = tmp_path / "patients.csv"
    appointments = tmp_path / "appointments.csv"
    patients.write_text(
        PATIENT_HEADER
        + " 00Ab ,Aster Quillfen,1990-01-01\n"
        + "00Ab,Aster Quillfen,1990-01-01\n"
        + " 00aB ,Other Fictional Name,1990-01-01\n"
    )
    appointments.write_text(
        APPOINTMENT_HEADER
        + "A1, 00Ab ,2026-10-01,Ellis Briarfen,scheduled,Review,\n"
        + " A2 ,00aB,2026-10-01,Ellis Briarfen,scheduled,Review,\n"
        + "A2, 00aB ,2026-10-01,Ellis Briarfen,scheduled,Review,\n"
        + "A3,00aB,2026-10-02,Ellis Briarfen,scheduled,Review,\n"
    )
    report = validate_dataset(load_patients(patients), load_appointments(appointments))
    assert [row.record.patient_id for row in report.accepted.patients] == ["00aB"]
    assert [row.record.appointment_id for row in report.accepted.appointments] == ["A3"]
    assert {(finding.code, finding.record_ids) for finding in report.rejected} == {
        (ReasonCode.DUPLICATE_ID, ("00Ab",)),
        (ReasonCode.DUPLICATE_ID, ("A2",)),
        (ReasonCode.REJECTED_PATIENT, ("A1", "00Ab")),
    }
