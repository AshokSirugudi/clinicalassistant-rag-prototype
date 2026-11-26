from datetime import date
from pathlib import Path

import pytest

from src.csv_loader import load_appointments, load_patients
from src.models import AppointmentStatus

PATIENT_HEADER = "patient_id,display_name,date_of_birth\n"
APPOINTMENT_HEADER = (
    "appointment_id,patient_id,appointment_date,clinician,status,visit_reason,visit_summary\n"
)
DATA = Path(__file__).resolve().parents[1] / "data" / "patients"


def test_clean_datasets():
    for loader, name, count in (
        (load_patients, "patients.csv", 10),
        (load_appointments, "appointments.csv", 40),
    ):
        result = loader(DATA / name)
        assert len(result.valid_records) == count
        assert not result.rejected_rows
        assert not result.file_errors


def test_reordered_headers_and_literal_strings(tmp_path):
    path = tmp_path / "patients.csv"
    path.write_text(
        "\ufeffdisplay_name,date_of_birth,patient_id\nNA,1992-02-29,0001\n",
        encoding="utf-8",
    )
    result = load_patients(str(path))
    assert not result.file_errors and not result.rejected_rows
    patient, = result.valid_records
    assert (patient.patient_id, patient.display_name, patient.date_of_birth) == (
        "0001", "NA", date(1992, 2, 29)
    )


@pytest.mark.parametrize(("header", "field", "message"), [
    ("patient_id,date_of_birth\n", "display_name", "Missing header"),
    ("patient_id,display_name,date_of_birth,display_name\n", "display_name", "Duplicate header"),
    ("patient_id,display_name,date_of_birth,extra\n", "extra", "Unexpected header"),
])
def test_invalid_headers(tmp_path, header, field, message):
    path = tmp_path / "wrong.csv"
    path.write_text(header, encoding="utf-8")
    result = load_patients(path)
    issue, = result.file_errors
    assert (issue.filename, issue.line_number, issue.field, issue.message) == (
        str(path), 1, field, message
    )
    assert not result.valid_records and not result.rejected_rows


def test_visit_summary_header_required_even_when_cells_are_optional(tmp_path):
    path = tmp_path / "appointments.csv"
    path.write_text(APPOINTMENT_HEADER.replace(",visit_summary", ""), encoding="utf-8")
    issue, = load_appointments(path).file_errors
    assert (issue.field, issue.message) == ("visit_summary", "Missing header")


def test_mixed_rows_keep_physical_line_numbers_and_raw_values(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_bytes((
        PATIENT_HEADER
        + '001,"Aster\nQuillfen",1992-02-29\n'
        + "002,,1992-02-30\n"
        + "003,Too few\n"
        + "\n"
        + "004,Too many,1990-01-01,extra\n"
        + "005,NA,1990-01-01\n"
    ).replace("\n", "\r\n").encode("utf-8"))
    result = load_patients(path)
    assert not result.file_errors
    assert [patient.patient_id for patient in result.valid_records] == ["001", "005"]
    assert result.valid_records[0].display_name == "Aster\r\nQuillfen"
    assert [row.errors[0].line_number for row in result.rejected_rows] == [4, 5, 6, 7]
    invalid, short, blank, long = result.rejected_rows
    assert invalid.raw_values == ("002", "", "1992-02-30")
    assert {issue.field for issue in invalid.errors} == {"display_name", "date_of_birth"}
    assert all(issue.filename == str(path) for row in result.rejected_rows for issue in row.errors)
    assert short.raw_values == ("003", "Too few")
    assert blank.raw_values == ()
    assert long.raw_values[-1] == "extra"


def test_optional_summary_and_dataset_checks_are_deferred(tmp_path):
    path = tmp_path / "appointments.csv"
    path.write_text(
        APPOINTMENT_HEADER
        + "0001,unknown,2026-10-10,Ellis Briarfen,scheduled,Review,\n"
        + "0001,unknown,2026-10-10,Ellis Briarfen,cancelled,Review,NA\n"
        + "0002,unknown,2026-10-10,Ellis Briarfen,completed,Review,\n",
        encoding="utf-8",
    )
    result = load_appointments(path)
    assert not result.file_errors
    assert [record.appointment_id for record in result.valid_records] == ["0001", "0001"]
    first, second = result.valid_records
    assert first.patient_id == "unknown"
    assert first.status is AppointmentStatus.SCHEDULED and first.visit_summary is None
    assert second.visit_summary == "NA"
    rejected, = result.rejected_rows
    issue, = rejected.errors
    assert issue.line_number == 4 and "visit_summary" in issue.message


def test_broken_quoting_reports_file_error_and_preserves_valid_prefix(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text(
        PATIENT_HEADER + '001,Aster,1992-02-29\n002,"unterminated\nstill open\n',
        encoding="utf-8",
    )
    result = load_patients(path)
    assert [record.patient_id for record in result.valid_records] == ["001"]
    issue, = result.file_errors
    assert (issue.filename, issue.line_number) == (str(path), 3)


def test_missing_file_reports_file_error(tmp_path):
    path = tmp_path / "missing.csv"
    issue, = load_patients(path).file_errors
    assert issue.filename == str(path) and issue.line_number is None


def test_header_only_file_is_valid(tmp_path):
    path = tmp_path / "patients.csv"
    path.write_text(PATIENT_HEADER, encoding="utf-8")
    result = load_patients(path)
    assert not result.valid_records and not result.rejected_rows and not result.file_errors


def test_padded_patient_fields_preserve_identifier_characters(tmp_path):
    path = tmp_path / "patients.csv"
    path.write_text(PATIENT_HEADER + "  00Ab 9  ,  Aster  Quillfen  , 1992-02-29 \n")
    result = load_patients(path)
    patient, = result.valid_records
    assert (patient.patient_id, patient.display_name, patient.date_of_birth) == (
        "00Ab 9", "Aster  Quillfen", date(1992, 2, 29)
    )
    assert not result.rejected_rows and not result.file_errors


def test_padded_appointment_fields_preserve_multiline_clinical_text(tmp_path):
    path = tmp_path / "appointments.csv"
    reason = "  Typing discomfort\n  during longer tasks  "
    summary = "  Reviewed  breaks.\nDiscussed timing.  "
    path.write_text(
        APPOINTMENT_HEADER
        + f' 00Az 1 , 00Ab 9 , 2026-10-10 , Ellis  Briarfen , completed ,"{reason}","{summary}"\n'
    )
    appointment, = load_appointments(path).valid_records
    assert appointment.appointment_id == "00Az 1"
    assert appointment.patient_id == "00Ab 9"
    assert appointment.appointment_date == date(2026, 10, 10)
    assert appointment.clinician == "Ellis  Briarfen"
    assert appointment.status is AppointmentStatus.COMPLETED
    assert appointment.visit_reason == reason
    assert appointment.visit_summary == summary


def test_whitespace_only_summaries_follow_appointment_status(tmp_path):
    path = tmp_path / "appointments.csv"
    path.write_text(
        APPOINTMENT_HEADER
        + 'A1,P1,2026-10-10,Ellis Briarfen,scheduled,Review," \t\n "\n'
        + ' A2 ,P1,2026-10-10,Ellis Briarfen,completed,Review," \t "\n'
    )
    result = load_appointments(path)
    appointment, = result.valid_records
    assert appointment.visit_summary is None
    rejected, = result.rejected_rows
    assert rejected.raw_values[0] == " A2 " and rejected.raw_values[-1] == " \t "
    assert rejected.record_id == "A2"
    assert rejected.errors[0].line_number == 4
    assert "visit_summary" in rejected.errors[0].message


def test_normalization_does_not_repair_invalid_dates_status_or_reason(tmp_path):
    path = tmp_path / "appointments.csv"
    path.write_text(
        APPOINTMENT_HEADER + " A1 ,P1, 2026/10/10 ,Ellis Briarfen, Scheduled ,   ,\n"
    )
    result = load_appointments(path)
    assert not result.valid_records
    rejected, = result.rejected_rows
    assert rejected.raw_values == (
        " A1 ", "P1", " 2026/10/10 ", "Ellis Briarfen", " Scheduled ", "   ", ""
    )
    assert {issue.field for issue in rejected.errors} == {"appointment_date", "status", "visit_reason"}
