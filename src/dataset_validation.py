"""Validate relationships and identities after individual CSV rows are loaded."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

from .csv_loader import CsvIssue, LoadResult, LocatedRecord, RejectedRow, SourceReference
from .models import Appointment, Patient


class ReasonCode(StrEnum):
    INVALID_CSV_ROW = "invalid_csv_row"
    DUPLICATE_ID = "duplicate_id"
    CONFLICTING_ID = "conflicting_id"
    UNKNOWN_PATIENT = "unknown_patient"
    REJECTED_PATIENT = "rejected_patient"
    PATIENT_REFERENCE_UNVERIFIED = "patient_reference_unverified"
    AMBIGUOUS_APPOINTMENT = "ambiguous_appointment"


type DatasetRow = LocatedRecord[Patient] | LocatedRecord[Appointment] | RejectedRow


@dataclass(frozen=True)
class ValidationFinding:
    code: ReasonCode
    message: str
    record_ids: tuple[str, ...]
    records: tuple[DatasetRow, ...]
    sources: tuple[SourceReference, ...]
    related_sources: tuple[SourceReference, ...] = ()


@dataclass
class DatasetRecords:
    patients: list[LocatedRecord[Patient]] = field(default_factory=list)
    appointments: list[LocatedRecord[Appointment]] = field(default_factory=list)


@dataclass
class ValidationReport:
    accepted: DatasetRecords = field(default_factory=DatasetRecords)
    rejected: list[ValidationFinding] = field(default_factory=list)
    review_required: list[ValidationFinding] = field(default_factory=list)
    file_errors: list[CsvIssue] = field(default_factory=list)

    @property
    def incomplete(self) -> bool:
        return bool(self.file_errors)

    @property
    def dataset_ready(self) -> DatasetRecords | None:
        """Return the accepted subset only after both files were fully read."""
        return None if self.incomplete else self.accepted


def _rejected_source(row: RejectedRow) -> SourceReference:
    issue = row.errors[0]
    assert issue.line_number is not None
    return SourceReference(issue.filename, issue.line_number)


def _check_ids[Record: (Patient, Appointment)](
    load: LoadResult[Record], id_field: str, report: ValidationReport
) -> tuple[list[LocatedRecord[Record]], dict[str, tuple[SourceReference, ...]]]:
    groups: dict[str, list[LocatedRecord[Record]]] = defaultdict(list)
    invalid: dict[str, list[RejectedRow]] = defaultdict(list)
    rejected_ids: dict[str, tuple[SourceReference, ...]] = {}
    for row in load.located_records:
        groups[getattr(row.record, id_field)].append(row)
    for row in load.rejected_rows:
        if row.record_id is not None:
            invalid[row.record_id].append(row)
        report.rejected.append(ValidationFinding(
            ReasonCode.INVALID_CSV_ROW, " | ".join(issue.message for issue in row.errors),
            (row.record_id,) if row.record_id is not None else (), (row,), (_rejected_source(row),),
        ))

    for record_id in sorted(groups.keys() | invalid.keys()):
        valid_rows = groups.get(record_id, [])
        invalid_rows = invalid.get(record_id, [])
        sources = tuple(row.source for row in valid_rows) + tuple(
            _rejected_source(row) for row in invalid_rows
        )
        if invalid_rows or len(valid_rows) > 1:
            rejected_ids[record_id] = sources
        if len(sources) < 2:
            continue
        if invalid_rows:
            identical = not valid_rows and all(
                row.raw_values == invalid_rows[0].raw_values for row in invalid_rows
            )
            description = "at least one row failed CSV validation"
        else:
            first = valid_rows[0].record.model_dump()
            differing = [
                name for name, value in first.items()
                if any(getattr(row.record, name) != value for row in valid_rows[1:])
            ]
            identical = not differing
            description = f"differing fields: {', '.join(differing)}"
        code = ReasonCode.DUPLICATE_ID if identical else ReasonCode.CONFLICTING_ID
        message = f"Identical rows share {id_field}" if identical else f"Rows share {id_field} with {description}"
        report.rejected.append(ValidationFinding(
            code, message, (record_id,), (*valid_rows, *invalid_rows), sources,
        ))

    return [
        row for row in load.located_records if getattr(row.record, id_field) not in rejected_ids
    ], rejected_ids


def validate_dataset(
    patient_load: LoadResult[Patient], appointment_load: LoadResult[Appointment]
) -> ValidationReport:
    """Classify records while retaining all original CSV diagnostics."""
    report = ValidationReport(file_errors=[*patient_load.file_errors, *appointment_load.file_errors])
    patients, rejected_patients = _check_ids(patient_load, "patient_id", report)
    appointments, _ = _check_ids(appointment_load, "appointment_id", report)
    report.accepted.patients = patients
    patient_ids = {row.record.patient_id for row in patients}
    eligible = []

    for row in appointments:
        appointment = row.record
        patient_id = appointment.patient_id
        ids = (appointment.appointment_id, patient_id)
        if patient_id in rejected_patients:
            report.rejected.append(ValidationFinding(
                ReasonCode.REJECTED_PATIENT, "Appointment references a rejected patient",
                ids, (row,), (row.source,), rejected_patients[patient_id],
            ))
        elif patient_id in patient_ids:
            eligible.append(row)
        elif patient_load.file_errors:
            report.review_required.append(ValidationFinding(
                ReasonCode.PATIENT_REFERENCE_UNVERIFIED, "Patient reference cannot be verified from an incomplete load",
                ids, (row,), (row.source,),
            ))
        else:
            report.rejected.append(ValidationFinding(
                ReasonCode.UNKNOWN_PATIENT, "Appointment references an unknown patient",
                ids, (row,), (row.source,),
            ))

    same_visit: dict[tuple[str, date, str, str], list[LocatedRecord[Appointment]]] = defaultdict(list)
    for row in eligible:
        record = row.record
        key = (record.patient_id, record.appointment_date, record.clinician, record.visit_reason)
        same_visit[key].append(row)

    review_ids: set[str] = set()
    for group in same_visit.values():
        details = {(row.record.status, row.record.visit_summary) for row in group}
        if len(details) > 1:
            ids = tuple(sorted(row.record.appointment_id for row in group))
            review_ids.update(ids)
            report.review_required.append(ValidationFinding(
                ReasonCode.AMBIGUOUS_APPOINTMENT, "Possible same-day encounter has differing status or summary",
                ids, tuple(group), tuple(row.source for row in group),
            ))
    report.accepted.appointments = [
        row for row in eligible if row.record.appointment_id not in review_ids
    ]
    return report
