"""Load CSV records with source locations and row-level validation errors."""

import csv
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import TypeVar

import pandas as pd
from pydantic import ValidationError

from .models import Appointment, Patient

Record = TypeVar("Record", Patient, Appointment)
TRIMMED_FIELDS = {
    "patient_id", "appointment_id", "display_name", "date_of_birth",
    "appointment_date", "clinician", "status",
}


@dataclass(frozen=True)
class CsvIssue:
    filename: str
    line_number: int | None
    field: str | None
    message: str


@dataclass(frozen=True)
class RejectedRow:
    raw_values: tuple[str, ...]
    errors: tuple[CsvIssue, ...]
    record_id: str | None = None


@dataclass(frozen=True)
class SourceReference:
    filename: str
    line_number: int


@dataclass(frozen=True)
class LocatedRecord[RecordType]:
    record: RecordType
    source: SourceReference


@dataclass
class LoadResult[RecordType]:
    located_records: list[LocatedRecord[RecordType]] = field(default_factory=list)
    rejected_rows: list[RejectedRow] = field(default_factory=list)
    file_errors: list[CsvIssue] = field(default_factory=list)

    @property
    def valid_records(self) -> list[RecordType]:
        return [row.record for row in self.located_records]


def load_patients(path: str | Path) -> LoadResult[Patient]:
    return _load(path, Patient)


def load_appointments(path: str | Path) -> LoadResult[Appointment]:
    return _load(path, Appointment)


def _load(path: str | Path, model: type[Record]) -> LoadResult[Record]:
    result: LoadResult[Record] = LoadResult()
    filename = str(path)
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as source:
            lines = source.readlines()
    except (OSError, UnicodeError) as error:
        result.file_errors.append(CsvIssue(filename, None, None, str(error)))
        return result

    reader = csv.reader(lines, strict=True)
    try:
        header = next(reader, [])
    except csv.Error as error:
        result.file_errors.append(CsvIssue(filename, 1, None, str(error)))
        return result

    expected = set(model.model_fields)
    for column in sorted(expected - set(header)):
        result.file_errors.append(CsvIssue(filename, 1, column, "Missing header"))
    for column in sorted(set(header) - expected):
        result.file_errors.append(CsvIssue(filename, 1, column, "Unexpected header"))
    for column in sorted(set(header)):
        if header.count(column) > 1:
            result.file_errors.append(CsvIssue(filename, 1, column, "Duplicate header"))
    if result.file_errors:
        return result

    # Keep the original text of accepted rows for Pandas and map them back to
    # their starting physical lines, including rows with quoted line breaks.
    accepted_lines = lines[:reader.line_num]
    locations: list[int] = []
    while True:
        start = reader.line_num
        try:
            cells = next(reader)
        except StopIteration:
            break
        except csv.Error as error:
            result.file_errors.append(CsvIssue(filename, start + 1, None, str(error)))
            break
        if len(cells) == len(header):
            accepted_lines.extend(lines[start:reader.line_num])
            locations.append(start + 1)
        else:
            issue = CsvIssue(
                filename, start + 1, None,
                f"Expected {len(header)} cells, found {len(cells)}",
            )
            result.rejected_rows.append(RejectedRow(tuple(cells), (issue,)))

    try:
        frame = pd.read_csv(
            StringIO("".join(accepted_lines), newline=""),
            engine="python",
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            skip_blank_lines=False,
        )
    except pd.errors.ParserError as error:
        result.file_errors.append(CsvIssue(filename, None, None, str(error)))
        return result
    if len(frame) != len(locations):
        result.file_errors.append(CsvIssue(filename, None, None, "CSV row count mismatch"))
        return result

    id_field = "patient_id" if model is Patient else "appointment_id"
    for line_number, values in zip(locations, frame.itertuples(index=False, name=None), strict=True):
        normalized = {
            name: value.strip() if name in TRIMMED_FIELDS else value
            for name, value in zip(header, values, strict=True)
        }
        if "visit_summary" in normalized and not normalized["visit_summary"].strip():
            normalized["visit_summary"] = ""
        try:
            record = model.model_validate(normalized)
        except ValidationError as error:
            issues = tuple(
                CsvIssue(
                    filename,
                    line_number,
                    ".".join(str(part) for part in detail["loc"]) or None,
                    detail["msg"],
                )
                for detail in error.errors(include_url=False)
            )
            result.rejected_rows.append(RejectedRow(values, issues, normalized[id_field] or None))
        else:
            result.located_records.append(LocatedRecord(record, SourceReference(filename, line_number)))
    result.rejected_rows.sort(key=lambda row: row.errors[0].line_number or 0)
    return result
