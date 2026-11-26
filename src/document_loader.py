"""Load whole Markdown documents with validated source identity and patient scope."""

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

from .config import PROJECT_ROOT, AppConfig
from .dataset_validation import ValidationReport
from .models import SHARED_PATIENT_ID, Appointment, AppointmentStatus

NOTE_FIELDS = {"appointment_id", "patient_id", "visit_date", "clinician"}
NOTE_SECTIONS = ("Symptoms", "Assessment", "Recommendations")
NOTE_TITLE = "Fictional clinical visit note"
HEADING = re.compile(r"^(#{1,2})[ \t]+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class MarkdownDocument:
    source_id: str
    source_path: str
    source_type: Literal["visit_note", "shared_guidance"]
    patient_id: str
    title: str
    body: str
    section_headings: tuple[str, ...]
    appointment_id: str | None = None
    visit_date: date | None = None
    clinician: str | None = None


@dataclass(frozen=True)
class DocumentIssue:
    source_path: str | None
    message: str


@dataclass
class DocumentLoadResult:
    documents: list[MarkdownDocument] = field(default_factory=list)
    errors: list[DocumentIssue] = field(default_factory=list)


def _note_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Parse the flat YAML subset used by the notes: four quoted string values."""
    match = re.match(r"\A---[ \t]*\n(.*?)^---[ \t]*\n", text, re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError("Missing or unclosed visit-note front matter")
    metadata = {}
    for line in match[1].splitlines():
        key, _, raw_value = line.partition(":")
        key = key.strip()
        if key not in NOTE_FIELDS or key in metadata:
            raise ValueError(f"Unknown or duplicate front matter field {key!r}")
        try:
            value = json.loads(raw_value)
        except ValueError:
            raise ValueError(f"Front matter field {key} must be a double-quoted string") from None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Front matter field {key} must be a non-blank string")
        metadata[key] = value
    if missing := NOTE_FIELDS - metadata.keys():
        raise ValueError(f"Missing front matter fields {', '.join(sorted(missing))}")
    return metadata, text[match.end():]


def _document_headings(body: str) -> tuple[str, tuple[str, ...]]:
    headings = list(HEADING.finditer(body))
    if not headings or headings[0][1] != "#" or not body.lstrip().startswith("# "):
        raise ValueError("Expected a Markdown title at the start of the document")
    if len(headings) < 2 or any(heading[1] != "##" for heading in headings[1:]):
        raise ValueError("Expected one title and at least one section heading")
    titles = tuple(heading[2].strip() for heading in headings)
    if not all(titles) or len(set(titles[1:])) != len(titles[1:]):
        raise ValueError("Section headings must be non-blank and unique")
    ends = [heading.start() for heading in headings[2:]] + [len(body)]
    if any(not body[heading.end():end].strip() for heading, end in zip(headings[1:], ends, strict=True)):
        raise ValueError("Section guidance or visit content must not be empty")
    return titles[0], titles[1:]


def _visit_note(
    text: str, relative_path: Path, source_path: str, completed: dict[str, Appointment]
) -> MarkdownDocument:
    metadata, body = _note_front_matter(text)
    appointment_id = metadata["appointment_id"]
    appointment = completed.get(appointment_id)
    if appointment is None:
        raise ValueError("Visit note must reference an accepted, completed appointment")
    for field_name, value in (
        ("patient_id", appointment.patient_id),
        ("visit_date", appointment.appointment_date.isoformat()),
        ("clinician", appointment.clinician),
    ):
        if metadata[field_name] != value:
            raise ValueError(f"Visit note {field_name} does not match the source appointment")
    if relative_path.parts != (appointment.patient_id, f"{appointment_id}.md"):
        raise ValueError("Visit note folder or filename does not match the source appointment")
    title, sections = _document_headings(body)
    if sections != NOTE_SECTIONS:
        raise ValueError("Expected Symptoms, Assessment, and Recommendations sections")
    return MarkdownDocument(
        appointment_id, source_path, "visit_note", appointment.patient_id, title, body, sections,
        appointment_id, appointment.appointment_date, appointment.clinician,
    )


def _shared_document(body: str, source_path: str) -> MarkdownDocument:
    if body.lstrip().splitlines()[:1] == ["---"]:
        raise ValueError("Shared guidance cannot have front matter")
    title, sections = _document_headings(body)
    if title.casefold() == NOTE_TITLE.casefold() or set(NOTE_SECTIONS) <= set(sections):
        raise ValueError("Visit-note content cannot be loaded as shared guidance")
    return MarkdownDocument(
        source_path, source_path, "shared_guidance", SHARED_PATIENT_ID, title, body, sections,
    )


def load_documents(
    config: AppConfig, report: ValidationReport, *, project_root: Path = PROJECT_ROOT
) -> DocumentLoadResult:
    """Return accepted Markdown documents and file-level issues without indexing."""
    result = DocumentLoadResult()
    dataset = report.dataset_ready
    if dataset is None:
        result.errors.append(DocumentIssue(None, "Dataset validation is incomplete"))
        return result

    root = project_root.resolve()
    notes_dir, shared_dir = config.visit_notes_dir, config.shared_docs_dir
    if notes_dir.is_relative_to(shared_dir) or shared_dir.is_relative_to(notes_dir):
        result.errors.append(DocumentIssue(None, "Visit-note and shared directories must be separate"))
        return result
    completed = {
        row.record.appointment_id: row.record
        for row in dataset.appointments if row.record.status is AppointmentStatus.COMPLETED
    }
    loaded_notes = set()
    for directory in (notes_dir, shared_dir):
        if not directory.is_relative_to(root) or not directory.is_dir() or directory.resolve() != directory:
            result.errors.append(DocumentIssue(str(directory), "Expected a source directory within the project"))
            continue
        try:
            paths = sorted(directory.rglob("*.md"))
        except OSError:
            result.errors.append(DocumentIssue(str(directory), "Unable to list Markdown documents"))
            continue
        for path in paths:
            source_path = path.relative_to(root).as_posix()
            try:
                if not path.is_file() or path.resolve() != path:
                    raise ValueError("Expected a regular Markdown file without symlink redirection")
                text = path.read_text(encoding="utf-8-sig")
                if directory == notes_dir:
                    document = _visit_note(text, path.relative_to(directory), source_path, completed)
                    loaded_notes.add(document.source_id)
                else:
                    document = _shared_document(text, source_path)
            except (OSError, UnicodeError):
                result.errors.append(DocumentIssue(source_path, "Unable to read UTF-8 Markdown"))
            except (ValueError, RuntimeError) as error:
                result.errors.append(DocumentIssue(source_path, str(error)))
            else:
                result.documents.append(document)

    for appointment_id in sorted(completed.keys() - loaded_notes):
        appointment = completed[appointment_id]
        path = notes_dir / appointment.patient_id / f"{appointment_id}.md"
        source_path = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        result.errors.append(DocumentIssue(source_path, "Missing valid note for completed appointment"))
    return result
