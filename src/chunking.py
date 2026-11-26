"""Create one searchable chunk per section of a loaded Markdown document."""

import re

from .document_loader import HEADING, MarkdownDocument
from .models import DocumentChunk, DocumentSourceMetadata


def chunk_document(document: MarkdownDocument) -> list[DocumentChunk]:
    """Split on level-two headings in source order and retain the section text."""
    body = document.body
    headings = list(HEADING.finditer(body))
    sections = headings[1:]
    section_names = tuple(heading[2].strip() for heading in sections)
    if (
        not sections
        or headings[0][1] != "#"
        or headings[0][2].strip() != document.title
        or body[:headings[0].start()].strip()
        or any(heading[1] != "##" for heading in sections)
        or not all(section_names)
        or len(set(section_names)) != len(section_names)
        or section_names != document.section_headings
        or re.search(r"(?m)^##[ \t\r]*$", body)
    ):
        raise ValueError(f"{document.source_path}: malformed Markdown title or section headings")

    chunks = []
    for position, heading in enumerate(sections):
        end = sections[position + 1].start() if position + 1 < len(sections) else len(body)
        section_text = body[heading.end():end].removeprefix("\n")
        name = section_names[position]
        if not section_text.strip():
            raise ValueError(f"{document.source_path}: empty section {name!r}")
        chunks.append(DocumentChunk(
            patient_id=document.patient_id,
            source_id=document.source_id,
            chunk_position=position,
            text=f"# {document.title}\n\n## {name}\n{section_text}",
            source_metadata=DocumentSourceMetadata(
                source_type=document.source_type,
                source_path=document.source_path,
                title=document.title,
                section_heading=name,
                appointment_id=document.appointment_id,
                visit_date=document.visit_date,
                clinician=document.clinician,
            ),
        ))
    return chunks
