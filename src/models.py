"""Record and document models for the Clinical RAG prototype."""

import json
import re
from datetime import date
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field
from pydantic import computed_field, field_validator, model_validator

SHARED_PATIENT_ID = "SHARED"


def require_non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def check_iso_date(value: object) -> object:
    if type(value) is date:
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        return value
    raise ValueError("must be a date in YYYY-MM-DD format")


NonBlankString = Annotated[str, AfterValidator(require_non_blank)]
ISODate = Annotated[date, BeforeValidator(check_iso_date)]


class Patient(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: NonBlankString
    display_name: NonBlankString
    date_of_birth: ISODate

    @field_validator("patient_id")
    @classmethod
    def reject_shared_identifier(cls, value: str) -> str:
        if value == SHARED_PATIENT_ID:
            raise ValueError("SHARED is reserved for shared guidance")
        return value


class AppointmentStatus(StrEnum):
    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"


class Appointment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    appointment_id: NonBlankString
    patient_id: NonBlankString
    appointment_date: ISODate
    clinician: NonBlankString
    status: AppointmentStatus
    visit_reason: NonBlankString
    visit_summary: NonBlankString | None = None

    @field_validator("visit_summary", mode="before")
    @classmethod
    def empty_summary_to_none(cls, value: object) -> object:
        return None if value == "" else value

    @model_validator(mode="after")
    def require_completed_summary(self) -> Self:
        if self.status == AppointmentStatus.COMPLETED and self.visit_summary is None:
            raise ValueError("completed appointments require a visit_summary")
        return self


class DocumentSourceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal["visit_note", "shared_guidance"]
    source_path: NonBlankString
    title: NonBlankString
    section_heading: NonBlankString
    appointment_id: NonBlankString | None = None
    visit_date: ISODate | None = None
    clinician: NonBlankString | None = None

    @model_validator(mode="after")
    def check_source_fields(self) -> Self:
        visit_fields = (self.appointment_id, self.visit_date, self.clinician)
        if self.source_type == "visit_note" and any(value is None for value in visit_fields):
            raise ValueError("visit notes require appointment_id, visit_date, and clinician")
        if self.source_type == "shared_guidance" and any(value is not None for value in visit_fields):
            raise ValueError("shared guidance cannot have visit-specific metadata")
        return self


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: NonBlankString
    source_id: NonBlankString
    chunk_position: Annotated[int, Field(strict=True, ge=0)]
    text: NonBlankString
    source_metadata: DocumentSourceMetadata

    @model_validator(mode="after")
    def check_source_identity(self) -> Self:
        metadata = self.source_metadata
        shared = metadata.source_type == "shared_guidance"
        if shared != (self.patient_id == SHARED_PATIENT_ID):
            raise ValueError("only shared guidance uses patient_id SHARED")
        expected_source_id = metadata.source_path if shared else metadata.appointment_id
        if self.source_id != expected_source_id:
            raise ValueError("source_id must match the shared path or visit appointment ID")
        return self

    @computed_field
    @property
    def chunk_id(self) -> str:
        """Hash the source, zero-based position, and exact text with unambiguous framing."""
        payload = json.dumps(
            [self.source_id, self.chunk_position, self.text], ensure_ascii=False
        ).encode("utf-8")
        return sha256(payload).hexdigest()

    def to_chroma_metadata(self) -> dict[str, str | int]:
        """Return scalar filter and citation fields with absent values omitted."""
        metadata = self.source_metadata.model_dump(mode="json", exclude_none=True)
        return {
            "patient_id": self.patient_id,
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "chunk_position": self.chunk_position,
            **metadata,
        }
