"""Load application settings without changing the process environment."""

import os
import re
from pathlib import Path
from typing import Annotated

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from .models import NonBlankString

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATH_DEFAULTS = {
    "patients_csv": "data/patients/patients.csv",
    "appointments_csv": "data/patients/appointments.csv",
    "visit_notes_dir": "data/patients/visit_notes",
    "shared_docs_dir": "data/shared",
    "chroma_dir": "chroma",
}
ENV_OVERRIDES = {
    "embedding_model": "EMBEDDING_MODEL",
    "llm_model": "LLM_MODEL",
    "top_k": "RETRIEVAL_TOP_K",
    "max_cosine_distance": "RETRIEVAL_MAX_COSINE_DISTANCE",
    "openai_api_key": "OPENAI_API_KEY",
}


class AppConfig(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True, frozen=True)

    patients_csv: Path = PROJECT_ROOT / PATH_DEFAULTS["patients_csv"]
    appointments_csv: Path = PROJECT_ROOT / PATH_DEFAULTS["appointments_csv"]
    visit_notes_dir: Path = PROJECT_ROOT / PATH_DEFAULTS["visit_notes_dir"]
    shared_docs_dir: Path = PROJECT_ROOT / PATH_DEFAULTS["shared_docs_dir"]
    chroma_dir: Path = PROJECT_ROOT / PATH_DEFAULTS["chroma_dir"]
    embedding_model: NonBlankString = "text-embedding-3-small"
    llm_model: NonBlankString = "gpt-4.1-mini"
    top_k: Annotated[int, Field(strict=True, gt=0)] = 5
    max_cosine_distance: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)] | None = None
    openai_api_key: SecretStr | None = Field(default=None, repr=False, exclude=True)

    @field_validator("top_k", mode="before")
    @classmethod
    def parse_top_k(cls, value: object) -> object:
        if isinstance(value, str):
            if not re.fullmatch(r"[0-9]+", value.strip()):
                raise ValueError("top_k must be a positive integer")
            return int(value)
        return value

    @field_validator("max_cosine_distance", mode="before")
    @classmethod
    def parse_max_cosine_distance(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            return None if value.lower() in ("", "off") else value
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("max_cosine_distance must be a finite number between 0 and 2, or off")
        return value

    @field_validator("openai_api_key")
    @classmethod
    def blank_key_to_none(cls, value: SecretStr | None) -> SecretStr | None:
        return value if value is not None and value.get_secret_value().strip() else None


def load_config(project_root: str | Path | None = None) -> AppConfig:
    """Load defaults, then repository .env values, then exported overrides."""
    root = PROJECT_ROOT if project_root is None else Path(project_root).resolve()
    file_values = dotenv_values(root / ".env", interpolate=False)
    chroma_override = os.environ.get("CHROMA_DIR", file_values.get("CHROMA_DIR"))
    paths = {name: (root / relative).resolve() for name, relative in PATH_DEFAULTS.items()}
    if chroma_override is not None:
        if not chroma_override.strip():
            raise ValueError("CHROMA_DIR must not be blank")
        paths["chroma_dir"] = (root / chroma_override).resolve()

    values = {}
    for field, variable in ENV_OVERRIDES.items():
        value = os.environ.get(variable, file_values.get(variable))
        if value is not None:
            values[field] = value
    return AppConfig.model_validate({**paths, **values})
