import os
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from src.config import AppConfig, load_config


@pytest.fixture(autouse=True)
def clear_config_environment(monkeypatch):
    for variable in (
        "CHROMA_DIR", "EMBEDDING_MODEL", "LLM_MODEL", "RETRIEVAL_TOP_K",
        "RETRIEVAL_MAX_COSINE_DISTANCE", "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)


def test_default_root_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path(__file__).resolve().parents[1]
    config = load_config()
    assert config.patients_csv == root / "data/patients/patients.csv"
    assert config.appointments_csv == root / "data/patients/appointments.csv"
    assert config.visit_notes_dir == root / "data/patients/visit_notes"
    assert config.shared_docs_dir == root / "data/shared"
    assert config.chroma_dir == root / "chroma"


def test_defaults_with_no_credentials_or_directories(tmp_path):
    config = load_config(tmp_path)
    assert config.patients_csv == tmp_path / "data/patients/patients.csv"
    assert config.chroma_dir == tmp_path / "chroma"
    assert config.embedding_model == "text-embedding-3-small"
    assert config.llm_model == "gpt-4.1-mini"
    assert config.top_k == 5
    assert config.openai_api_key is None
    assert not list(tmp_path.iterdir())


def test_dotenv_and_exported_overrides(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "CHROMA_DIR=generated/vectors\nEMBEDDING_MODEL=dotenv-embedding\n"
        "LLM_MODEL=dotenv-llm\nRETRIEVAL_TOP_K=7\nOPENAI_API_KEY=dummy-file-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EMBEDDING_MODEL", "exported-embedding")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-exported-key")
    config = load_config(tmp_path)
    assert config.chroma_dir == tmp_path / "generated/vectors"
    assert not config.chroma_dir.exists()
    assert config.embedding_model == "exported-embedding"
    assert config.llm_model == "dotenv-llm"
    assert config.top_k == 7
    assert isinstance(config.openai_api_key, SecretStr)
    assert config.openai_api_key.get_secret_value() == "dummy-exported-key"
    assert "LLM_MODEL" not in os.environ
    assert "dummy-exported-key" not in repr(config)
    assert "openai_api_key" not in config.model_dump()
    assert "dummy-exported-key" not in config.model_dump_json()


def test_absolute_chroma_override(tmp_path, monkeypatch):
    path = tmp_path / "storage"
    monkeypatch.setenv("CHROMA_DIR", str(path))
    assert load_config(tmp_path / "project").chroma_dir == path
    assert not path.exists()


@pytest.mark.parametrize("value", ["", " \t "])
def test_blank_exported_credential_clears_dotenv_value(tmp_path, monkeypatch, value):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=dummy-file-key\n", encoding="utf-8")
    assert load_config(tmp_path).openai_api_key.get_secret_value() == "dummy-file-key"
    monkeypatch.setenv("OPENAI_API_KEY", value)
    assert load_config(tmp_path).openai_api_key is None


@pytest.mark.parametrize(("variable", "value", "field"), [
    ("RETRIEVAL_TOP_K", "0", "top_k"),
    ("RETRIEVAL_TOP_K", "-1", "top_k"),
    ("RETRIEVAL_TOP_K", "1.0", "top_k"),
    ("RETRIEVAL_TOP_K", "bad", "top_k"),
    ("EMBEDDING_MODEL", "", "embedding_model"),
    ("LLM_MODEL", "  ", "llm_model"),
])
def test_invalid_settings_do_not_expose_credentials(tmp_path, monkeypatch, variable, value, field):
    monkeypatch.setenv(variable, value)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-error-key")
    with pytest.raises(ValidationError, match=field) as error:
        load_config(tmp_path)
    assert "dummy-error-key" not in str(error.value)


def test_direct_model_rejects_boolean_top_k():
    with pytest.raises(ValidationError, match="top_k"):
        AppConfig(top_k=True)


def test_blank_storage_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_DIR", " ")
    with pytest.raises(ValueError, match="CHROMA_DIR"):
        load_config(tmp_path)


@pytest.mark.parametrize(("override", "expected"), [
    ("", None), ("  ", None), (" off ", None), ("0", 0.0), (" 1.25 ", 1.25), ("2", 2.0),
])
def test_max_cosine_distance_environment_overrides(tmp_path, monkeypatch, override, expected):
    assert load_config(tmp_path).max_cosine_distance is None
    (tmp_path / ".env").write_text("RETRIEVAL_MAX_COSINE_DISTANCE=1\n", encoding="utf-8")
    assert load_config(tmp_path).max_cosine_distance == 1.0
    monkeypatch.setenv("RETRIEVAL_MAX_COSINE_DISTANCE", override)
    assert load_config(tmp_path).max_cosine_distance == expected


@pytest.mark.parametrize("value", [
    True, False, -0.01, 2.01, float("nan"), float("inf"), float("-inf"), "NaN", "Infinity", "invalid",
])
def test_invalid_max_cosine_distance(value):
    with pytest.raises(ValidationError, match="max_cosine_distance"):
        AppConfig(max_cosine_distance=value)
