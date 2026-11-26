from types import SimpleNamespace

import pytest
from openai import OpenAIError

from src.config import AppConfig
from src.embeddings import EmbeddingError, embed_chunks
from src.models import DocumentChunk, DocumentSourceMetadata


def make_chunks(count):
    metadata = DocumentSourceMetadata(
        source_type="visit_note",
        source_path="data/patients/visit_notes/P001/A001.md",
        title="Fictional visit",
        section_heading="Assessment",
        appointment_id="A001",
        visit_date="2026-10-10",
        clinician="Fictional Clinician",
    )
    return [
        DocumentChunk(
            patient_id="P001", source_id="A001", chunk_position=position,
            text=f"Fictional section {position}", source_metadata=metadata,
        )
        for position in range(count)
    ]


class FakeClient:
    def __init__(self, dimension=1536, transform=None, fail_batch=None):
        self.embeddings = self
        self.calls = []
        self.dimension = dimension
        self.transform = transform
        self.fail_batch = fail_batch

    def create(self, **request):
        self.calls.append(request)
        if len(self.calls) == self.fail_batch:
            raise OpenAIError("dummy-secret and Fictional section 32")
        data = [
            SimpleNamespace(index=index, embedding=[float(text.split()[-1])] * self.dimension)
            for index, text in reversed(list(enumerate(request["input"])))
        ]
        return SimpleNamespace(data=self.transform(data) if self.transform else data)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def test_batches_and_shuffled_indices_send_text_only():
    chunks = make_chunks(72)
    client = FakeClient()
    result = embed_chunks(chunks, AppConfig(), client=client)
    assert [len(call["input"]) for call in client.calls] == [32, 32, 8]
    assert [text for call in client.calls for text in call["input"]] == [chunk.text for chunk in chunks]
    assert all(set(call) == {"input", "model", "encoding_format"} for call in client.calls)
    assert all(call["model"] == "text-embedding-3-small" for call in client.calls)
    assert result == {chunk.chunk_id: [float(i)] * 1536 for i, chunk in enumerate(chunks)}


@pytest.mark.parametrize(("transform", "message"), [
    (lambda data: data[:-1], "missing response indices"),
    (lambda data: [data[0], data[0]], "duplicate response index"),
    (lambda data: [SimpleNamespace(index=-1, embedding=data[0].embedding)], "invalid response index"),
    (lambda data: [SimpleNamespace(index=1, embedding=[1.0])], "expected 1536 vector dimensions"),
    (lambda data: [SimpleNamespace(index=1, embedding=[float("nan")] * 1536)], "non-finite"),
    (lambda data: [SimpleNamespace(index=1, embedding=["bad"] * 1536)], "non-numeric"),
    (lambda data: None, "invalid response data"),
])
def test_invalid_response_is_rejected(transform, message):
    client = FakeClient(transform=transform)
    with pytest.raises(EmbeddingError, match=message):
        embed_chunks(make_chunks(2), AppConfig(), client=client)


def test_duplicate_ids_are_rejected_before_requests():
    chunk, = make_chunks(1)
    client = FakeClient()
    with pytest.raises(EmbeddingError, match="Duplicate chunk IDs"):
        embed_chunks([chunk, chunk], AppConfig(), client=client)
    assert not client.calls


def test_later_api_failure_stops_and_redacts_details():
    client = FakeClient(fail_batch=2)
    with pytest.raises(EmbeddingError, match=r"batch 2 failed \(OpenAIError\)") as error:
        embed_chunks(make_chunks(72), AppConfig(openai_api_key="dummy-secret"), client=client)
    assert len(client.calls) == 2
    assert "dummy-secret" not in str(error.value)
    assert "Fictional section" not in str(error.value)
    assert error.value.__suppress_context__


def test_missing_credentials_and_empty_input_do_not_initialize_sdk(monkeypatch):
    def unexpected_client(**kwargs):
        pytest.fail("SDK client must not be initialized")

    monkeypatch.setattr("src.embeddings.OpenAI", unexpected_client)
    assert embed_chunks([], AppConfig()) == {}
    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
        embed_chunks(make_chunks(1), AppConfig())


def test_sdk_settings_and_configured_model(monkeypatch):
    client = FakeClient(dimension=3072)

    def fake_openai(*, api_key, timeout, max_retries):
        assert (api_key, timeout, max_retries) == ("dummy-secret", 30.0, 2)
        return client

    monkeypatch.setattr("src.embeddings.OpenAI", fake_openai)
    config = AppConfig(openai_api_key="dummy-secret", embedding_model="text-embedding-3-large")
    result = embed_chunks(make_chunks(1), config)
    assert client.calls[0]["model"] == "text-embedding-3-large"
    assert len(next(iter(result.values()))) == 3072


def test_unknown_dimensions_are_rejected_before_requests():
    client = FakeClient()
    config = AppConfig(embedding_model="unsupported")
    with pytest.raises(EmbeddingError, match="no known vector dimensions"):
        embed_chunks(make_chunks(1), config, client=client)
    assert not client.calls
