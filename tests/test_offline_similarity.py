"""Synthetic ranking experiment. The assigned vectors do not measure semantic relevance."""

from math import fsum, hypot, isfinite

import pytest

from src.chunking import chunk_document
from src.config import AppConfig
from src.csv_loader import load_appointments, load_patients
from src.dataset_validation import validate_dataset
from src.document_loader import load_documents

QUERY = (1.0, 0.0, 0.0)
NEUTRAL = (0.0, 0.0, 1.0)
KNEE = ("A001", "Recommendations")
OTHER_PATIENT = ("A034", "Recommendations")
ACTIVITY = ("data/shared/activity-related-discomfort.md", "Tracking Activity and Function")
BLOOD_PRESSURE = ("data/shared/home-blood-pressure-log.md", "What to Record")
SYNTHETIC_VECTORS = {
    OTHER_PATIENT: (1.0, 0.0, 0.0),
    KNEE: (0.8, 0.6, 0.0),
    ACTIVITY: (0.6, 0.8, 0.0),
    BLOOD_PRESSURE: NEUTRAL,
}


def cosine_similarity(left, right):
    if not left or len(left) != len(right):
        raise ValueError("Vectors must have the same nonzero dimension")
    left_norm, right_norm = hypot(*left), hypot(*right)
    if not all(isfinite(norm) and norm > 0 for norm in (left_norm, right_norm)):
        raise ValueError("Vectors must have finite, nonzero norms")
    return fsum((a / left_norm) * (b / right_norm) for a, b in zip(left, right, strict=True))


def source_section(chunk):
    return chunk.source_id, chunk.source_metadata.section_heading


def rank_synthetic(chunks, *, patient_id=None):
    eligible = [
        chunk for chunk in chunks
        if patient_id is None or chunk.patient_id in {patient_id, "SHARED"}
    ]
    scores = [
        (chunk, cosine_similarity(QUERY, SYNTHETIC_VECTORS.get(source_section(chunk), NEUTRAL)))
        for chunk in eligible
    ]
    return sorted(scores, key=lambda result: (-result[1], result[0].chunk_id))


def test_synthetic_rankings_and_patient_isolation():
    config = AppConfig()
    report = validate_dataset(load_patients(config.patients_csv), load_appointments(config.appointments_csv))
    assert report.dataset_ready is not None
    documents = load_documents(config, report)
    assert not documents.errors
    chunks = [chunk for doc in documents.documents for chunk in chunk_document(doc)]
    assert len(chunks) == 72
    assert all(sum(source_section(chunk) == key for chunk in chunks) == 1 for key in SYNTHETIC_VECTORS)

    unfiltered = rank_synthetic(chunks)
    assert [source_section(chunk) for chunk, _ in unfiltered[:3]] == [OTHER_PATIENT, KNEE, ACTIVITY]
    assert [score for _, score in unfiltered[:3]] == pytest.approx([1.0, 0.8, 0.6])
    assert all(score == 0 for _, score in unfiltered[3:])

    eligible = rank_synthetic(chunks, patient_id="P001")
    assert len(eligible) == 18
    assert {chunk.patient_id for chunk, _ in eligible} == {"P001", "SHARED"}
    top_two = eligible[:2]
    assert [source_section(chunk) for chunk, _ in top_two] == [KNEE, ACTIVITY]
    assert [score for _, score in top_two] == pytest.approx([0.8, 0.6])
    assert len([chunk for chunk, _ in unfiltered[:2] if chunk.patient_id in {"P001", "SHARED"}]) == 1
    assert [chunk.chunk_id for chunk, _ in eligible[2:]] == sorted(
        chunk.chunk_id for chunk, _ in eligible[2:]
    )


@pytest.mark.parametrize(("left", "right", "score"), [
    ((2.0, 0.0, 0.0), QUERY, 1.0),
    ((-2.0, 0.0, 0.0), QUERY, -1.0),
    (NEUTRAL, QUERY, 0.0),
])
def test_cosine_normalizes_magnitude(left, right, score):
    assert cosine_similarity(left, right) == pytest.approx(score)


@pytest.mark.parametrize(("left", "right", "message"), [
    ((), (), "nonzero dimension"),
    ((1.0, 0.0), QUERY, "same nonzero dimension"),
    (QUERY, (0.0, 0.0, 0.0), "finite, nonzero norms"),
    ((float("inf"), 0.0, 0.0), QUERY, "finite, nonzero norms"),
])
def test_cosine_rejects_invalid_vectors(left, right, message):
    with pytest.raises(ValueError, match=message):
        cosine_similarity(left, right)
