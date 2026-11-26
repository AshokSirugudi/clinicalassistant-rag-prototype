"""Format existing CSV load and validation results for the console."""

from collections import defaultdict

from .csv_loader import LoadResult, SourceReference
from .dataset_validation import ReasonCode, ValidationReport
from .models import Appointment, Patient


def format_data_quality_summary(
    patient_load: LoadResult[Patient],
    appointment_load: LoadResult[Appointment],
    report: ValidationReport,
) -> str:
    """Summarize parsed rows and readiness of the accepted subset."""
    patient_total = len(patient_load.located_records) + len(patient_load.rejected_rows)
    appointment_total = len(appointment_load.located_records) + len(appointment_load.rejected_rows)
    accepted = len(report.accepted.patients) + len(report.accepted.appointments)
    rejected = {source for finding in report.rejected for source in finding.sources}
    review = {source for finding in report.review_required for source in finding.sources}
    lines = [
        "Data quality summary",
        f"Rows read: patients={patient_total}, appointments={appointment_total}",
        f"Accepted: {accepted} | Rejected: {len(rejected)} | Review required: {len(review)}",
    ]
    if rejected:
        by_reason: dict[ReasonCode, set[SourceReference]] = defaultdict(set)
        for finding in report.rejected:
            by_reason[finding.code].update(finding.sources)
        reasons = ", ".join(f"{code.value} ({len(sources)})" for code, sources in sorted(by_reason.items()))
        lines.append(f"Rejection reasons: {reasons}")

    if report.dataset_ready is None:
        lines.append(
            "Dataset ready for ingestion: NO "
            f"(NOT READY, incomplete load, file errors: {len(report.file_errors)})"
        )
    elif rejected or review:
        lines.append("Dataset ready for ingestion: YES (accepted records only)")
    else:
        lines.append("Dataset ready for ingestion: YES")
    return "\n".join(lines)
