# Invalid CSV fixtures

All records are fictional. Each directory is an independent test case. Appointment cases include a local `patients.csv`.

| Fixture | Expected validation failure |
| --- | --- |
| `duplicate-patient-id/` | Duplicate `patient_id` `P901` in two identical patient rows. |
| `duplicate-appointment-id/` | Duplicate `appointment_id` `A901` in two identical appointment rows. |
| `missing-display-name/` | Required patient field `display_name` is empty. |
| `missing-clinician/` | Required appointment field `clinician` is empty. |
| `invalid-birth-date/` | `date_of_birth` uses the required shape but is an impossible calendar date. |
| `invalid-appointment-date/` | `appointment_date` uses slashes instead of the required `YYYY-MM-DD` format. |
| `unknown-patient-reference/` | Appointment references `P999`, absent from the local patient table. |
| `invalid-appointment-status/` | `rescheduled` is outside the approved status enum. |
| `conflicting-appointment-same-id/` | One appointment ID has two contradictory statuses, violating appointment ID uniqueness. |
| `conflicting-appointment-distinct-ids/` | Two unique IDs share the same patient, date, clinician, and visit reason but have contradictory statuses. The intended failure is a record conflict under a future conflict rule. |

The distinct-ID conflict fixture passes the current CSV schema and primary-key checks. Its expected failure depends on defining a separate record conflict rule. Separate appointments on the same date remain valid under the approved schema.
