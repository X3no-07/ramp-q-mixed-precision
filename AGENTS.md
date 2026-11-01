# Repository guidance

This is the English source-and-report edition of RAMP-Q. Read `docs/STATUS.md` before interpreting historical reports. Stage locations and entry points are listed in `README.md`.

- Preserve negative results, fixed decision rules, data boundaries, and the distinction between engineering completion and scientific acceptance.
- Historical scripts may require complete local artifacts and frozen contracts that are not distributed here. Inspect their parameters and output destinations before execution.
- `tools/check_repository.py` performs static packaging checks; it does not certify experiments or runtime behavior.
- When editing an imported file, update its `export_sha256` in `export/SOURCE_FILES.jsonl`. Original source digests identify the pre-export records; translations have different file digests.
- Keep machine-specific `*.local.json` and `*.local.txt`, credentials, datasets, weights, engines, and large numerical outputs out of Git.
- Keep the materialized shared `gate0.py`, baseline configurations, and split membership lists consistent across the stages that reuse them.
