# Export provenance and validation

- [SOURCE_FILES.jsonl](SOURCE_FILES.jsonl): pre-export source digests, exported paths, and current exported digests.
- [ARTIFACT_AVAILABILITY.json](ARTIFACT_AVAILABILITY.json): all 16 direct ZIP download URLs and expected/server-reported hashes.
- [HISTORICAL_RELEASE_VERIFICATION.json](HISTORICAL_RELEASE_VERIFICATION.json): original ZIP publication completion and verification receipt.
- [HISTORICAL_ASSEMBLY_PLAN.json](HISTORICAL_ASSEMBLY_PLAN.json): immutable transport-segment plan used for recovery, rather than a current download inventory.
- [EXCLUDED_SCOPE.json](EXCLUDED_SCOPE.json): excluded categories and selected operational files.
- [REFERENCE_MAP.json](REFERENCE_MAP.json): scope of local-only evidence references redirected to the artifact guide.
- [VALIDATION.json](VALIDATION.json): completed repository checks.
- [TRANSLATION_AUDIT.json](TRANSLATION_AUDIT.json): translation integrity and source-structure checks.
- [ENGLISH_EDITION.json](ENGLISH_EDITION.json): translation scope and verification boundaries.

Imported files are distinguished from newly authored navigation and helper tools. The original local research collection and a separate Chinese snapshot are retained outside the submitted tree.

English drafting was performed locally and followed by terminology normalization, review of the main research decisions, and structural checks. Numeric values, formulas, scientific state identifiers, and negative outcomes are not new experimental measurements. The English edition is a translated reading/source copy, not a byte-identical replacement for frozen historical records.

Packaging checks do not certify scientific acceptance, runtime reproducibility, or device performance.
