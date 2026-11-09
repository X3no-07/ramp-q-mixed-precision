# VisDrone test-dev confirmation, November 8, 2026

[Report 95](../../04_research_roadmap/docs/95_visdrone_held_out_confirmation.md) covers the protocol, results, and limitations.

| File | Contents |
|---|---|
| [completed.json](completed.json) | Completed image and forward counts, elapsed time, and training update count |
| [qualification.json](qualification.json) | All 35 development-image qualification comparisons |
| [analysis.json](analysis.json) | Recovery, calibration redraw, fixed module, and label association statistics |
| [all_63_tag_audits.csv](all_63_tag_audits.csv) | All prespecified comparisons and bootstrap intervals |
| [all_19_arms_fixed_abs0_pool.csv](all_19_arms_fixed_abs0_pool.csv) | All configurations evaluated against the draw 0 target pool |
| [analysis_contract.public.json](analysis_contract.public.json) | Scientific fields from the frozen analysis contract |
| [image_output_manifest.json](image_output_manifest.json) | Expected paths, sizes and SHA256 for all 1,610 prediction files |
| [provenance.json](provenance.json) | SHA256 digests of the original source records |

The prediction archive contains generated model outputs and match records. The data required to rerun the original models, including VisDrone images and annotations, checkpoints, and calibration inputs, is separate.

## Download and verify

The [VisDrone confirmation release](https://github.com/X3no-07/ramp-q-mixed-precision/releases/tag/visdrone-testdev-2026-11-08) contains 96 numbered archive parts, their checksum manifest, and a ZIP with the report and result tables. The reassembled prediction archive is 12,776,232,960 bytes. All 98 assets were verified against GitHub's reported sizes and SHA256 digests. See the [release verification record](GITHUB_RELEASE_VERIFICATION.json).

Download `prediction_archive.json` and all 96 numbered parts into one directory. From the repository root, run the assembly helper with Python 3.12:

```bash
python tools/assemble_visdrone_confirmation.py /path/to/parts /path/to/visdrone-20261108-predictions.tar
```

The helper verifies each part, the reassembled archive, and the SHA256 of all 1,610 image output files. It leaves the files in the TAR and performs no inference.
