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
