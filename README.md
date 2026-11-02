# RAMP-Q: Object-Level Quantization Failure and Recovery

Research software and chronological reports on quantization failures, module-level precision interventions, and reliable recovery in small-object detection.

This English edition was prepared on **November 2, 2026**, from the October 29 source-and-report snapshot. Report dates describe the original experiments; translation does not add experimental evidence.

## Start here

- [Current status and research decisions](docs/STATUS.md)
- [Environment and reproduction requirements](docs/REPRODUCIBILITY.md)
- [Data and artifact availability](docs/DATA_AND_ARTIFACTS.md)
- [Export provenance and validation](export/README.md)

## Research stages

| Stage | Scope | Entry point |
|---|---|---|
| Gate 0 | RUOD-R simulated quantization, object-level failures, and module recovery; reports 00–22 | [Mechanism screening](01_gate0_mechanism_screening/README.md) |
| Gate 0.5 | Orin/TensorRT deployment, numerical diagnostics, and 150-image acceptance; reports 23–39 | [Deployment validation](02_gate05_orin_deployment/README.md) |
| Gate 1 | VisDrone protocols, evaluation, baselines, and correction experiments; reports 40–90 | [VisDrone validation](03_gate1_visdrone_validation/README.md) |
| Research roadmap | Supervision-decoupling proposal, report 91, and earlier proposals | [Research roadmap](04_research_roadmap/README.md) |

Each stage uses `docs/`, `src/`, `scripts/`, `config/`, `tests/`, `artifacts/`, and `logs/`. Shared baseline files are materialized copies rather than Windows directory junctions. Reports retain their original sequence, including negative results.

## What the evidence supports

- **Gate 0:** the simulations exhibit object-level failure and recovery under module precision interventions. General superiority over conventional module-selection methods has not been established.
- **Gate 0.5:** the selected real deployment configurations satisfy a limited same-target recovery existence criterion. The original full-model PyTorch/ORT numerical guardrail remains `FAILED_UNCHANGED`.
- **Gate 1:** the three-arm experiment in report 90 completed training and saved-artifact audits, but intervention-guided correction failed both predefined joint criteria. Report 91 proposes a new hypothesis and has not established its effectiveness.

Implementation readiness, artifact integrity, scientific acceptance, and measured device performance are distinct claims.

## Repository checks

With Python 3.12, run from the repository root:

```bash
python tools/check_repository.py
```

This performs static syntax, link, provenance, file-scope, and sensitive-pattern checks. It does not train a model, run inference, compute AP, or validate a device. Model-related tests require the dependencies, data, and contracts described in the reproduction guide.

This repository contains a filtered source-and-report snapshot. Raw datasets, model weights, engines, prediction arrays, and complete historical run directories are not included. All **16 planned October 23 ZIP assets** are now uploaded, with local and GitHub-reported SHA256 values verified. This historical release does not cover all later reports. See [what is omitted, how to obtain inputs, and current release availability](docs/DATA_AND_ARTIFACTS.md).

Attribution and third-party licensing information are retained in [NOTICE.md](NOTICE.md).
