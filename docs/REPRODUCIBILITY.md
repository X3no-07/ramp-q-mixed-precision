# Environment and reproduction requirements

## Host environment

Host-side tooling uses Python 3.12 syntax. Dependencies are declared in [requirements.txt](../requirements.txt) and [requirements-dev.txt](../requirements-dev.txt). These files are dependency declarations, not a fully validated environment lock. PyTorch/CUDA and the Orin/TensorRT stack must match their respective platforms.

`gate1_ignore_trainer.py` explicitly requires **Ultralytics 8.4.142**, the reviewed interface version. Keep host and device environments separate; a successful host installation does not certify Orin deployment.

## Obtain omitted inputs first

Check the [material-by-material availability table](DATA_AND_ARTIFACTS.md) before attempting reproduction. The historical release is incomplete (7 of 16 planned ZIP assets), and no complete public artifact set for later reports is provided. Local configuration does not download or reconstruct missing data, weights, or contracts.

## Configure your own data and model

From the repository root, provide existing dataset and checkpoint paths:

```bash
python tools/configure_local.py --data-root /path/to/RUOD-R --model /path/to/best.pt
```

The tool creates `config/gate0.local.json` and `splits/*.local.txt` in the three experimental stages. Git ignores these machine-specific files. Tracked examples use relative paths; export preparation does not change quantization parameters. PowerShell launchers prefer the local configuration and select Python through `RAMPQ_PYTHON`, falling back to `python` on `PATH`.

For a direct Gate 0 invocation, change to its stage directory and supply the local configuration:

```bash
python src/gate0.py inspect --config config/gate0.local.json --project-root .
```

This requires the actual model and dependencies. Subsequent smoke, screen, selection, and holdout runs also require the applicable protocols, data, calibration records, baselines, and frozen contracts. The configuration tool does not create these contracts. Do not bypass a historical integrity failure by relaxing a threshold.

## Tests and limitations

Existing tests are retained in each stage's `tests/` directory. Run relevant `pytest` targets only after configuring their dependencies. Some tests and historical audits bind specific run directories or contracts; the filtered repository does not contain all required historical artifacts.

English-edition verification checks syntax, document links, sensitive patterns, file scope, provenance digests, and Git content. It does not execute training, inference, AP evaluation, or Orin experiments. Static checks are not model validation.

Shared baseline code, configurations, and membership lists are ordinary file copies. Their provenance is recorded in the export manifest. Maintain consistency when editing these copies. Explicit drive or device restrictions in historical tools are retained as protocol safeguards.

Translation changes prose, labels, and report destinations. It does not update frozen source hashes embedded in historical protocols, which still identify their original versions. Reproducing those audits requires the matching original source and artifacts; [artifact availability](DATA_AND_ARTIFACTS.md) explains the scope.
