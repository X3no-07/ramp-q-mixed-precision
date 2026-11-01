# Gate1: Data prepared, GPU training pre-check passed



Date: 2026-10-22. This round completes VisDrone data preparation and training workflow inspection, status is **PREPARATION_AND_ENGINEERING_PREFLIGHT_PASSED**. The formal floating-point baseline has not yet been trained, and the quantization mechanism and selection model acceptance for Gate1 have not yet begun.

# # What was accomplished in this round

The training and development sets of VisDrone-DET have been obtained, with the original ZIP files, images, and raw annotations preserved. The test-dev compressed package has been separately sealed and its internal contents are not listed, decompressed, inspected, or annotated. Data sources are consistent with the [official VisDrone task entry](https://github.com/VisDrone/VisDrone-Dataset), and the download mirror used is the one referenced by the Ultralytics 8.4.142 data configuration for [Ultralytics fixed release resources](https://github.com/ultralytics/assets/releases/tag/v0.0.0).

| Data portion | Number of original images | Current usage |
|---|---:|---|
| Official train set | 6,471 | Reserved for training, validation, calibration, and debugging by group; the remainder will be used for subsequent training |
| Official val set | 548 | Selection; this round only involves counting and format conversion, with no detection results or module selection performed |
| test-dev | The internal file count within the package was not checked in this round | The original ZIP files are preserved; only the file size and SHA values are verified |

The two prepared datasets collectively retain **7,019 images and 393,719 non-empty original annotations**. The YOLO-formatted labels saved here are solely for training convenience, with classes 1–10 mapped to 0–9; region, `others`, original physical row identities, and cropping records are preserved in sidecar files. Training bounding boxes are cropped according to the original image boundaries, and no modifications are made to the original image or annotation bytes.

The independent verification tool did not invoke the preparation script's conversion and grouping functions. It rechecked the original ZIP file and the delivery checklist contents, including **21,077 files**, 7,019 images in terms of dimensions, image content, original annotations, YOLO labels, and two annotation sidecar files, and reconstructed the prefix-to-content SHA connection components. 8 items passed verification; its conclusion is limited to **PASS_PREPARATION_FIDELITY_ONLY**, and does not represent generalization independence, official AP, or Gate1 scientific conclusion acceptance.

# # The current split is a reproducible preparation version

| Role | Images | Number of grouped connections | Description |
|---|---:|---:|---|
| train | 4,885 | 155 | Candidate data for floating-point training in the future |
| training_val | 409 | 2 | Internal validation during training; coverage is insufficient and requires improvement prior to formal training |
| debug | 109 | 6 | Engineering debugging; only 64 images from this group were used for training inspection in this round |
| calibration_pool | 571 | 18 | Isolated from training, debugging, and selection; calibration has not yet been performed |
| selection | 548 | 75 | Complete official validation set, not yet inspected or scored |
| quarantine | 497 | 27 | Cross-split associations, duplicates, or annotation quality issues causing isolation |

A full split proposal is generated using fixed seed `20261022`; groups are established based on the prefix of filenames and exact image SHA values. These remain scene proxies and are not considered verified real-world scenes.

Three calibration lists use seed `2026102201/2026102202/2026102203`, each containing 64 images, all drawn from the calibration pool with pairwise non-overlapping content SHAs. These images were not loaded for calibration or model inference in this round.

3 groups of exact duplicate records and 24 cross-train/val prefix or content-linked records were identified. Training components associated with these links are isolated; the official validation set is retained. Exact duplicates and prefix links have been verified; near-duplicate images and real-world scene associations remain under investigation.

# # Anomalies in original annotations are already well-documented

A full scan revealed two bounding boxes originally labeled as ignored regions with zero height in the training set, and one additional zero-height box with SHA `score=1,category=4`; no non-positive-sized boxes were found in the development set.

Original ignored boxes are fully preserved and do not generate training targets. Abnormal positive boxes are `train/9999985_00000_d_0000020:11`, originally on line `611,158,4,0,1,4,0,0`. This line retains its original score, class, original text, and physical line ID, and is marked with `excluded_for_preparation_quality=true`; no false claims are made that these annotations were originally ignored.

All 79 training images within the entire associated component are quarantined; these 79 images are included in the aforementioned 497 images and are not used in training, training validation, debugging, or calibration. If similar anomalies appear in the development set, they will be strictly rejected. Relevant code has passed **28 tests**; the first failure record and the original script at the time are preserved.

# # The computer is now capable of completing the training workflow

This machine is an NVIDIA RTX 5060 Laptop GPU with 8,151 MiB of VRAM; the environment is Python 3.12.10, PyTorch 2.10.0+cu128, Ultralytics 8.4.142. CUDA operations and 10-architecture checks have passed; the mapping of 88 convolutions from G1 to G8 has been re-verified.

A general YOLO11n weight from [Ultralytics v8.3.0 official resources](https://github.com/ultralytics/assets/releases/tag/v8.3.0) is obtained, and its SHA matches exactly with the existing local general weight SHA: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`. This is a general initialization, and RUOD-R detection weights are not used as a baseline for VisDrone.

The pre-check training setup is as follows:

- 64 images are selected from the debug list for training, and 32 images from training_val for validation, all drawn from the official train set.
- Training for 1 epoch, 640×640 resolution, actual batch size=4, workers=0, AMP disabled, seed=20261022.
- The 96 original images and labels are individually verified for source role and SHA, then copied to a dedicated pre-check directory. Caches are also written to this copy directory. The SHA of the original 96 image-label pairs remains unchanged after pre-check; no new cache entries are added in the preparation directory.
- Training, end-of-epoch validation, and final best-weight validation by the framework are completed, with the entire process taking approximately **33.66 seconds**. The best and last weights are saved; checks for 10 class names, 88 convolutions, and model parameter bounds have passed.

This time includes overhead from the small-scale check and cannot be used to infer the full training duration. Detection scores in the logs are solely for engineering diagnostics; the model is trained only once and cannot serve as a valid floating-point baseline, nor can it be used as evidence for VisDrone official AP, quantization recovery, or model selection superiority. The framework's built-in OOM/NaN recovery policies are disclosed in the runtime logs; no external automatic restarts of training occurred.

A configuration directory write warning was generated during the initial import. Subsequent entry points have been modified to first create a dedicated configuration directory for this round and perform independent path checks; the original source code, console output, and model results are preserved, and no retraining has been run to fix this setting.

# # Items still required to be addressed before establishing the official baseline

The most clearly defined design issue is the narrow coverage of the training-in-validation set scenario: only 409 images contain 2 connected groups, one of which accounts for 407 images. This preliminary check only confirms that the data and training workflow are executable; prior to formal training, a revised split with broader group coverage must be designed, and the new version should be recorded while the current proposal is retained.

The next steps are to proceed in sequence:

1. Improve the coverage of training-validation group splits and verify real-world scene identities and near-duplicate images; maintain the boundaries of calibration, selection, and final confirmation data.
2. Implement a fixed detection evaluation for ignored regions. Standard YOLO training may still treat ignored regions as background negative samples, and the built-in validation does not fully implement VisDrone's ignored region rules; sidecar files cannot substitute for a complete evaluation implementation.
3. Freeze the list of frozen floating-point training, initialization, all parameters, training budget, and checkpoint selection rules, and then train a model suitable for quantization research.
4. Complete the numerical, module, and research acceptance agreement according to the [Gate1 Research Plan](research_design/04_gate1_visdrone_independent_validation_protocol.md); after that, proceed with selection interventions and final holdout confirmation.

There is currently no module screening, quantization metrics, independent validation, or performance conclusion for Gate1. The historical conclusions for Gate0.5 remain valid based on [Report #39](../../02_gate05_orin_deployment/docs/39_gate05_complete_150_image_acceptance_results.md), and the original complete numerical guardrail failure is preserved.

# # Data and Review Entry Points

- [Prepare Results and Original Content Identity](../../docs/DATA_AND_ARTIFACTS.md), [Per-Image Checklist](../../docs/DATA_AND_ARTIFACTS.md), [Split Proposal](../../docs/DATA_AND_ARTIFACTS.md), [Full File List](../../docs/DATA_AND_ARTIFACTS.md).
- [Independent Preparation for Audit](../../docs/DATA_AND_ARTIFACTS.md), [Test Records](../../docs/DATA_AND_ARTIFACTS.md).
- [Training Pre-check Records](../../docs/DATA_AND_ARTIFACTS.md), [Actual Source Code of This Run](../../docs/DATA_AND_ARTIFACTS.md), [Subsequent Integrity Review](../../docs/DATA_AND_ARTIFACTS.md).
- [Final Test Set Seal Record](../../docs/DATA_AND_ARTIFACTS.md), [Official General Weights Identity](../../docs/DATA_AND_ARTIFACTS.md).
- Engineering entry points: [Data Preparation](../scripts/prepare_gate1_visdrone.py), [Independent Verification](../scripts/verify_gate1_prepared_data.py), [Training Pre-check](../scripts/run_gate1_training_preflight.py). These entry points do not cover existing outputs; new attempts must use new directories.

Downloaded package matches the published resource ID and byte count; local SHA records its identity. The published API does not provide digests for these resources, so local SHA computations cannot be declared as the publisher's provided checksum. The test-dev seal belongs to execution flow constraints and is not related to cryptographic or OS-level read access isolation.
