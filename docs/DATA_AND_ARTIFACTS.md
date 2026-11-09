# Data and artifact availability

**Availability checked on November 2, 2026.** This repository is a filtered source-and-report edition. A clone provides the published source and reports; reproducing historical model results requires additional inputs listed below.

## What is included and what to obtain separately

| Material | In the current Git tree? | Where to obtain it / next action | Effect on reproduction |
|---|---|---|---|
| Source code, existing tests, example configurations, reports 00–91, and a selected result summary | Yes | Clone the repository; begin with [current status](STATUS.md). | Sufficient to inspect the code, reports, and packaging checks. |
| Ordered 350-image Selection and 150-image Holdout membership identifiers | Yes | Stage `splits/` directories; tracked paths are relative to the RUOD-R root. | Identifiers do not include the actual images or annotations. |
| Raw images, annotations, and dataset downloads | No | Obtain the matching dataset/version from its original provider under the applicable terms, or supply the existing local research copy. No verified RUOD-R download is advertised by this edition. | Model execution and historical scoring require the correct data, labels, and frozen roles. An arbitrary replacement dataset does not reproduce the reported results. |
| External pretrained checkpoints and project-trained weights | No | Supply your own checkpoint for a new run. For an exact historical run, locate the required checkpoint in the original run records or the historical release, and verify its expected identity. | The example `best.pt` path is a placeholder. A generic pretrained checkpoint does not substitute for the reported trained model. |
| ONNX models and TensorRT engines | No | Locate the matching historical artifact if available, or regenerate with the required weights, export procedure, platform, and precision contract. | Regeneration alone does not establish byte identity, numerical equivalence, or identical device behavior. |
| Prediction arrays, complete run directories, large logs, calibration records, and frozen execution contracts | Selected evidence in Git and release assets | The October 23 archive and the November 8 VisDrone prediction archive are listed below. Consult each report for other required inputs. | Prediction archives support saved-output verification. Exact model reproduction requires the matching checkpoints, data, and environment. |
| External runtimes and environment distributions | No | Install the platform-appropriate dependencies; see [reproduction requirements](REPRODUCIBILITY.md). | Dependency declarations are not a fully validated environment lock. |
| Credentials, private keys, device-access details, machine-specific configuration, and local caches | No | Create your own local configuration and access setup. Access credentials are not distributed. | These are not research artifacts to restore from this repository. |

The omitted local records are retained in the original research collection. This documentation does not promise that every omitted item is publicly downloadable or available on request. Any further sharing requires checking ownership, permissions, intended scope, and the exact files to publish.

## VisDrone held-out confirmation release

The November 8 confirmation has a separate release with the verified 1,610-image prediction archive, checksum manifest, and report bundle. [Download and verify the VisDrone archive](../results/visdrone_testdev_20261108/README.md#download-and-verify).

## Historical release: all 16 planned ZIP assets are available

The [snapshot-2026-10-23 release](https://github.com/X3no-07/ramp-q-mixed-precision/releases/tag/snapshot-2026-10-23) now contains **all 16 original ZIP assets**, parts **001–016**, totaling **10,468,367,310 bytes**. The nine previously missing archives have been added, and the original seven assets were preserved.

All local original ZIP hashes were checked against the historical manifest. GitHub reports every ZIP uploaded with the expected size and SHA256. To recover from interrupted transfers, the nine archives were uploaded in verified byte segments and reconstructed on GitHub-hosted runners. Their reconstructed full ZIPs have the original SHA256 values. Temporary transport segments were removed only after all 16 original ZIPs were verified; no manual concatenation is required to download the release.

- [ARTIFACT_AVAILABILITY.json](../export/ARTIFACT_AVAILABILITY.json): direct download URLs, expected sizes and hashes, and GitHub-reported digests.
- [HISTORICAL_RELEASE_VERIFICATION.json](../export/HISTORICAL_RELEASE_VERIFICATION.json): completion receipt and verification scope.

### Download and verify

Download ZIPs directly from the release, or use the included standard-library tool. From the repository root, list archives without downloading:

```bash
python tools/download_historical_assets.py --list
```

Download and SHA256-verify one archive, for example part 005:

```bash
python tools/download_historical_assets.py --part 5 --output-dir /path/to/rampq-artifacts
```

Omit `--part` to download all 16 ZIPs; repeat it to select several. A Windows output directory such as `D:/rampq-artifacts` can keep large downloads on a larger drive. The tool reuses an existing ZIP only when its size and SHA256 match and refuses to replace a different existing file. It does not extract archives or run experiments.

### Restore the matching historical snapshot

The [historical archive branch](https://github.com/X3no-07/ramp-q-mixed-precision/tree/archive/chinese-snapshot-2026-10-23) retains the original source version, publication file map, and restoration helper. Use a separate checkout of that branch. Its helper expects the original ZIPs under `.release-downloads/` in the historical checkout and verifies restored files against the original logical mapping.

The publication has **2,576 unique archived members** mapped to **2,857 logical result files**. It is a filtered October 23 snapshot; completing it does not publish the entire original research directory, raw datasets, or a complete artifact set for later reports 51–91. These binaries correspond to the earlier source snapshot rather than the current translated source tree. Exact historical reproduction still requires its matching environment, data, and frozen contracts.

The manually dispatched [assembly workflow](../.github/workflows/assemble-historical-release.yml) is a publication-maintenance tool. It checks existing ZIP identities, verifies required transport segments, reconstructs the original bytes, and requires the server-reported full ZIP SHA256 to match. It does not train or evaluate models. Its immutable [assembly plan](../export/HISTORICAL_ASSEMBLY_PLAN.json) records transport identities and does not assert that temporary segments remain available after cleanup.

This update changes artifact availability only. Recorded failures, numerical guardrails, and scientific acceptance states remain unchanged.

## Configure local data and a checkpoint

After obtaining the actual inputs, run from the repository root:

```bash
python tools/configure_local.py --data-root /path/to/RUOD-R --model /path/to/best.pt
```

This writes ignored local configuration and image-path lists for the three experimental stages. It does **not** download data, create a model, generate missing contracts, or restore historical results. Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before running stage-specific commands.

## Provenance and references to omitted evidence

[SOURCE_FILES.jsonl](../export/SOURCE_FILES.jsonl) records pre-export source digests and current exported paths and digests. English translation, portable configuration, and navigation updates have different file digests from the frozen original versions.

Links to local-only evidence in the reading copies resolve to this page. Run identifiers, metrics, negative results, and scientific limitations remain in the reports; a redirected reference is not a download link or evidence that the complete run is included. A hardware/software scope note replaces the detailed device-access record.

Historical manifests can refer to omitted members and original code versions. Do not apply an old whole-delivery audit to this filtered edition or interpret a missing file as evidence that the original experiment never produced it. Verify the current export first, then check the exact historical inputs required.

Path normalization changes split-list file hashes; it does not change membership order, replace frozen contracts, or authorize a new use of independent Holdout data. Exact historical reproduction requires the corresponding original code, contracts, data, and artifacts.
