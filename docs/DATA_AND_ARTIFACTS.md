# Data and artifact availability

**Availability checked on November 2, 2026.** This repository is a filtered source-and-report edition. A clone provides the published source and reports; reproducing historical model results requires additional inputs listed below.

## What is included and what to obtain separately

| Material | In the current Git tree? | Where to obtain it / next action | Effect on reproduction |
|---|---|---|---|
| Source code, existing tests, example configurations, reports 00–91, and a selected result summary | Yes | Clone the repository; begin with [current status](STATUS.md). | Sufficient to inspect the code, reports, and packaging checks. |
| Ordered 350-image Selection and 150-image Holdout membership identifiers | Yes | Stage `splits/` directories; tracked paths are relative to the RUOD-R root. | Identifiers do not include the actual images or annotations. |
| Raw images, annotations, and dataset downloads | No | Obtain the matching dataset/version from its original provider under the applicable terms, or supply the existing local research copy. No verified RUOD-R download is advertised by this edition. | Model execution and historical scoring require the correct data, labels, and frozen roles. An arbitrary replacement dataset does not reproduce the reported results. |
| External pretrained checkpoints and project-trained weights | No | Supply your own checkpoint for a new run. For an exact historical run, locate the required checkpoint in the original run records or the partial historical release, and verify its expected identity. | The example `best.pt` path is a placeholder. A generic pretrained checkpoint does not substitute for the reported trained model. |
| ONNX models and TensorRT engines | No | Locate the matching historical artifact if available, or regenerate with the required weights, export procedure, platform, and precision contract. | Regeneration alone does not establish byte identity, numerical equivalence, or identical device behavior. |
| Prediction arrays, complete run directories, large logs, calibration records, and frozen execution contracts | Not distributed as a complete set | Consult the relevant report and historical file map; some October 23 artifacts are available in the partial release below, while other records remain local. | Missing inputs can prevent historical integrity checks, rescoring, or exact reproduction. Later reports do not have a complete public artifact bundle here. |
| External runtimes and environment distributions | No | Install the platform-appropriate dependencies; see [reproduction requirements](REPRODUCIBILITY.md). | Dependency declarations are not a fully validated environment lock. |
| Credentials, private keys, device-access details, machine-specific configuration, and local caches | No | Create your own local configuration and access setup. Access credentials are not distributed. | These are not research artifacts to restore from this repository. |

The omitted local records are retained in the original research collection. This documentation does not promise that every omitted item is publicly downloadable or available on request. Any further sharing requires checking ownership, permissions, intended scope, and the exact files to publish.

## Historical release: only 7 of 16 planned ZIP assets are available

The earlier [snapshot-2026-10-23 release](https://github.com/X3no-07/ramp-q-mixed-precision/releases/tag/snapshot-2026-10-23) currently contains **7 assets**, while its original publication manifest lists **16**. The historical upload was incomplete. The presence of a release does not establish that all planned results have been published.

| Status | ZIP part numbers |
|---|---|
| Available in the release | 001, 002, 003, 004, 006, 008, 016 |
| Not present in the release | 005, 007, 009, 010, 011, 012, 013, 014, 015 |

[ARTIFACT_AVAILABILITY.json](../export/ARTIFACT_AVAILABILITY.json) lists every planned asset, its expected size and SHA256 from the historical manifest, and a download URL only when that asset is present. This update checked GitHub asset presence and sizes; it did not download and rehash the payloads.

These files belong to the October 23 snapshot. They are not complete evidence for reports 51–91 and are not a package matched to the current translated source tree. The [historical archive branch](https://github.com/X3no-07/ramp-q-mixed-precision/tree/archive/chinese-snapshot-2026-10-23) retains the original publication file map and restoration helper. Its original text and paths describe that earlier edition.

To inspect an available historical result:

1. Identify its exact run, source version, required files, and expected hashes in the report and historical file map.
2. Check which ZIP assets contain those files and whether each required asset is available. If an input is missing, that reproduction path remains incomplete.
3. Download available assets from the release and verify their hashes before using the original restoration mapping.
4. Preserve original contracts and failure states. Do not relax a threshold or replace an artifact merely to make an old integrity check pass.

No automatic download or restoration of the complete archive is claimed. Completing the remaining historical uploads or publishing later experiment bundles would be a separate publication update.

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
