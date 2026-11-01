# Gate1: Holdout Directory Audit and Pre-inspection Preparation



**The outer layer identity of the sealed package and the count of the ZIP central directory have been completed and verified. The directory contains 1610 image files and 1610 original annotation files, fully matched by filename. The content of image and label members has not yet been read.**

The user responded 'OK, proceed' after a clear instruction to 'only check archive integrity and directory structure, without viewing images or labels, without running evaluations'. This response is hereby recorded as a human decision, freezing the directory audit authorization contract; no expansion of directory access to member content reading or model evaluation authorization is being made.

# # Actual Completion and Identity

A single directory inventory was completed at Beijing Time 2026-10-25 23:42:17, and the launcher returned 0. A complete termination acknowledgment was received; currently, no directory inventory or content pre-check processes are underway.

| Item | Result |
|---|---|
| Sealed Package Byte Count | 311251787 |
| Outer SHA256 | `b28a94b06dfd9e36ce77ff8155fb82b9d2c030f198a76105933bd56c6ea6a68d` |
| Matches Original SEAL Record | Yes |
| Central directory entry | 3224: 1610 images, 1610 annotations, 3 directories, 1 other file |
| Image and annotation filename pairing | 1610 pairs; no missing, duplicate logical IDs, or case-sensitive path conflicts |
| Declaration of uncompressed byte totals | 312781612 |
| Reading and decompressing ZIP members | 0 / 0 |
| Image decoding, annotation parsing, model loading, NMS, AP computation | all are 0 |

Outer fingerprint proves local container matches the sealed record from 2026-10-22, with no publisher digest available to assert upstream authenticity. CRC comes from the central directory declaration and has not been verified by member reading; CRC is not a member SHA256. Member content integrity, image dimensions, annotation quality, and real-world scene isolation remain unknown.

# # Name-level isolation and artifact preservation audit

| Existing roles | Base name ID intersection | Prefix proxy intersection |
|---|---:|---:|
| calibration_pool | 0 | 0 |
| debug | 0 | 0 |
| quarantine | 0 | 0 |
| selection | 0 | 0 |
| train | 0 | 0 |
| training_val | 0 | 0 |

The table above only compares name-level metadata. Zero intersection does not prove absence of image duplication, near-duplication, or semantic scene leakage, nor does it establish scientific sample sufficiency. These name-based selections were not used to pick candidates or to modify G1, calibration seeds, or thresholds.

A standalone audit re-computes the sorting, roles, extensions, pairing, size totals, and canonical JSON order hash of the 3224 records. It also recalculates the intersection of all role names and prefix proxies. The five D-drive outputs show consistent hashes before and after the audit. This audit does not reopen the outer compressed archive nor read any members.

Directory output located at `[OMITTED_LOCAL_PATH]`, approximately 1.55MB. Original C-drive training results, 546 development outputs, original 548 annotation conflicts, numerical thresholds, and existing reports are all preserved.

# # Next-step limited content pre-check prepared

It is still recommended to refrain from running the model at this stage. For all fixed 1610-member entries in the ZIP memory stream, perform checks on: image byte SHA and decoded dimensions/pixel identity, non-trivial EXIF orientation, original annotation eight-field structure, category/score/ignore flags, and limited geometric information. Use image byte SHA to verify exact overlap with existing training, calibration, and development metadata, and check whether byte- or pixel-identical images in this set have conflicting original physical-row annotations.

Original 1610 members and their order are retained; no image selection is performed based on content outcomes. Any identity leakage, illegal structure, or content-conflicting annotations are immediately halted and preserved in situ, without automatic deletion, deduplication, annotation modification, selection of a single annotation set, or retry. Byte- or pixel-identical images only form dependency groups and do not increase independent sample count. Cross-set pixel-level near-duplication and real-world scene isolation are not claimed to be resolved within this limited pre-check.

The pre-check for this content has not been authorized or executed. Entry and 19 synthetic test cases have been prepared; the maximum duration is 1800 seconds, with each image member of 32 MB, each annotation member of 8 MB, decoding per image of 5000 ten-of-thousands of pixels, total input read of 400 MB, output of 100 MB, and at least 2 GB reserved on Drive D. The original data is not decompressed, not copied, not displayed, not used for training, calibration, NMS, target matching, or AP. 19 JUnit test cases include test methods and sub-tests, without exaggerating the count to 19 independent experiments.

Only 1610 member fingerprint, size, and structural inspection receipts will be generated in the future. It will not automatically initiate fixed G1 evaluation; the model and scoring range must still be frozen after real member identity formation. Content pre-inspection cannot substitute independent scientific validation, model superiority, or empirical evidence of real INT8/Orin performance.

# # Current Access Boundary

Holdout is currently stated as 'Outer identity and central directory metadata have been authorized and counted; member image and annotation content remains sealed'. Authorization for directory counting does not extend to subsequent content reading. A new specific decision is required: whether to allow the fixed 1610 to conduct pre-inspection of member content quality, without running any model or scoring. This issue stems from a previously explicitly defined metadata authorization and cannot be resolved by script-generated boolean fields opening content access.

All directory counting, preservation verification, and content pre-inspection preparations that can be autonomously completed in this round have been finalized, with scheduled checks maintained for deletion. The original complete 8400 bounding box numerical failures, historical `FAIL_FIXED_TOLERANCES` and `FAILED_UNCHANGED`, are preserved as-is. Neither independent scientific validation nor board-end acceptance has been passed as a result.

# # Files

- [Directory Counting Authorization Contract](../../docs/DATA_AND_ARTIFACTS.md)
- [D-drive Completion Receipt](../../docs/DATA_AND_ARTIFACTS.md)
- [Directory Preservation Audit](../../docs/DATA_AND_ARTIFACTS.md)
- [Pending Authorization Content Pre-inspection Contract](../../docs/DATA_AND_ARTIFACTS.md)
- [New Content Pre-inspection Entry](../scripts/gate1_holdout_content_preflight.py)
- [Full Delivery Verification](../../docs/DATA_AND_ARTIFACTS.md)
