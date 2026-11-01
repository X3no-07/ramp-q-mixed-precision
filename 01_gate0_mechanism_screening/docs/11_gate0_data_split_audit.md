# Gate 0 Data Split Check



> Conclusion: **PASS**. In this run, only the list of images for selection/holdout was frozen; no module isolation experiments were conducted.

# # Freezing Rules

- Random seed: `20260910`
- Image ratio: selection 70%, holdout 30%
- Split unit: entire images; all real targets within the same image belong to only one set.
- Algorithm: Sort images by filename first, then apply Python `random.Random(seed).shuffle` to select the first 70% for selection.
- The selection set is used for module selection; the holdout set must not be used for ranking, hyperparameter tuning, or module swapping prior to module and rule locking.

# # Overall Results

| Metric | selection | holdout | Total |
|---|---:|---:|---:|
| Images | 350 | 150 | 500 |
| True Object Count | 1920 | 865 | 2785 |
| FP32 Hit Count | 1571 | 722 | 2293 |
| Additional Misses in W8A8 | 119 | 46 | 165 |
| Number of Images with Additional Misses | 76 | 26 | 102 (no image overlap between sets) |

# # Distribution of Additional Misses by Category

| Category ID | selection | holdout | Total |
|---:|---:|---:|---:|
| 0 | 30 | 12 | 42 |
| 1 | 21 | 4 | 25 |
| 2 | 17 | 7 | 24 |
| 3 | 16 | 9 | 25 |
| 4 | 18 | 8 | 26 |
| 5 | 8 | 5 | 13 |
| 6 | 7 | 1 | 8 |
| 7 | 2 | 0 | 2 |

# # Integrity Check

- Image overlap between the two sets: 0 (required to be 0)
- Coverage of all screen images: True
- All paths in the list exist: True
- Selection has at least 100 additional misses: 119, passed
- Holdout has at least 40 additional misses: 46, passed
- Major categories (total additional misses ≥ 5) present on both sides: passed

# # Files and Reproducibility Fingerprint

- Selection list: `../splits/gate0_selection_images.txt`
- selection SHA-256:`a78348cb515553aece6e40dc02e21fc6be52df2c26989337f9d9a4638677e759`
- Holdout list: `../splits/gate0_holdout_images.txt`
- holdout SHA-256:`4061739eb2e3b1812f2daf0b4327a65cf0cb7922eb43b30fc9457db310f976c7`
- Machine-readable list: `../artifacts/manifests/gate0_split.json`
- Input prediction table SHA-256: `93f7a651201f2fdca747a2cfe1852f32bad5dd7cc2f6683c989ebdd258796388`

# # Usage Boundaries

- Category ID 7 has only 2 additional misses in total, which may appear on only one side after random split; it does not meet the primary category threshold and cannot be used for category-level conclusions.
- The current split is only valid for these 500 screen images and their existing FP32/W8A8 results; if the input list, random seed, or prediction file changes, a new one must be regenerated and a new SHA-256 obtained.
- The next step can only run `q_only:G1` on the selection; the holdout remains sealed.
