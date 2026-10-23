# Full Python AP/AR comparison against native official source

Result: `PASS_DECLARED_SYNTHETIC_FULL_AP_NATIVE_COMPARISON_ONLY`.

The frozen Python implementation was compared with saved outputs from unchanged official VisDrone functions at commit `005445782213e20cb91bc50a597db3dd949e749a`, using GNU Octave 11.3.0 and its real image-package `mean2`.

| Suite | Cases | Exact float64 matrices | Final AP/AR scalar comparisons | Matching NaN scalars | Maximum finite absolute difference (percentage units) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen synthetic suite | 20 | 166 | 77 | 14 | 0 |
| Predeclared stress suite | 12 | 54 | 84 | 14 | 7.105427357601002e-15 |
| Total | 32 | 220 | 161 | 28 | 7.105427357601002e-15 |

Every preprocessing/matching matrix has its original shape and exactly equal float64 elements, with zero numerical tolerance. Every finite final AP/AR scalar satisfies the predeclared `1e-10 + 1e-12 * abs(native)` allowance, in percentage units. NaN patterns match exactly; infinities are prohibited. The allowance permits binary64 reduction-order rounding only and does not change the failed original detector dtype guard.

Stress inputs cover 523-row global-cap order pairs, confidence/overlap ties, ignored GT, unequal class/image frequencies, empty and category-11-only inputs, category-zero raster boundaries, outside coordinates, 620 seeded predictions across ten classes, cross-image ties, and fractional recall endpoints. Seed, complete input bytes, wrapper and comparison policy were frozen before the new native run. The Python implementation and comparator were frozen before comparison. No handwritten expected AP answers were supplied to the implementation author.

The audit independently checked the eleven official source identities, 2,890 runtime members, 1,656 isolated MAT-reader members, launcher and output identities, all six resolved official function paths, real native `mean2`, MAT/JSON parity, empty dimensions and NaNs. Auditor negative tests passed 45/45; implementation synthetic tests passed 54/54.

This establishes agreement for these 32 declared synthetic cases, including 23 aggregate cases, on the pinned runtime and source. The native wrapper does not export internal AP/AR tables, so their independent native verification is not claimed. General equivalence for arbitrary inputs, real-dataset AP, detector inference, fresh training, board deployment and Gate1 scientific acceptance remain unproved. Selection and Holdout were not accessed; the original failed detector guards remain unchanged.

Use `audit_summary.json` for a compact machine-readable result and `comparison_attempt_v1.json` for each matrix identity and every metric value, NaN flag and absolute difference. `audit_delivery_manifest.json` seals this new audit package; its frozen contract transitively binds the prior native outputs and source/runtime dependencies.
