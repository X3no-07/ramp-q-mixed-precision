# Full AP/AR mirror: pre-comparison implementation contract

This is a Python translation of the pinned official source, not a new metric.
The implementation was designed and tested without reading native fixture AP,
AR, GT-match, or detection-match answers. The only native values inspected were
ten float64 threshold constants from a pure arithmetic Octave 11.3.0 probe.

## Public API

- `comp_oas(dt Nx4, gt Nx4, ignore length-N)` returns overlap matrix MxN.
- `eval_res(gt0 Nx5, dt0 Nx5, threshold=.7, allow_multiple=False)` returns
  GT Nx5 and detection Nx6, including all -1/0/1 match flags.
- `calc_accuracy(allgt, alldet, capture_intermediates=True)` accepts equal-length
  per-image lists/tuples of prepared eight-column matrices and returns AP 10x10,
  AR 10x10x4, `evalClass` with repetitions, seven aggregate metrics in percent,
  and optional auditable intermediate arrays.
- `prepare_raw_image(raw_gt8, raw_dt8, width=..., height=...)` calls only the
  pinned raster prefilter, preserves surviving raw matrix rows/order/attributes,
  then inverts publisher GT score 0/1 into ignore 1/0. It returns `filtered_gt`,
  `filtered_predictions`, and `prepared_gt`.
- `evaluate_raw_images(images, capture_intermediates=True)` is a convenience
  composition of the raw adapter and prepared aggregation API.

Prepared GT fifth column is already the inverted ignore flag. This is distinct
from raw publisher GT fifth-column score. Categories 0/11 are allowed inputs
but do not enter evaluated classes 1..10; their prediction rows can consume the
all-class input cap. The prepared API does not silently apply category-0 raster
filtering; callers wanting it must use the raw adapter first.

## Preserved source behavior

1. Each image first takes the submitted first 1/10/100/500 detection rows across
   all classes. Only afterwards does it extract the current class and perform
   stable descending score matching. Neither NMS nor pre-cap score sorting is
   introduced.
2. `evalRes` sorts GT ignore last, gives nonignored GT priority, and on equal
   overlap prefers the later GT row. Ignored GT can absorb multiple detections.
   The optional `mul` behavior is translated literally, including its -1 output
   after matching an already matched positive GT.
3. Cross-image detection records are concatenated in image order, then stable
   descending score sorted. Equal-score ties therefore retain image order and
   single-image row order. This is an explicit runtime policy, not a proof for
   all MATLAB/Octave versions.
4. Every GT output row counts in the recall denominator, including ignored GT.
   Ignored detection flag -1 counts in neither cumulative TP nor cumulative FP.
5. `evalClass` includes a class once per image containing that class. Aggregation
   therefore retains source image-frequency weighting instead of unique-class
   macro averaging.
6. Ten IoUs use the observed chosen-runtime float64 constants for `.5:.05:.95`.
   AP uses only the final cap=500 match results. It applies a backwards precision
   envelope and continuous VOC recall-step integral, not 11/101-point sampling.
7. Empty `evalClass` yields NaN for all seven metrics. AP/AR class tables retain
   their initialized zero values. Empty arrays retain explicit row/column axes.

## Finite input domain

Matrix entries must be finite real numeric values, with finite XYWH endpoints
and areas. Boolean, complex, object/string, malformed shapes, noninteger or
out-of-range categories, nonbinary prepared GT ignore flags, NaN/Inf, and box
arithmetic overflow are rejected. Empty input is `[]` or a correctly shaped
zero-row matrix. Scores are finite and can lie outside [0,1]; they are never
rescaled or clipped. Prepared geometry remains raw/unclipped, including finite
zero/negative extents with zero positive overlap. Threshold must be finite in
[0,1]. Invalid input is not repaired. All caller arrays are copied.

The raw adapter additionally inherits the old prefilter's declared image pixel
limit, positive prediction sizes, positive score-1 evaluated GT sizes, and
integer category-0 region coordinates. It verifies the exact old source SHA
before importing it. It does not invoke the old single-image matcher.

## Audit fields and limits

Each class/IoU/cap intermediate contains selected input indices per image,
GT/detection output matrices, concatenated GT flags and detection score/flags,
global stable rank, cumulative TP/FP, recall, precision, and recall denominator.
These permit direct recomputation independently of summary metrics. No native
comparison has occurred at this preparation stage. Oracle comparisons must pin
the frozen implementation hash, use a separate output directory, and report
both numeric error and NaN-pattern agreement.

Synthetic agreement cannot prove equivalence outside the admitted domain,
official dataset accuracy, deployment correctness, Gate 1 acceptance, or any
performance claim. No real dataset, Selection, Holdout, weight, image, or GPU
is read or run by this preparation workflow. Old guards and sealed evidence are
unchanged.
