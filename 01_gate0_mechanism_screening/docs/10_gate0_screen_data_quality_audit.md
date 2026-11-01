# Gate 0 Data Quality Check



> Data quality conclusion: **PASS**. This report checks whether the results table can be used for subsequent module isolation; risk stratification is exploratory description, not a frozen hypothesis test.

# # Dataset and Granularity

- Input: `../artifacts/predictions/screen.csv`
- Rows: 5570; number of targets: 2785; one target per row at the object level under a given configuration.
- FP32 matches: 2293; additional misses in full W8A8: 165 (7.20%).

# # Core Quality Checks

| Check | Result |
|---|---|
| Missing required columns | [] |
| Duplicate target keys within a configuration | {'fp32': 0, 'q_all': 0} |
| Consistency between FP32 and W8A8 target sets | True |
| Illegal values or rule contradictions | {} |

# # Concentration and Failure Modes

- Additional misses distributed across 102 images; maximum per image: 4.85%; top 10 images account for 27.88%.
- Additional misses cover 8 categories; maximum per category: 25.45%.
- Diagnostic failure classification: `{'confidence_threshold': 155, 'localization_or_assignment': 4, 'combined_or_no_candidate': 6}`

# # Exploratory Risk Signals

- Median area of FP32-matched targets: 0.019209; median area of additional misses: 0.007963.
- Relative risk at the lower quartile of area: 2.959.
- Median local contrast of FP32-matched targets: 0.054104; median contrast of additional misses: 0.033092.
- Relative risk at the lower quartile of contrast: 1.997.

# # Usage Boundaries

- Screen only verifies whether the sample size is sufficient to enter module scanning; it does not establish causal effects of any module.
- Quartile risk thresholds are derived from current data and are exploratory; the weight formula for formal Gate must be frozen before being applied to an independent holdout set.
- Simulated W8A8 is not TensorRT INT8; therefore, Jetson latency, power consumption, or actual INT8 accuracy cannot be reported here.
