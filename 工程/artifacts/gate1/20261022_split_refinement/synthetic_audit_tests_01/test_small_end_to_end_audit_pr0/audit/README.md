# Gate1 train/val near-duplicate screening

This output is a visual-review queue. Both radius-four hash screens are exhaustive within the listed 7019 train/val images, but they cannot prove absence of near duplicates/shared scenes. No detector, annotations, holdout, split modification, or automatic quarantine was used.

candidate_pairs.jsonl stores only cross-old-role/prefix/component candidates. Hashes, standardized low-resolution grayscale correlation and color/intensity differences are descriptive. prefix_component_screening_edges.jsonl contains candidate links, not confirmed scene identities. Inspect contact sheets and retain explicit visual decisions before using any links to refine splits. Original SHA checks were repeated after the audit. Method details and limitations are in method_contract.json.
