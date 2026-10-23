# VisDrone Gate1 preparation

Status: PREPARED_NOT_FROZEN_NOT_TRAINED_NOT_SCORED.

Only official train/val supplied archives were opened. Images and original annotations remain byte-exact. The official val is unscored Selection. training.yaml validates only on reserved training_val.

splits are a deterministic proposal using filename-prefix/exact-content connected components. Prefixes are proxies, not proven scene identities. Near-duplicate/scene review remains required. Cross train/val connected train groups are quarantined. Exact image duplicates are recorded and excluded from training reservations. Official val records are retained; review any duplicate annotation conflicts.

YOLO labels include only score=1, original classes 1..10 mapped to 0..9, and nonempty clipped boxes. All annotation rows retain original physical line IDs in targets_*.jsonl; ignored rows additionally appear in ignored_regions_*.jsonl. The sidecars do not implement ignore-aware evaluation. publisher_ignored distinguishes the publisher's ignored annotation from preparation exclusions. A non-positive score=1/class1..10 train box is retained verbatim with excluded_for_preparation_quality=true and invalid_positive_box_geometry; its entire connected train prefix/content component is quarantined. Such a val box causes strict rejection. Plain YOLO training still treats ignored areas as background for negative loss; this limitation must be stated or explicitly addressed before claiming an official ignore-aware baseline.
