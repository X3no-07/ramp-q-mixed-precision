# Restricted ignore Dataset component (synthetic only)

Implemented `RestrictedIgnoreDataset(manifest_path, manifest_sha256, split_role=..., image_size=..., class_count=10, augment=False, fliplr=0, flipud=0, seed=0, epoch=0)` and `build_synthetic_fixture(new_directory)`.

Only the caller hash-bound manifest's explicit members may load. Each entry's role, image ID, dimensions and image/positive-label/ignore-sidecar bytes bind independently. A train/ prefix is never a membership rule. Selection/Holdout/test paths and roles are rejected before member reads. This version requires `data_role=synthetic_only`; real training needs a separate reviewed contract.

Fresh verified PNG bytes decode with cv2; positives read separate normalized YOLO xywh; sidecar JSONL passes the old frozen fail-closed geometry helper. The same actual longest-side ceil resize, centered fixed LetterBox and explicit vertical/horizontal flip decisions transform images, positives and ignores. Nothing invokes the ordinary probability-zero complex augmentation chain. Explicit zero-row sidecars are allowed; zero/negative/fully-clipped source boxes reject the whole sample, never drop a row. No image or labels are silently skipped.

Batch outputs are RGB uint8 BCHW, float32 cls [N,1], normalized xywh [N,4], batch_idx [N], plus one float32 pixel XYXY tensor per image under ignore_boxes_xyxy, image_identities, sample_identities, source_records, image_hw and transform_trace. Only workers=0 CPU DataLoader is tested here. Ragged ignores require the Trainer/criterion to move all tensors to the model's device; ordinary DetectionTrainer preprocess only moves top-level tensors.

45/45 synthetic file-I/O and data-contract tests pass. Initial pytest attempt failed before Dataset code because the sandbox's default Temp path could not create a numbered directory; the unchanged source passes with a new owned basetemp. Both logs remain bound. Frozen fixture is six newly generated synthetic PNGs (four train, two training_val), all image hashes distinct and image IDs disjoint; three distinct publisher-ignore reasons are retained. The first score-0 region spans 90% of each original dimension and overlaps positives so a separate root Trainer smoke can verify positive priority and ignored negatives are used.

This package proves the Dataset component on synthetic files. It does not prove actual Detector/Trainer integration, real training, official AP, clean evaluation, scientific Gate1 acceptance, quantization, or deployment. No real images, old checkpoints, GPU, Selection or Holdout were read.
