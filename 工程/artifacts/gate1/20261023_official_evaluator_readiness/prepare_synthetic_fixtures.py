"""Generate synthetic arrays only; no model, image, annotation, or archive access."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def gt(x=5, y=5, w=10, h=10, *, category=1, score=1):
    return [x, y, w, h, score, category, 0, 0]


def dt(x=5, y=5, w=10, h=10, *, category=1, score=0.9):
    return [x, y, w, h, score, category, -1, -1]


def image(targets=(), predictions=(), width=100, height=100):
    return {"width": width, "height": height,
            "raw_ground_truth": list(targets), "raw_predictions": list(predictions)}


def case(identity, kind, images, purpose, **kwargs):
    return {"id": identity, "kind": kind, "purpose": purpose,
            "images": images, **kwargs}


def main():
    rows = [
        case("A01_perfect_single_class", "aggregate", [image([gt()], [dt()])],
             "Basic full calcAccuracy invocation, ten IoU thresholds and four maxDets."),
        case("A02_gt_without_predictions", "aggregate", [image([gt()], [])],
             "Empty detection matrices retain 0x8 shape; valid GT remains."),
        case("A03_empty_both", "aggregate", [image()],
             "Observe native empty evalClass/mean2 behavior; never impose an invented zero score."),
        case("A04_predictions_without_gt", "aggregate", [image([], [dt()])],
             "Observe unavailable-class handling and empty ground-truth behavior."),
        case("A05_same_class_ignore_only", "aggregate", [image([gt(score=0)], [dt()])],
             "Observe availability and denominator for surviving score=0 GT."),
        case("A06_positive_plus_ignored_denominator", "aggregate",
             [image([gt(), gt(50, 50, score=0)], [dt()])],
             "Record literal gtMatch length denominator including ignored GT."),
        case("A07_unequal_class_image_frequency", "aggregate",
             [image([gt()], [dt()]), image([gt()], []),
              image([gt(category=2)], [dt(category=2)])],
             "Expose evalClass repetition: class 1 occurs in two images, class 2 in one."),
        case("A08_input_order_before_cap", "aggregate",
             [image([gt()], [dt(50, 50, category=2, score=0.1), dt(score=0.99)])],
             "Native maxDets=1 is applied before class extraction and score sorting."),
        case("A09_swapped_submission_order", "aggregate",
             [image([gt()], [dt(score=0.99), dt(50, 50, category=2, score=0.1)])],
             "Order-paired case for A08; no upstream global reordering is introduced."),
        case("A10_precision_envelope", "aggregate",
             [image([gt(), gt(50, 50)],
                    [dt(80, 80, score=0.99), dt(score=0.9), dt(50, 50, score=0.8)])],
             "Native VOC precision envelope versus point-sampled AP; oracle supplies all outputs."),
        case("A11_three_gt_one_hit", "aggregate",
             [image([gt(), gt(30, 30), gt(60, 60)], [dt()])],
             "Non-grid recall endpoint distinguishes VOC integration from 101-point sampling."),
        case("R01_exact_half_region", "region",
             [image([gt(1, 1, 4, 9, category=0, score=0), gt(1, 1, 8, 8)],
                    [dt(1, 1, 8, 8)], 20, 20)],
             "Category-0 raster boundary at exactly half; inspect native retained matrices."),
        case("R02_overlapping_region_union", "region",
             [image([gt(1, 1, 3, 9, category=0, score=0), gt(1, 1, 3, 9, category=0, score=0)],
                    [dt(1, 1, 8, 8)], 20, 20)],
             "Overlapping region masks form a pixel union without double counting."),
        case("R03_border_and_rounding", "region",
             [image([gt(1, 1, 100, 100, category=0, score=0)],
                    [dt(9, 9, 10, 10), dt(-0.5, 0.5, 2.5, 2.5)], 10, 10)],
             "Native round half-away behavior and unclipped rounded area denominator."),
        case("R04_zero_height_ignored_region", "region",
             [image([gt(1, 1, 2, 0, category=0, score=0)], [dt(1, 1, 1, 1)], 10, 10)],
             "Source max(1,xywh) promotes original ignored zero height."),
        case("M01_score_first_over_iou_first", "match",
             [image([gt()], [dt(6, 5, score=0.95), dt(score=0.8)])],
             "Native evalRes processes high confidence before exact overlap.", threshold=0.5),
        case("M02_tied_score_and_overlap", "match",
             [image([gt(4, 5), gt(6, 5)], [dt(score=0.9), dt(score=0.9)])],
             "Native runtime sort ties and later equal-overlap GT assignment.", threshold=0.5),
        case("M03_ignore_absorbs_multiple_same_class", "match",
             [image([gt(1, 1, 80, 80, score=0)],
                    [dt(2, 2, 5, 5), dt(3, 3, 5, 5, score=0.8), dt(2, 2, 5, 5, category=2)])],
             "Repeated same-class ignore IoF absorption does not absorb other class.", threshold=0.5),
        case("M04_raw_outside_coordinates", "match",
             [image([gt(-5, 5, 10, 10)], [dt(0, 5, 5, 10)])],
             "Native matcher keeps raw coordinates; no project clipping.", threshold=0.75),
        case("M05_others_and_positive_priority", "match",
             [image([gt(), gt(1, 1, 80, 80, score=0), gt(category=11)], [dt(), dt(category=11)])],
             "Valid match precedes ignore absorption; category 11 outside classes 1..10.", threshold=0.5),
    ]
    fixture = {"schema": "rampq.gate1.synthetic_official_oracle_fixtures/v1",
               "data_role": "synthetic", "official_commit": "005445782213e20cb91bc50a597db3dd949e749a",
               "no_real_data_or_model_access": True,
               "expected_oracle_outputs": None,
               "expectation_policy": "Record outputs from unmodified pinned MATLAB sources; no handwritten expected metric values.",
               "cases": rows}
    target = HERE / "synthetic_fixtures.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(fixture, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "PREPARED_SYNTHETIC_INPUTS_ONLY", "case_count": len(rows),
                      "fixture_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
