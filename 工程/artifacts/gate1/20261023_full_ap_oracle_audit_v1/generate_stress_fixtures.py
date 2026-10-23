"""Generate the predeclared, synthetic-only AP stress inputs without oracle outputs."""
from __future__ import annotations
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
SEED = 2026102301


def gt(x, y, w, h, category=1, score=1):
    return [x, y, w, h, score, category, 0, 0]


def dt(x, y, w, h, score=.9, category=1):
    return [x, y, w, h, score, category, -1, -1]


def image(ground_truth=(), predictions=(), *, width=128, height=96):
    return {"width": width, "height": height, "raw_ground_truth": list(ground_truth), "raw_predictions": list(predictions)}


def case(identifier, purpose, *images):
    return {"id": identifier, "kind": "aggregate", "purpose": purpose, "images": list(images)}


def generate():
    rng = random.Random(SEED)
    # Deterministic generated row order is part of the input, not sorted upstream.
    cap_gt = [gt(4, 5, 8, 8, 1), gt(70, 60, 9, 9, 10)]
    cap_dt = [dt(100 + i % 7, 10 + i % 9, 2, 3, (i % 11) / 10, 1 + i % 10) for i in range(521)]
    cap_dt += [dt(4, 5, 8, 8, .99, 1), dt(70, 60, 9, 9, .98, 10)]
    cases = [
        case("S01_global_caps_523_unsorted", "523 submitted rows cross every maxDets cap; true matches occur at the tail.", image(cap_gt, cap_dt)),
        case("S02_global_caps_523_reverse", "The same 523 rows reversed before global caps; preserve input order.", image(cap_gt, list(reversed(cap_dt)))),
        case("S03_ties_and_equal_overlap", "Equal confidence and equal overlaps span two positive classes and repeated boxes.",
             image([gt(4, 5, 10, 10, 1), gt(6, 5, 10, 10, 1), gt(45, 5, 10, 10, 2)],
                   [dt(5, 5, 10, 10, .5, 1), dt(5, 5, 10, 10, .5, 1), dt(45, 5, 10, 10, .5, 2), dt(80, 5, 10, 10, .5, 2)])),
        case("S04_score_zero_absorption", "Positive priority, repeated same-class ignore absorption and ignored-GT denominator.",
             image([gt(5, 5, 8, 8, 1), gt(1, 1, 90, 80, 1, 0), gt(70, 60, 8, 8, 2, 0)],
                   [dt(5, 5, 8, 8, .9, 1), dt(20, 20, 3, 3, .8, 1), dt(25, 25, 3, 3, .8, 1), dt(70, 60, 8, 8, .7, 2), dt(20, 20, 3, 3, .6, 2)])),
        case("S05_unequal_image_frequencies", "Class occurrences across four images have deliberately unequal frequencies.",
             image([gt(5, 5, 8, 8, 1)], [dt(5, 5, 8, 8, .9, 1)]),
             image([gt(5, 5, 8, 8, 1)], []),
             image([gt(5, 5, 8, 8, 1), gt(45, 45, 8, 8, 2)], [dt(45, 45, 8, 8, .8, 2)]),
             image([gt(70, 5, 8, 8, 3)], [dt(70, 5, 8, 8, .7, 3)])),
        case("S06_empty_two_images", "No invented scores for two completely empty images.", image(), image()),
        case("S07_only_class_eleven", "Class 11 survives preprocessing but is outside aggregate categories 1 through 10.",
             image([gt(5, 5, 8, 8, 11)], [dt(5, 5, 8, 8, .9, 11)])),
        case("S08_regions_rounding_and_border", "Overlapping category-zero raster union, half-away rounding and unclipped area denominator.",
             image([gt(1, 1, 4, 9, 0, 0), gt(3, 1, 4, 9, 0, 0), gt(20, 20, 8, 8, 1)],
                   [dt(1, 1, 8, 8, .9, 1), dt(-.5, .5, 2.5, 2.5, .8, 1), dt(29, 29, 10, 10, .7, 2), dt(20, 20, 8, 8, .6, 1)], width=30, height=30)),
        case("S09_raw_outside_iou_edges", "Raw out-of-image coordinates and fractional overlaps cross the ten matching thresholds.",
             image([gt(-5, 5, 10, 10, 1), gt(45, 45, 10, 10, 2)],
                   [dt(0, 5, 5, 10, .9, 1), dt(45.5, 45, 10, 10, .8, 2), dt(47, 45, 10, 10, .7, 2)])),
    ]
    dense_gt = [gt(4 + 10 * i, 4 + 7 * (i % 6), 6, 6, i + 1) for i in range(10)]
    dense_dt = [dt(rng.randrange(-4, 125) + .25 * rng.randrange(4), rng.randrange(-4, 94),
                   rng.randrange(2, 15), rng.randrange(2, 15), rng.randrange(7) / 6, 1 + rng.randrange(10)) for _ in range(610)]
    dense_dt += [dt(*row[:4], .99, int(row[5])) for row in dense_gt]
    rng.shuffle(dense_dt)
    cases.append(case("S10_seeded_dense_ten_classes", "620 seeded unsorted detections span all ten categories, score ties and global caps.", image(dense_gt, dense_dt)))
    cases += [
        case("S11_cross_image_score_ties", "Stable cross-image confidence ties and an empty-middle image preserve concatenation order.",
             image([gt(5, 5, 8, 8, 1)], [dt(80, 5, 8, 8, .5, 1), dt(5, 5, 8, 8, .5, 1)]),
             image(),
             image([gt(45, 45, 8, 8, 1), gt(80, 60, 8, 8, 2)], [dt(45, 45, 8, 8, .5, 1), dt(80, 60, 8, 8, .5, 2)])),
        case("S12_fractional_recall_and_duplicates", "One of three and two of seven GT, duplicated detections and non-grid recall integration.",
             image([gt(3 + 15*i, 5, 8, 8, 1) for i in range(3)] + [gt(3 + 15*i, 40, 8, 8, 2) for i in range(7)],
                   [dt(3, 5, 8, 8, .9, 1), dt(3, 5, 8, 8, .8, 1), dt(3, 40, 8, 8, .7, 2), dt(18, 40, 8, 8, .6, 2)])),
    ]
    assert len(cases) == 12
    return {"schema": "rampq.gate1.synthetic_official_oracle_stress_fixtures/v1", "data_role": "synthetic",
            "official_commit": "005445782213e20cb91bc50a597db3dd949e749a", "no_real_data_or_model_access": True,
            "expected_oracle_outputs": None, "expectation_policy": "Native source supplies outputs; none are handwritten",
            "generator_contract": {"version": 1, "seed": SEED, "algorithm": "Python stdlib random.Random MT19937",
                                   "generated_fixture_bytes_are_authoritative": True}, "cases": cases}


def main():
    path = HERE / "stress_fixtures.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(generate(), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(path)


if __name__ == "__main__":
    main()
