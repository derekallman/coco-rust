"""
fixture_builder.py — helpers for building adversarial parity fixtures.

Fixture format:
    {
      "info": {...},
      "licenses": [],
      "images": [...],
      "categories": [...],
      "annotations": [...],   // ground truth
      "detections": [...]     // predictions
    }

Usage:
    from fixture_builder import *
    path = write_fixture("my_scenario", scenario_all_crowd())
    passed, report = run_harness(path)
"""

import json
import subprocess
import sys
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "adversarial"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

HARNESS = Path(__file__).parent / "adversarial_harness.py"


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def make_image(id, width=640, height=480):
    return {"id": id, "width": width, "height": height, "file_name": f"img_{id}.jpg"}


def make_category(id, name=None):
    return {"id": id, "name": name or f"cat_{id}", "supercategory": "thing"}


def make_gt(id, image_id, category_id, bbox, area=None, iscrowd=0, segmentation=None):
    """bbox = [x, y, w, h]"""
    if area is None:
        area = float(bbox[2] * bbox[3])
    return {
        "id": id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [float(v) for v in bbox],
        "area": float(area),
        "iscrowd": iscrowd,
        "segmentation": segmentation if segmentation is not None else [],
    }


def make_dt(image_id, category_id, bbox, score):
    return {"image_id": image_id, "category_id": category_id, "bbox": [float(v) for v in bbox], "score": float(score)}


def make_fixture(images, categories, annotations, detections):
    return {
        "info": {"description": "adversarial parity fixture"},
        "licenses": [],
        "images": images,
        "categories": categories,
        "annotations": annotations,
        "detections": detections,
    }


# ---------------------------------------------------------------------------
# Pre-built scenarios
# ---------------------------------------------------------------------------


def scenario_all_crowd():
    """All GTs are crowd — every DT should be ignored."""
    img = make_image(1)
    cat = make_category(1)
    gts = [make_gt(i, 1, 1, [float((i - 1) * 10), 10.0, 80.0, 60.0], iscrowd=1) for i in range(1, 4)]
    dts = [make_dt(1, 1, [float((i - 1) * 10 + 2), 12.0, 76.0, 56.0], score=1.0 - i * 0.1) for i in range(1, 4)]
    return make_fixture([img], [cat], gts, dts)


def scenario_max_det_boundary():
    """Exactly 100 DTs (the max_dets limit) plus one more that should be cut."""
    img = make_image(1, width=2000, height=2000)
    cat = make_category(1)
    gt = make_gt(1, 1, 1, [0.0, 0.0, 100.0, 100.0])
    # 101 DTs with descending scores; only top 100 should be evaluated
    dts = [make_dt(1, 1, [float(i), 0.0, 80.0, 80.0], score=round(0.99 - i * 0.001, 4)) for i in range(101)]
    return make_fixture([img], [cat], [gt], dts)


def scenario_tied_scores():
    """Multiple DTs with identical scores competing for matching."""
    img = make_image(1)
    cat = make_category(1)
    # 10 non-overlapping GTs
    gts = [make_gt(i, 1, 1, [float((i - 1) * 60), 0.0, 50.0, 50.0]) for i in range(1, 11)]
    # 10 DTs all with the same score, each near the corresponding GT
    dts = [make_dt(1, 1, [float((i - 1) * 60 + 2), 2.0, 46.0, 46.0], score=0.5) for i in range(1, 11)]
    return make_fixture([img], [cat], gts, dts)


def scenario_area_boundary():
    """GT area exactly on the boundary between small/medium/large."""
    img = make_image(1, width=1000, height=1000)
    cat = make_category(1)
    # small/medium boundary = 32^2=1024; medium/large boundary = 96^2=9216
    # Place GTs right at the boundaries
    gts = [
        make_gt(1, 1, 1, [0.0, 0.0, 32.0, 32.0], area=1024.0),  # exactly small/medium boundary
        make_gt(2, 1, 1, [100.0, 0.0, 96.0, 96.0], area=9216.0),  # exactly medium/large boundary
    ]
    dts = [make_dt(1, 1, [1.0, 1.0, 30.0, 30.0], score=0.9), make_dt(1, 1, [101.0, 1.0, 94.0, 94.0], score=0.8)]
    return make_fixture([img], [cat], gts, dts)


def scenario_zero_area_bbox():
    """GT bbox with zero area (degenerate — width or height is 0)."""
    img = make_image(1)
    cat = make_category(1)
    gts = [
        make_gt(1, 1, 1, [10.0, 10.0, 0.0, 50.0], area=0.0),  # zero width
        make_gt(2, 1, 1, [50.0, 10.0, 50.0, 0.0], area=0.0),  # zero height
    ]
    dts = [make_dt(1, 1, [10.0, 10.0, 10.0, 50.0], score=0.9), make_dt(1, 1, [50.0, 10.0, 50.0, 10.0], score=0.8)]
    return make_fixture([img], [cat], gts, dts)


def scenario_no_dt_for_image():
    """One image has GT but no detections at all."""
    images = [make_image(1), make_image(2)]
    cat = make_category(1)
    gts = [make_gt(1, 1, 1, [0.0, 0.0, 100.0, 100.0]), make_gt(2, 2, 1, [0.0, 0.0, 100.0, 100.0])]
    dts = [
        make_dt(1, 1, [0.0, 0.0, 80.0, 80.0], score=0.9)
        # no DT for image 2
    ]
    return make_fixture(images, [cat], gts, dts)


def scenario_no_gt_for_category():
    """DTs reference a category that has no GTs."""
    img = make_image(1)
    cats = [make_category(1), make_category(2)]
    gts = [make_gt(1, 1, 1, [0.0, 0.0, 100.0, 100.0])]
    dts = [
        make_dt(1, 1, [0.0, 0.0, 80.0, 80.0], score=0.9),
        make_dt(1, 2, [50.0, 50.0, 80.0, 80.0], score=0.8),  # cat 2 has no GT
    ]
    return make_fixture([img], cats, gts, dts)


def scenario_overlapping_gts():
    """Two GTs heavily overlapping — only one DT should match."""
    img = make_image(1)
    cat = make_category(1)
    gts = [
        make_gt(1, 1, 1, [0.0, 0.0, 100.0, 100.0]),
        make_gt(2, 1, 1, [5.0, 5.0, 100.0, 100.0]),  # heavily overlaps with GT 1
    ]
    dts = [
        make_dt(1, 1, [2.0, 2.0, 96.0, 96.0], score=0.9)  # one DT, two candidate GTs
    ]
    return make_fixture([img], [cat], gts, dts)


def scenario_iou_at_threshold():
    """DT with IoU computed to be exactly at a threshold boundary."""
    img = make_image(1)
    cat = make_category(1)
    # GT: [0,0,100,100] area=10000
    # DT placed so IoU ≈ 0.50 exactly: intersection/union = 0.50
    # GT area=10000, DT bbox chosen so intersection=5000, union=10000+dt_area-5000
    # For IoU=0.5: intersection=union/2 → intersection=5000, union=10000 → dt_area=5000
    # DT=[50,0,50,100] → intersection=[50,0,50,100]=5000, union=10000+5000-5000=10000
    gt = make_gt(1, 1, 1, [0.0, 0.0, 100.0, 100.0])
    # This gives exactly 0.5 IoU
    dt = make_dt(1, 1, [50.0, 0.0, 50.0, 100.0], score=0.9)
    return make_fixture([img], [cat], [gt], [dt])


def scenario_iscrowd_mixed():
    """Mix of crowd and non-crowd GTs with DTs."""
    img = make_image(1)
    cat = make_category(1)
    gts = [
        make_gt(1, 1, 1, [0.0, 0.0, 100.0, 100.0], iscrowd=0),
        make_gt(2, 1, 1, [50.0, 50.0, 100.0, 100.0], iscrowd=1),
    ]
    dts = [
        make_dt(1, 1, [5.0, 5.0, 90.0, 90.0], score=0.9),  # should match GT 1
        make_dt(1, 1, [55.0, 55.0, 90.0, 90.0], score=0.8),  # overlaps crowd GT 2
    ]
    return make_fixture([img], [cat], gts, dts)


# ---------------------------------------------------------------------------
# Novel scenarios
# ---------------------------------------------------------------------------


def scenario_crowd_polygon_seg():
    """iscrowd=1 GT with polygon segmentation (not RLE) — unusual but valid."""
    img = make_image(1)
    cat = make_category(1)
    # Polygon: a simple square [x1,y1, x2,y2, x3,y3, x4,y4]
    polygon = [[10.0, 10.0, 90.0, 10.0, 90.0, 90.0, 10.0, 90.0]]
    gt = make_gt(1, 1, 1, [10.0, 10.0, 80.0, 80.0], area=6400.0, iscrowd=1, segmentation=polygon)
    dt = make_dt(1, 1, [15.0, 15.0, 70.0, 70.0], score=0.9)
    return make_fixture([img], [cat], [gt], [dt])


def scenario_cat_no_gt():
    """Category present in dataset but with no GTs at all."""
    img = make_image(1)
    cats = [make_category(1), make_category(2)]
    gts = [make_gt(1, 1, 1, [0.0, 0.0, 100.0, 100.0])]
    dts = [
        make_dt(1, 1, [0.0, 0.0, 80.0, 80.0], score=0.9),
        make_dt(1, 2, [0.0, 0.0, 80.0, 80.0], score=0.8),  # cat 2 has no GTs at all
    ]
    return make_fixture([img], cats, gts, dts)


def scenario_dt_unknown_cat():
    """DT category_id not present in GT categories at all."""
    img = make_image(1)
    cat = make_category(1)
    gt = make_gt(1, 1, 1, [0.0, 0.0, 100.0, 100.0])
    dts = [
        make_dt(1, 1, [0.0, 0.0, 80.0, 80.0], score=0.9),
        make_dt(1, 99, [0.0, 0.0, 80.0, 80.0], score=0.8),  # category 99 doesn't exist
    ]
    return make_fixture([img], [cat], [gt], dts)


def scenario_single_gt_101_dts_tied():
    """Single GT, 101 DTs all with identical score and slightly varying IoU."""
    img = make_image(1, width=2000, height=2000)
    cat = make_category(1)
    gt = make_gt(1, 1, 1, [0.0, 0.0, 100.0, 100.0])
    # 101 DTs, all same score, varying x offset (so slightly varying IoU)
    dts = [make_dt(1, 1, [float(i), 0.0, 95.0, 95.0], score=0.5) for i in range(101)]
    return make_fixture([img], [cat], [gt], dts)


def scenario_1x1_image():
    """Image dimensions of 1x1 (degenerate image)."""
    img = make_image(1, width=1, height=1)
    cat = make_category(1)
    gt = make_gt(1, 1, 1, [0.0, 0.0, 1.0, 1.0], area=1.0)
    dt = make_dt(1, 1, [0.0, 0.0, 1.0, 1.0], score=0.9)
    return make_fixture([img], [cat], [gt], [dt])


def scenario_bbox_outside_image():
    """GT bbox that extends outside image bounds."""
    img = make_image(1, width=100, height=100)
    cat = make_category(1)
    gts = [
        make_gt(1, 1, 1, [80.0, 80.0, 50.0, 50.0], area=2500.0),  # extends outside
        make_gt(2, 1, 1, [-10.0, -10.0, 50.0, 50.0], area=2500.0),  # negative coords
    ]
    dts = [make_dt(1, 1, [85.0, 85.0, 40.0, 40.0], score=0.9), make_dt(1, 1, [-5.0, -5.0, 40.0, 40.0], score=0.8)]
    return make_fixture([img], [cat], gts, dts)


def scenario_disjoint_image_sets():
    """Two images: DTs for image 1 only, GT for image 2 only (no overlap)."""
    images = [make_image(1), make_image(2)]
    cat = make_category(1)
    gts = [make_gt(1, 2, 1, [0.0, 0.0, 100.0, 100.0])]  # only image 2
    dts = [make_dt(1, 1, [0.0, 0.0, 80.0, 80.0], score=0.9)]  # only image 1
    return make_fixture(images, [cat], gts, dts)


def scenario_kp_zero_keypoints():
    """GT with num_keypoints=0 for keypoints eval."""
    img = make_image(1)
    cat = {
        "id": 1,
        "name": "person",
        "supercategory": "person",
        "keypoints": [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ],
        "skeleton": [],
    }
    # GT with all keypoints invisible (v=0) → num_keypoints=0
    kps = [0.0] * 51  # 17 keypoints * 3
    gt = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [0.0, 0.0, 100.0, 200.0],
        "area": 20000.0,
        "iscrowd": 0,
        "segmentation": [],
        "keypoints": kps,
        "num_keypoints": 0,
    }
    # DT with some keypoints
    kps_dt = [50.0, 100.0, 2.0] * 17
    dt = {"image_id": 1, "category_id": 1, "bbox": [0.0, 0.0, 100.0, 200.0], "score": 0.9, "keypoints": kps_dt}
    return make_fixture([img], [cat], [gt], [dt])


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def write_fixture(name, fixture_data):
    """Write fixture to scripts/fixtures/adversarial/<name>.json. Returns Path."""
    path = FIXTURE_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(fixture_data, f, indent=2)
    return path


def run_harness(fixture_path, iou_type="bbox", metric_thr=1e-4):
    """
    Run adversarial_harness.py on the fixture.
    Returns (passed: bool, report: str).
    """
    cmd = [sys.executable, str(HARNESS), str(fixture_path), "--iou-type", iou_type, "--metric-thr", str(metric_thr)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    report = result.stdout + (("\nSTDERR:\n" + result.stderr) if result.stderr.strip() else "")
    passed = result.returncode == 0
    return passed, report
