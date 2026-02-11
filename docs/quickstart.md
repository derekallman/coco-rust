# Quick Start

This guide walks through a complete COCO evaluation using Python. By the end, you'll have loaded a dataset, run evaluation, and interpreted the results.

## 1. Install

```bash
pip install coco-rs
```

!!! note "numpy required"
    coco-rs requires numpy for mask operations. It is installed automatically as a dependency.

## 2. Load ground truth annotations

```python
from coco_rs import COCO

coco_gt = COCO("instances_val2017.json")

# Explore the dataset
print(f"Images: {len(coco_gt.get_img_ids())}")
print(f"Categories: {len(coco_gt.get_cat_ids())}")
print(f"Annotations: {len(coco_gt.get_ann_ids())}")
```

The `COCO` object indexes your annotation file and provides fast lookups by image, category, and annotation ID.

## 3. Load detection results

```python
coco_dt = coco_gt.load_res("detections.json")
```

!!! tip
    `load_res` automatically computes missing `area` fields from bounding boxes or segmentation masks — no need to add them to your results file.

Your results file should be a JSON array of detection dicts:

```json
[
  {"image_id": 42, "category_id": 1, "bbox": [10, 20, 30, 40], "score": 0.95},
  ...
]
```

## 4. Run evaluation

```python
from coco_rs import COCOeval

ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.evaluate()
ev.accumulate()
ev.summarize()
```

This prints the standard 12 COCO metrics:

```
 Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.382
 Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.584
 ...
```

## 5. Access metrics programmatically

```python
stats = ev.stats

ap = stats[0]       # AP @ IoU=0.50:0.95, area=all
ap_50 = stats[1]    # AP @ IoU=0.50
ap_75 = stats[2]    # AP @ IoU=0.75
ap_s = stats[3]     # AP @ area=small
ap_m = stats[4]     # AP @ area=medium
ap_l = stats[5]     # AP @ area=large
ar_1 = stats[6]     # AR @ maxDets=1
ar_10 = stats[7]    # AR @ maxDets=10
ar_100 = stats[8]   # AR @ maxDets=100
ar_s = stats[9]     # AR @ area=small
ar_m = stats[10]    # AR @ area=medium
ar_l = stats[11]    # AR @ area=large
```

## 6. Customize evaluation

You can modify evaluation parameters before calling `evaluate()`:

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")

# Evaluate only specific categories
ev.params.cat_ids = [1, 2, 3]

# Evaluate only specific images
ev.params.img_ids = [42, 139, 285]

# Change max detections
ev.params.max_dets = [1, 10, 50]

ev.evaluate()
ev.accumulate()
ev.summarize()
```

## Next steps

- [Python API](python.md) — full API reference for all classes and methods
- [CLI Reference](cli.md) — run evaluations from the terminal
- [Rust Library](rust.md) — use coco-rust directly from Rust
