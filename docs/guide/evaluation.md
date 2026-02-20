# Evaluation

coco-rust supports three evaluation types: bounding box, segmentation, and keypoints. All three follow the same workflow.

## The three-step pipeline

Every COCO evaluation follows the same pattern:

=== "Python"

    ```python
    from coco_rs import COCO, COCOeval

    coco_gt = COCO("annotations.json")
    coco_dt = coco_gt.load_res("detections.json")

    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.evaluate()    # Per-image matching
    ev.accumulate()  # Aggregate into precision/recall curves
    ev.summarize()   # Print and compute the 12 summary metrics
    ```

=== "Rust"

    ```rust
    use coco_rs::{COCO, COCOeval};
    use coco_rs::params::IouType;
    use std::path::Path;

    let coco_gt = COCO::new(Path::new("annotations.json"))?;
    let coco_dt = coco_gt.load_res(Path::new("detections.json"))?;

    let mut ev = COCOeval::new(coco_gt, coco_dt, iou_type);
    ev.evaluate();    // Per-image matching
    ev.accumulate();  // Aggregate into precision/recall curves
    ev.summarize();   // Print and compute the 12 summary metrics
    ```

The only thing that changes between eval types is the `iou_type` parameter and the format of your detections.

## Bounding box evaluation

Set `iou_type` to `"bbox"` (Python) or `IouType::Bbox` (Rust).

Detection format — each result needs `image_id`, `category_id`, `bbox` as `[x, y, width, height]`, and `score`:

```json
[
  {"image_id": 42, "category_id": 1, "bbox": [10.0, 20.0, 30.0, 40.0], "score": 0.95},
  ...
]
```

IoU is computed as the intersection-over-union of the two bounding boxes.

## Segmentation evaluation

Set `iou_type` to `"segm"` (Python) or `IouType::Segm` (Rust).

Detection format — each result needs `image_id`, `category_id`, `segmentation` as an RLE dict, and `score`:

```json
[
  {
    "image_id": 42,
    "category_id": 1,
    "segmentation": {"counts": "abc123...", "size": [480, 640]},
    "score": 0.95
  },
  ...
]
```

IoU is computed on the binary masks after RLE decoding.

!!! tip
    If your results only have bounding boxes, use bbox evaluation instead. `load_res` generates polygon segmentations from bboxes, but these are axis-aligned rectangles — not instance masks.

## Keypoint evaluation

Set `iou_type` to `"keypoints"` (Python) or `IouType::Keypoints` (Rust).

Detection format — each result needs `image_id`, `category_id`, `keypoints` as a flat list of `[x1, y1, v1, x2, y2, v2, ...]`, and `score`:

```json
[
  {
    "image_id": 42,
    "category_id": 1,
    "keypoints": [x1, y1, v1, x2, y2, v2, ...],
    "score": 0.95
  },
  ...
]
```

Each keypoint has an `(x, y)` position and a visibility flag `v` (0 = not labeled, 1 = labeled but not visible, 2 = labeled and visible).

Similarity is measured using Object Keypoint Similarity (OKS) instead of IoU. OKS uses per-keypoint sigma values that account for annotation noise — keypoints with higher variance (like hips) are weighted less strictly than precise ones (like eyes).

**Differences from bbox/segm:**

- 10 metrics instead of 12 (no small area range — keypoints are only meaningful on medium and large objects)
- Default max detections is `[20]` instead of `[1, 10, 100]`
- Ground truth annotations with `num_keypoints == 0` are automatically ignored

## The 12 COCO metrics

`summarize()` computes and prints these metrics (10 for keypoints):

| Index | Metric | IoU | Area | MaxDets |
|-------|--------|-----|------|---------|
| 0 | **AP** | 0.50:0.95 | all | 100 |
| 1 | AP | 0.50 | all | 100 |
| 2 | AP | 0.75 | all | 100 |
| 3 | AP | 0.50:0.95 | small | 100 |
| 4 | AP | 0.50:0.95 | medium | 100 |
| 5 | AP | 0.50:0.95 | large | 100 |
| 6 | AR | 0.50:0.95 | all | 1 |
| 7 | AR | 0.50:0.95 | all | 10 |
| 8 | AR | 0.50:0.95 | all | 100 |
| 9 | AR | 0.50:0.95 | small | 100 |
| 10 | AR | 0.50:0.95 | medium | 100 |
| 11 | AR | 0.50:0.95 | large | 100 |

- **AP** (Average Precision) is the area under the precision-recall curve, averaged across IoU thresholds.
- **AR** (Average Recall) is the maximum recall at a fixed number of detections per image, averaged across IoU thresholds.
- **Area ranges**: small (0-32²), medium (32²-96²), large (96²+) pixels.

## Customizing parameters

Modify `ev.params` before calling `evaluate()`:

=== "Python"

    ```python
    ev = COCOeval(coco_gt, coco_dt, "bbox")

    # Evaluate a subset of categories
    ev.params.cat_ids = [1, 3]

    # Evaluate a subset of images
    ev.params.img_ids = [42, 139]

    # Custom IoU thresholds
    ev.params.iou_thrs = [0.5, 0.75, 0.9]

    # Custom max detections
    ev.params.max_dets = [1, 10, 100]

    # Category-agnostic evaluation (pool all categories)
    ev.params.use_cats = False

    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    ```

=== "Rust"

    ```rust
    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);

    ev.params.cat_ids = vec![1, 3];
    ev.params.img_ids = vec![42, 139];
    ev.params.iou_thrs = vec![0.5, 0.75, 0.9];
    ev.params.max_dets = vec![1, 10, 100];
    ev.params.use_cats = false;

    ev.evaluate();
    ev.accumulate();
    ev.summarize();
    ```

!!! note
    Changing `iou_thrs`, `max_dets`, or `area_rng_lbl` from their defaults affects what `summarize()` can display. The 12-metric output format is fixed — for example, AP50 looks for IoU=0.50 in your thresholds and shows `-1.000` if it's not there. A warning is printed when your parameters don't match the expected defaults. Filtering by `img_ids`, `cat_ids`, or setting `use_cats` is safe and won't trigger warnings.

See [Params](../api/params.md) for the full list of configurable parameters.
