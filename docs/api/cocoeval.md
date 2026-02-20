# COCOeval

Run COCO evaluation to compute AP/AR metrics.

=== "Python"

    ```python
    from coco_rs import COCO, COCOeval

    coco_gt = COCO("instances_val2017.json")
    coco_dt = coco_gt.load_res("detections.json")

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    ```

=== "Rust"

    ```rust
    use coco_rs::{COCO, COCOeval};
    use coco_rs::params::IouType;
    use std::path::Path;

    let coco_gt = COCO::new(Path::new("instances_val2017.json"))?;
    let coco_dt = coco_gt.load_res(Path::new("detections.json"))?;

    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();
    ev.accumulate();
    ev.summarize();
    ```

---

## Constructor

=== "Python"

    ```python
    COCOeval(coco_gt: COCO, coco_dt: COCO, iou_type: str)
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `coco_gt` | `COCO` | Ground truth COCO object |
    | `coco_dt` | `COCO` | Detections COCO object (from `load_res`) |
    | `iou_type` | `str` | `"bbox"`, `"segm"`, or `"keypoints"` |

=== "Rust"

    ```rust
    COCOeval::new(coco_gt: COCO, coco_dt: COCO, iou_type: IouType) -> Self
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `coco_gt` | `COCO` | Ground truth COCO object |
    | `coco_dt` | `COCO` | Detections COCO object (from `load_res`) |
    | `iou_type` | `IouType` | `IouType::Bbox`, `IouType::Segm`, or `IouType::Keypoints` |

---

## Properties

### `params`

=== "Python"

    ```python
    params: Params
    ```

    Evaluation parameters. Modify before calling `evaluate()`.

    ```python
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.cat_ids = [1, 2, 3]
    ev.params.max_dets = [1, 10, 100]
    ```

=== "Rust"

    ```rust
    pub params: Params
    ```

    ```rust
    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.params.cat_ids = vec![1, 2, 3];
    ev.params.max_dets = vec![1, 10, 100];
    ```

See [Params](params.md) for all configurable fields.

---

### `stats`

=== "Python"

    ```python
    stats: list[float] | None
    ```

    The 12 summary metrics (10 for keypoints), populated after `summarize()`. `None` before `summarize()` is called.

    ```python
    ev.summarize()
    print(f"AP: {ev.stats[0]:.3f}")
    print(f"AP50: {ev.stats[1]:.3f}")
    ```

=== "Rust"

    ```rust
    pub stats: Option<Vec<f64>>
    ```

    ```rust
    ev.summarize();
    if let Some(stats) = &ev.stats {
        println!("AP: {:.3}", stats[0]);
        println!("AP50: {:.3}", stats[1]);
    }
    ```

---

### `eval_imgs`

Per-image evaluation results, populated after `evaluate()`. See [Working with Results](../guide/results.md) for details.

=== "Python"

    ```python
    eval_imgs: list[dict | None]
    ```

=== "Rust"

    ```rust
    pub eval_imgs: Vec<Option<EvalImg>>
    ```

---

### `eval`

Accumulated precision/recall arrays, populated after `accumulate()`. See [Working with Results](../guide/results.md) for details.

=== "Python"

    ```python
    eval: dict | None
    ```

    Contains `"precision"`, `"recall"`, and `"scores"` arrays.

=== "Rust"

    ```rust
    pub eval: Option<AccumulatedEval>
    ```

    Access elements with `precision_idx(t, r, k, a, m)` and `recall_idx(t, k, a, m)`.

---

## Methods

### `evaluate`

```python
evaluate() -> None
```

Run per-image evaluation. Matches detections to ground truth annotations using greedy matching sorted by confidence. Must be called before `accumulate()`.

Populates `eval_imgs`.

---

### `accumulate`

```python
accumulate() -> None
```

Accumulate per-image results into precision/recall curves using interpolated precision at 101 recall thresholds.

Populates `eval`.

---

### `summarize`

```python
summarize() -> None
```

Compute and print the standard COCO metrics. Populates `stats`.

!!! warning "Non-default parameters"
    `summarize()` uses a fixed display format that assumes default `iou_thrs`, `max_dets`, and `area_rng_lbl`. If you've changed any of these, a warning is printed to stderr and some metrics may show `-1.000` (e.g. AP50 when `iou_thrs` doesn't include 0.50). The `stats` array always has 12 entries (10 for keypoints) regardless of your parameters.

Prints 12 lines for bbox/segm (10 for keypoints):

```
 Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.382
 Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.584
 ...
```
