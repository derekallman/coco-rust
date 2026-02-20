# Quick Start

A complete COCO evaluation in under a minute.

## 1. Install

=== "Python"

    ```bash
    pip install coco-rs
    ```

=== "Rust"

    ```bash
    cargo add coco-rs
    ```

=== "CLI"

    ```bash
    cargo install coco-cli
    ```

## 2. Load ground truth

=== "Python"

    ```python
    from coco_rs import COCO

    coco_gt = COCO("instances_val2017.json")

    print(f"Images: {len(coco_gt.get_img_ids())}")
    print(f"Categories: {len(coco_gt.get_cat_ids())}")
    print(f"Annotations: {len(coco_gt.get_ann_ids())}")
    ```

=== "Rust"

    ```rust
    use coco_rs::COCO;
    use std::path::Path;

    let coco_gt = COCO::new(Path::new("instances_val2017.json"))?;

    println!("Images: {}", coco_gt.get_img_ids(&[], &[]).len());
    println!("Categories: {}", coco_gt.get_cat_ids(&[], &[], &[]).len());
    println!("Annotations: {}", coco_gt.get_ann_ids(&[], &[], None, None).len());
    ```

=== "CLI"

    The CLI handles loading automatically — skip to step 4.

## 3. Load detection results

=== "Python"

    ```python
    coco_dt = coco_gt.load_res("detections.json")
    ```

=== "Rust"

    ```rust
    let coco_dt = coco_gt.load_res(Path::new("detections.json"))?;
    ```

Your results file should be a JSON array of detection dicts:

```json
[
  {"image_id": 42, "category_id": 1, "bbox": [10, 20, 30, 40], "score": 0.95},
  ...
]
```

!!! tip
    `load_res` automatically computes missing `area` fields from bounding boxes or segmentation masks.

## 4. Run evaluation

=== "Python"

    ```python
    from coco_rs import COCOeval

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    ```

=== "Rust"

    ```rust
    use coco_rs::COCOeval;
    use coco_rs::params::IouType;

    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();
    ev.accumulate();
    ev.summarize();
    ```

=== "CLI"

    ```bash
    coco-eval --gt instances_val2017.json --dt detections.json --iou-type bbox
    ```

Output:

```
 Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.382
 Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.584
 Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.412
 Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.209
 Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.420
 Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.529
 Average Recall (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.323
 Average Recall (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.498
 Average Recall (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.520
 Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.308
 Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.562
 Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.680
```

## 5. Access metrics programmatically

=== "Python"

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

=== "Rust"

    ```rust
    if let Some(stats) = &ev.stats {
        let ap = stats[0];       // AP @ IoU=0.50:0.95, area=all
        let ap_50 = stats[1];    // AP @ IoU=0.50
        let ap_75 = stats[2];    // AP @ IoU=0.75
        println!("AP: {ap:.3}, AP50: {ap_50:.3}, AP75: {ap_75:.3}");
    }
    ```

## 6. Customize evaluation

=== "Python"

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

=== "Rust"

    ```rust
    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);

    // Evaluate only specific categories
    ev.params.cat_ids = vec![1, 2, 3];

    // Evaluate only specific images
    ev.params.img_ids = vec![42, 139, 285];

    // Change max detections
    ev.params.max_dets = vec![1, 10, 50];

    ev.evaluate();
    ev.accumulate();
    ev.summarize();
    ```

## Next steps

- [Evaluation](../guide/evaluation.md) — bbox, segm, and keypoint workflows explained
- [Working with Results](../guide/results.md) — load_res, eval_imgs, precision/recall arrays
- [API Reference](../api/coco.md) — full class and method reference
