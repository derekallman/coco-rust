# coco-rust

**A drop-in replacement for pycocotools — 11-26x faster.**

Swap one import line, keep your existing code, and get identical COCO evaluation results in a fraction of the time. Available as a Python package, CLI tool, and Rust library.

## Get started in 5 lines

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

    let mut eval = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    eval.evaluate();
    eval.accumulate();
    eval.summarize();
    ```

=== "CLI"

    ```bash
    coco-eval --gt instances_val2017.json --dt detections.json --iou-type bbox
    ```

## Performance

Benchmarked on COCO val2017 (5,000 images, 36,781 ground truth annotations, ~43,700 detections), Apple M1 MacBook Air:

| Eval Type | pycocotools | faster-coco-eval | coco-rust |
|-----------|-------------|------------------|-----------|
| bbox      | 11.79s      | 3.47s (3.4x)    | 0.74s (15.9x) |
| segm      | 19.49s      | 10.52s (1.9x)   | 1.58s (12.3x) |
| keypoints | 4.79s       | 3.08s (1.6x)    | 0.19s (25.0x) |

Speedups in parentheses are vs pycocotools. All metrics match pycocotools within 0.003 (many are exact).

## Why coco-rust?

- **Exact parity** — identical AP/AR numbers on the same inputs, verified on COCO val2017
- **11-26x faster** — pure Rust with automatic parallelism, no C extensions to compile
- **Three interfaces** — use it from Python, the command line, or as a Rust library
- **Drop-in replacement** — same API as pycocotools, just change the import
- **All eval types** — bbox, segmentation, and keypoint evaluation

## License

MIT
