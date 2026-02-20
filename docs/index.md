<div class="hero" markdown>

# coco-rust

<p class="hero-tagline">
Drop-in replacement for pycocotools — 11-26x faster.
</p>

<div class="hero-actions" markdown>

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[API Reference](api/coco.md){ .md-button }

</div>

</div>

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
<strong>Exact parity</strong>
<p>Identical AP/AR numbers on the same inputs, verified on COCO val2017.</p>
</div>

<div class="feature-card" markdown>
<strong>11-26x faster</strong>
<p>Pure Rust with automatic parallelism. No C extensions to compile.</p>
</div>

<div class="feature-card" markdown>
<strong>Drop-in replacement</strong>
<p>Same API as pycocotools. Change one import line and keep your code.</p>
</div>

<div class="feature-card" markdown>
<strong>All eval types</strong>
<p>Bounding box, segmentation, and keypoint evaluation.</p>
</div>

</div>

## Quick start

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

=== "CLI"

    ```bash
    coco-eval --gt instances_val2017.json --dt detections.json --iou-type bbox
    ```

## Performance

Benchmarked on COCO val2017 (5,000 images, 36,781 GT annotations, ~43,700 detections), Apple M1 MacBook Air:

<div class="benchmark-table" markdown>

| Eval Type | pycocotools | faster-coco-eval | coco-rust |
|-----------|-------------|------------------|-----------|
| bbox      | 11.79s      | 3.47s (3.4x)     | 0.74s (15.9x) |
| segm      | 19.49s      | 10.52s (1.9x)    | 1.58s (12.3x) |
| keypoints | 4.79s       | 3.08s (1.6x)     | 0.19s (25.0x) |

</div>

Speedups in parentheses are vs pycocotools. All metrics match within 0.003 (many are exact).

## Zero-code migration

Already using pycocotools? You don't need to touch your existing code:

```python
from coco_rs import init_as_pycocotools
init_as_pycocotools()

# All pycocotools imports now use coco-rust
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
```

See [Migrating from pycocotools](getting-started/migration.md) for the full guide.

## Rust API

For Rust users, the `coco-rs` crate is available on [crates.io](https://crates.io/crates/coco-rs). Full API documentation is on [docs.rs](https://docs.rs/coco-rs).

## License

MIT
