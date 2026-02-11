# Rust Library

The `coco-rs` crate provides the full COCO evaluation pipeline as a Rust library.

## Quick Start

Add `coco-rs` to your project:

```bash
cargo add coco-rs
```

Run an evaluation:

```rust
use coco_rs::{COCO, COCOeval};
use coco_rs::params::IouType;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let coco_gt = COCO::new(Path::new("annotations.json"))?;
    let coco_dt = coco_gt.load_res(Path::new("detections.json"))?;

    let mut eval = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    eval.evaluate();
    eval.accumulate();
    eval.summarize();

    if let Some(stats) = &eval.stats {
        println!("AP @[IoU=0.50:0.95]: {:.3}", stats[0]);
    }

    Ok(())
}
```

## Key Types

| Type | Module | Description |
|------|--------|-------------|
| `COCO` | `coco_rs` | Dataset loader and query interface |
| `COCOeval` | `coco_rs` | Evaluation engine (evaluate, accumulate, summarize) |
| `Params` | `coco_rs::params` | Evaluation parameters (IoU thresholds, area ranges, etc.) |
| `IouType` | `coco_rs::params` | Enum: `Bbox`, `Segm`, `Keypoints` |
| `Rle` | `coco_rs::mask` | Run-length encoded mask |
| `Dataset` | `coco_rs::types` | Full COCO dataset structure |
| `Annotation` | `coco_rs::types` | Single annotation (bbox, segmentation, keypoints) |

## Mask Operations

The `coco_rs::mask` module provides low-level RLE mask operations:

```rust
use coco_rs::mask;

// Encode a binary mask to RLE
let rle = mask::encode(&pixel_data, height, width);

// Decode RLE to pixels
let pixels = mask::decode(&rle);

// Compute IoU between masks
let ious = mask::iou(&dt_rles, &gt_rles, &iscrowd);

// Rasterize a polygon
let rle = mask::fr_poly(&[x1, y1, x2, y2, x3, y3], h, w);
```

## API Documentation

For the full API reference, see the auto-generated documentation on [docs.rs](https://docs.rs/coco-rs).
