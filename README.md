# coco-rust

A Rust implementation of [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools) — the standard COCO dataset API for object detection, segmentation, and keypoint evaluation.

## Features

- Full COCO evaluation pipeline: load annotations, compute IoU, match detections, accumulate metrics
- All three evaluation types: **bbox**, **segm**, **keypoints**
- Pure Rust RLE mask operations (encode, decode, merge, IoU, polygon rasterization)
- Exact metric parity with pycocotools — identical AP/AR numbers on the same inputs
- ~3x faster than pycocotools on bbox evaluation

## Performance

Bbox evaluation on COCO val2017 (5,000 images, 80 categories, 36k detections):

| Tool | Time |
|------|------|
| pycocotools | ~15s |
| coco-rust | ~5s |

Benchmarked on Apple M-series, single-threaded.

## Installation

```bash
# Build from source
cargo build --release

# Run tests
cargo test
```

## Usage

### CLI

```bash
coco-eval --gt annotations.json --dt detections.json --iou-type bbox
```

Options:
- `--iou-type` — `bbox`, `segm`, or `keypoints`
- `--img-ids` — filter to specific image IDs (comma-separated)
- `--cat-ids` — filter to specific category IDs (comma-separated)
- `--max-dets` — max detections per image (comma-separated, e.g. `1,10,100`)

### Library

```rust
use coco_rs::{COCO, COCOeval};
use coco_rs::params::IouType;
use std::path::Path;

let coco_gt = COCO::new(Path::new("annotations.json")).unwrap();
let coco_dt = coco_gt.load_res(Path::new("detections.json")).unwrap();

let mut eval = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
eval.evaluate();
eval.accumulate();
eval.summarize();
```

### Output

```
 Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.783
 Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.971
 Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.849
 Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.621
 Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.893
 Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.988
 Average Recall (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.502
 Average Recall (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.835
 Average Recall (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.854
 Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.701
 Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.935
 Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.997
```

## Project Structure

```
crates/
  coco-rs/     # Core library
    src/
      types.rs   # Dataset, Annotation, Image, Category, Segmentation, Rle
      mask.rs    # RLE encode/decode, IoU, polygon rasterization, LEB128 codec
      coco.rs    # Dataset loading, indexing, querying
      params.rs  # Evaluation parameters and defaults
      eval.rs    # COCOeval: evaluate, accumulate, summarize
  coco-cli/    # CLI binary
```

## License

MIT
