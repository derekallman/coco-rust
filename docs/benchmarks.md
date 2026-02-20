# Benchmarks

## Setup

- **Hardware:** Apple M1 MacBook Air, 16 GB RAM
- **Dataset:** COCO val2017 — 5,000 images, 36,781 ground truth annotations
- **Detections:** ~43,700 detections (1x scale)
- **Timing:** Wall clock time, best of 3 runs
- **Versions:** pycocotools 2.0.8, faster-coco-eval 1.6.5, coco-rust 0.1.0

## Results (1x detections)

| Eval Type | pycocotools | faster-coco-eval | coco-rust |
|-----------|-------------|------------------|-----------|
| bbox      | 11.79s      | 3.47s (3.4x)     | 0.74s (15.9x) |
| segm      | 19.49s      | 10.52s (1.9x)    | 1.58s (12.3x) |
| keypoints | 4.79s       | 3.08s (1.6x)     | 0.19s (25.0x) |

Speedups in parentheses are vs pycocotools.

## Results (10x detections)

Synthetic benchmark scaling detections by 10x (~437,000 detections) to test behavior at scale:

| Eval Type | pycocotools | faster-coco-eval | coco-rust |
|-----------|-------------|------------------|-----------|
| bbox      | 106.27s     | 27.68s (3.8x)    | 4.07s (26.1x) |
| segm      | 184.35s     | 99.73s (1.8x)    | 10.84s (17.0x) |
| keypoints | 42.60s      | 26.54s (1.6x)    | 0.93s (45.8x) |

coco-rust scales better at higher detection counts due to multi-threaded evaluation.

## Metric parity

All metrics match pycocotools within 0.003 (many are exact). Verified on COCO val2017:

### Bounding box

| Metric | pycocotools | coco-rust | Diff |
|--------|-------------|-----------|------|
| AP     | 0.382       | 0.382     | 0.000 |
| AP50   | 0.584       | 0.584     | 0.000 |
| AP75   | 0.412       | 0.412     | 0.000 |
| APs    | 0.209       | 0.209     | 0.000 |
| APm    | 0.420       | 0.420     | 0.000 |
| APl    | 0.529       | 0.529     | 0.000 |
| AR1    | 0.323       | 0.323     | 0.000 |
| AR10   | 0.498       | 0.498     | 0.000 |
| AR100  | 0.520       | 0.520     | 0.000 |
| ARs    | 0.308       | 0.308     | 0.000 |
| ARm    | 0.562       | 0.562     | 0.000 |
| ARl    | 0.680       | 0.680     | 0.000 |

Bounding box metrics are exact.

### Segmentation

| Metric | pycocotools | coco-rust | Diff |
|--------|-------------|-----------|------|
| AP     | 0.355       | 0.355     | 0.000 |
| AP50   | 0.568       | 0.568     | 0.000 |
| AP75   | 0.377       | 0.377     | 0.000 |
| APs    | 0.163       | 0.163     | 0.000 |
| APm    | 0.384       | 0.384     | 0.000 |
| APl    | 0.531       | 0.531     | 0.000 |
| AR1    | 0.303       | 0.303     | 0.000 |
| AR10   | 0.462       | 0.462     | 0.000 |
| AR100  | 0.482       | 0.482     | 0.000 |
| ARs    | 0.259       | 0.259     | 0.000 |
| ARm    | 0.521       | 0.521     | 0.000 |
| ARl    | 0.672       | 0.672     | 0.000 |

Segmentation metrics match within 0.003 (shown rounded to 3 decimal places).

### Keypoints

| Metric | pycocotools | coco-rust | Diff |
|--------|-------------|-----------|------|
| AP     | 0.669       | 0.669     | 0.000 |
| AP50   | 0.873       | 0.873     | 0.000 |
| AP75   | 0.730       | 0.730     | 0.000 |
| APm    | 0.635       | 0.635     | 0.000 |
| APl    | 0.732       | 0.732     | 0.000 |
| AR1    | 0.291       | 0.291     | 0.000 |
| AR10   | 0.707       | 0.707     | 0.000 |
| AR100  | 0.739       | 0.739     | 0.000 |
| ARm    | 0.685       | 0.685     | 0.000 |
| ARl    | 0.815       | 0.815     | 0.000 |

Keypoint metrics are exact.

## Methodology

- **Wall clock time** includes file I/O, evaluation, and accumulation. Excludes Python import time.
- **Only detections are scaled** for the 10x benchmark — ground truth annotations are unchanged.
- All three tools were verified to produce identical metrics before timing.
- Benchmark scripts are in the repository under `data/`.
