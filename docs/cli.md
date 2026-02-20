# CLI

The `coco-eval` command runs COCO evaluation from the terminal.

## Installation

```bash
cargo install coco-cli
```

## Usage

```bash
coco-eval --gt annotations.json --dt detections.json --iou-type bbox
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--gt <path>` | Path to ground truth annotations JSON file | *required* |
| `--dt <path>` | Path to detection results JSON file | *required* |
| `--iou-type <type>` | Evaluation type: `bbox`, `segm`, or `keypoints` | `bbox` |
| `--img-ids <ids>` | Filter to specific image IDs (comma-separated) | all images |
| `--cat-ids <ids>` | Filter to specific category IDs (comma-separated) | all categories |
| `--no-cats` | Pool all categories (disable per-category evaluation) | off |

## Examples

### Bounding box evaluation

```bash
coco-eval --gt instances_val2017.json --dt bbox_results.json --iou-type bbox
```

### Segmentation evaluation

```bash
coco-eval --gt instances_val2017.json --dt segm_results.json --iou-type segm
```

### Keypoint evaluation

```bash
coco-eval --gt person_keypoints_val2017.json --dt kpt_results.json --iou-type keypoints
```

### Filter by category

Evaluate only "person" (category 1) and "car" (category 3):

```bash
coco-eval --gt instances_val2017.json --dt results.json --cat-ids 1,3
```

### Filter by image

```bash
coco-eval --gt instances_val2017.json --dt results.json --img-ids 139,285,632
```

### Category-agnostic evaluation

Pool all categories together (ignores category labels):

```bash
coco-eval --gt instances_val2017.json --dt results.json --no-cats
```

## Output

The tool prints the standard 12 COCO metrics (10 for keypoints):

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
