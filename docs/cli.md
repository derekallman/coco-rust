# CLI Reference

The `coco-eval` command runs COCO evaluation from the terminal.

!!! note "Installation"
    ```bash
    cargo install coco-cli
    ```
    See [Installation](installation.md) for more options.

## Basic Usage

```bash
coco-eval --gt annotations.json --dt detections.json --iou-type bbox
```

This loads ground truth and detection files, runs evaluation, and prints the standard COCO metrics.

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--gt <path>` | Path to ground truth annotations JSON file | *required* |
| `--dt <path>` | Path to detection results JSON file | *required* |
| `--iou-type <type>` | Evaluation type: `bbox`, `segm`, or `keypoints` | `bbox` |
| `--img-ids <ids>` | Filter to specific image IDs (comma-separated) | all images |
| `--cat-ids <ids>` | Filter to specific category IDs (comma-separated) | all categories |
| `--max-dets <values>` | Max detections per image (comma-separated) | `1,10,100` |
| `-a`, `--category-agnostic` | Pool all categories for category-agnostic evaluation | off |

## Examples

### Bounding Box Evaluation

```bash
coco-eval --gt instances_val2017.json --dt bbox_results.json --iou-type bbox
```

### Segmentation Evaluation

```bash
coco-eval --gt instances_val2017.json --dt segm_results.json --iou-type segm
```

### Keypoint Evaluation

```bash
coco-eval --gt person_keypoints_val2017.json --dt kpt_results.json --iou-type keypoints
```

### Filter by Category

Evaluate only "person" (category 1) and "car" (category 3):

```bash
coco-eval --gt instances_val2017.json --dt results.json --cat-ids 1,3
```

### Filter by Image

Evaluate only specific images:

```bash
coco-eval --gt instances_val2017.json --dt results.json --img-ids 139,285,632
```

### Custom Max Detections

```bash
coco-eval --gt instances_val2017.json --dt results.json --max-dets 1,10,50
```

### Category-Agnostic Evaluation

Pool all categories together (ignores category labels):

```bash
coco-eval --gt instances_val2017.json --dt results.json -a
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

A machine-readable stats line is also printed to stdout:

```
stats: [0.783..., 0.971..., ...]
```
