# Python API

The `coco_rs` Python module provides a drop-in replacement for pycocotools with the same API conventions. All objects (annotations, images, categories) are returned as plain Python dicts.

!!! tip "Migrating from pycocotools"
    **Option 1: Change imports** (recommended for your own code)

    ```python
    # Before
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # After
    from coco_rs import COCO, COCOeval
    ```

    **Option 2: Zero-code-change drop-in** (useful for third-party libraries)

    Call `init_as_pycocotools()` once at startup and all existing `pycocotools` imports will use coco-rust:

    ```python
    from coco_rs import init_as_pycocotools
    init_as_pycocotools()

    # These now use coco-rust — no changes needed in library code
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask
    ```

    This patches `sys.modules` so that `pycocotools`, `pycocotools.coco`, `pycocotools.cocoeval`, and `pycocotools.mask` all resolve to `coco_rs`. Both camelCase (`getAnnIds`, `loadRes`, `maxDets`) and snake_case (`get_ann_ids`, `load_res`, `max_dets`) method/property names are supported.

## End-to-end example

```python
from coco_rs import COCO, COCOeval

# Load ground truth and detections
coco_gt = COCO("instances_val2017.json")
coco_dt = coco_gt.load_res("bbox_results.json")

# Run evaluation
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.evaluate()
ev.accumulate()
ev.summarize()

# Access results
print(f"AP @[IoU=0.50:0.95]: {ev.stats[0]:.3f}")
print(f"AP @[IoU=0.50]:      {ev.stats[1]:.3f}")
```

---

## COCO

Load and query COCO-format datasets.

```python
from coco_rs import COCO

coco = COCO("instances_val2017.json")
```

### Constructor

```python
COCO(annotation_file=None)
```

- `annotation_file` — Path to a COCO JSON annotation file. If `None`, creates an empty dataset.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `dataset` | `dict` | The full dataset with `images`, `annotations`, and `categories` lists |

### Methods

#### `get_ann_ids(img_ids=[], cat_ids=[], area_rng=None, iscrowd=None)`

Get annotation IDs matching the given filters.

```python
ann_ids = coco.get_ann_ids(img_ids=[42], cat_ids=[1])
```

#### `get_cat_ids(cat_nms=[], sup_nms=[], cat_ids=[])`

Get category IDs matching the given filters.

```python
cat_ids = coco.get_cat_ids(cat_nms=["person", "dog"])
```

#### `get_img_ids(img_ids=[], cat_ids=[])`

Get image IDs matching the given filters.

```python
img_ids = coco.get_img_ids(cat_ids=[1])
```

#### `load_anns(ids)`

Load annotations by their IDs. Returns a list of annotation dicts.

```python
anns = coco.load_anns([101, 102, 103])
```

#### `load_cats(ids)`

Load categories by their IDs. Returns a list of category dicts.

```python
cats = coco.load_cats([1, 2, 3])
```

#### `load_imgs(ids)`

Load images by their IDs. Returns a list of image dicts.

```python
imgs = coco.load_imgs([42])
```

#### `load_res(res_file)`

Load detection results from a JSON file. Returns a new `COCO` object.

!!! tip
    `load_res` automatically computes missing fields: `area` from bounding boxes or segmentation masks, and polygon segmentations from bbox results. This matches pycocotools behavior.

```python
coco_dt = coco.load_res("detections.json")
```

#### `ann_to_rle(ann)`

Convert an annotation dict to RLE format. Returns an RLE dict with `counts` (string) and `size` [h, w].

```python
rle = coco.ann_to_rle(ann)
```

#### `ann_to_mask(ann)`

Convert an annotation dict to a binary mask. Returns a numpy array of shape (h, w).

```python
mask = coco.ann_to_mask(ann)
```

---

## COCOeval

Run COCO evaluation to compute AP/AR metrics.

```python
from coco_rs import COCO, COCOeval

coco_gt = COCO("instances_val2017.json")
coco_dt = coco_gt.load_res("detections.json")

ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.evaluate()
ev.accumulate()
ev.summarize()

print(ev.stats)
```

### Constructor

```python
COCOeval(coco_gt, coco_dt, iou_type)
```

- `coco_gt` — Ground truth `COCO` object
- `coco_dt` — Detections `COCO` object (from `load_res`)
- `iou_type` — `"bbox"`, `"segm"`, or `"keypoints"`

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `params` | `Params` | Evaluation parameters (read/write) |
| `stats` | `list[float]` or `None` | Result metrics after `summarize()` |

### Methods

#### `evaluate()`

Run per-image evaluation. Must be called before `accumulate()`.

#### `accumulate()`

Accumulate per-image results into summary metrics.

#### `summarize()`

Print and compute the standard 12 COCO metrics (10 for keypoints). Populates `stats`.

---

## Params

Evaluation parameters. Created automatically by `COCOeval`, but can be modified before calling `evaluate()`.

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.params.max_dets = [1, 10, 100]
ev.params.area_rng = [[0, 10000000000]]
ev.params.area_rng_lbl = ["all"]
```

### Constructor

```python
Params(iou_type="bbox")
```

### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `iou_type` | `str` | `"bbox"` | Evaluation type |
| `img_ids` | `list[int]` | `[]` | Image IDs to evaluate (empty = all) |
| `cat_ids` | `list[int]` | `[]` | Category IDs to evaluate (empty = all) |
| `iou_thrs` | `list[float]` | `[0.5, 0.55, ..., 0.95]` | IoU thresholds |
| `rec_thrs` | `list[float]` | `[0.0, 0.01, ..., 1.0]` | Recall thresholds for interpolation |
| `max_dets` | `list[int]` | `[1, 10, 100]` | Max detections per image |
| `area_rng` | `list[[float, float]]` | 4 ranges | Area ranges for small/medium/large |
| `area_rng_lbl` | `list[str]` | `["all", "small", "medium", "large"]` | Labels for area ranges |
| `use_cats` | `bool` | `True` | Category-aware evaluation (`False` for class-agnostic) |
| `kpt_oks_sigmas` | `list[float]` | COCO keypoint sigmas | Per-keypoint OKS sigmas |

---

## mask Submodule

Low-level mask operations. Access via `coco_rs.mask`.

```python
from coco_rs import mask
```

### Functions

#### `encode(mask, h, w)`

Encode a binary mask (numpy uint8 array of shape h x w) to RLE.

```python
import numpy as np
from coco_rs import mask

m = np.zeros((100, 100), dtype=np.uint8)
m[10:50, 20:80] = 1
rle = mask.encode(m, 100, 100)
```

#### `decode(rle)`

Decode an RLE dict to a binary mask. Returns a numpy uint8 array of shape (h, w).

```python
m = mask.decode(rle)
```

#### `area(rle)`

Compute the area (number of foreground pixels) of an RLE mask.

```python
a = mask.area(rle)
```

#### `to_bbox(rle)`

Convert an RLE mask to a bounding box `[x, y, w, h]`.

```python
bbox = mask.to_bbox(rle)
```

#### `merge(rles, intersect=False)`

Merge multiple RLE masks. Union by default, intersection if `intersect=True`.

```python
merged = mask.merge([rle1, rle2])
intersected = mask.merge([rle1, rle2], intersect=True)
```

#### `iou(dt, gt, iscrowd)`

Compute pairwise IoU between two lists of RLE masks. Returns a 2D list of shape (len(dt), len(gt)).

```python
ious = mask.iou(dt_rles, gt_rles, [False] * len(gt_rles))
```

#### `bbox_iou(dt, gt, iscrowd)`

Compute pairwise IoU between two lists of bounding boxes `[x, y, w, h]`.

```python
ious = mask.bbox_iou(dt_boxes, gt_boxes, [False] * len(gt_boxes))
```

#### `fr_poly(xy, h, w)`

Rasterize a polygon to an RLE mask. `xy` is a flat list of [x1, y1, x2, y2, ...] coordinates.

```python
rle = mask.fr_poly([10, 10, 50, 10, 50, 50, 10, 50], 100, 100)
```

#### `fr_bbox(bb, h, w)`

Convert a bounding box `[x, y, w, h]` to an RLE mask.

```python
rle = mask.fr_bbox([10, 10, 40, 40], 100, 100)
```

#### `rle_to_string(rle)`

Encode an RLE to its LEB128 string representation.

```python
s = mask.rle_to_string(rle)
```

#### `rle_from_string(s, h, w)`

Decode an LEB128 string to an RLE dict.

```python
rle = mask.rle_from_string(s, 100, 100)
```

---

## init_as_pycocotools()

Patch `sys.modules` so that `from pycocotools.coco import COCO` and similar imports transparently use coco-rust. Call once at startup before any pycocotools imports.

```python
from coco_rs import init_as_pycocotools
init_as_pycocotools()
```

This registers the following module aliases:

| Import path | Resolves to |
|-------------|-------------|
| `pycocotools` | `coco_rs` |
| `pycocotools.coco` | `coco_rs` |
| `pycocotools.cocoeval` | `coco_rs` |
| `pycocotools.mask` | `coco_rs.mask` |
