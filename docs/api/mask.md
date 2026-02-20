# mask

Low-level mask operations on Run-Length Encoded (RLE) binary masks.

=== "Python"

    ```python
    from coco_rs import mask
    ```

=== "Rust"

    ```rust
    use coco_rs::mask;
    ```

For background on RLE and usage patterns, see the [Mask Operations](../guide/masks.md) guide.

---

## Functions

### `encode`

Encode a binary mask to RLE.

=== "Python"

    ```python
    encode(mask: numpy.ndarray, h: int, w: int) -> dict
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `mask` | `numpy.ndarray` | Binary mask, shape (h, w), dtype `uint8` |
    | `h` | `int` | Height |
    | `w` | `int` | Width |

    **Returns:** `dict` — RLE dict with `"counts"` (str) and `"size"` ([h, w]).

    ```python
    import numpy as np
    from coco_rs import mask

    m = np.zeros((100, 100), dtype=np.uint8)
    m[10:50, 20:80] = 1
    rle = mask.encode(m, 100, 100)
    ```

=== "Rust"

    ```rust
    fn encode(mask: &[u8], h: u32, w: u32) -> Rle
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `mask` | `&[u8]` | Binary mask in column-major order (h * w pixels) |
    | `h` | `u32` | Height |
    | `w` | `u32` | Width |

    **Returns:** `Rle`

    ```rust
    let rle = mask::encode(&pixels, 100, 100);
    ```

---

### `decode`

Decode an RLE to a binary mask.

=== "Python"

    ```python
    decode(rle: dict) -> numpy.ndarray
    ```

    **Returns:** `numpy.ndarray` — Binary mask, shape (h, w), dtype `uint8`.

    ```python
    m = mask.decode(rle)
    ```

=== "Rust"

    ```rust
    fn decode(rle: &Rle) -> Vec<u8>
    ```

    **Returns:** `Vec<u8>` — Flat binary mask in column-major order.

    ```rust
    let pixels = mask::decode(&rle);
    ```

---

### `area`

Compute the area (number of foreground pixels) of an RLE mask.

=== "Python"

    ```python
    area(rle: dict) -> int
    ```

    ```python
    a = mask.area(rle)
    ```

=== "Rust"

    ```rust
    fn area(rle: &Rle) -> u64
    ```

    ```rust
    let a = mask::area(&rle);
    ```

---

### `to_bbox`

Convert an RLE mask to a bounding box.

=== "Python"

    ```python
    to_bbox(rle: dict) -> list[float]
    ```

    **Returns:** `[x, y, width, height]`

    ```python
    bbox = mask.to_bbox(rle)
    ```

    !!! note "camelCase alias"
        Also available as `toBbox()`.

=== "Rust"

    ```rust
    fn to_bbox(rle: &Rle) -> [f64; 4]
    ```

    **Returns:** `[x, y, width, height]`

    ```rust
    let bbox = mask::to_bbox(&rle);
    ```

---

### `merge`

Merge multiple RLE masks. Union by default, intersection if `intersect=True`.

=== "Python"

    ```python
    merge(rles: list[dict], intersect: bool = False) -> dict
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `rles` | `list[dict]` | | List of RLE dicts to merge |
    | `intersect` | `bool` | `False` | If `True`, compute intersection instead of union |

    ```python
    merged = mask.merge([rle1, rle2])
    intersected = mask.merge([rle1, rle2], intersect=True)
    ```

=== "Rust"

    ```rust
    fn merge(rles: &[Rle], intersect: bool) -> Rle
    ```

    ```rust
    let merged = mask::merge(&[rle1, rle2], false);
    let intersected = mask::merge(&[rle1, rle2], true);
    ```

---

### `iou`

Compute pairwise IoU between two lists of RLE masks.

=== "Python"

    ```python
    iou(dt: list[dict], gt: list[dict], iscrowd: list[bool]) -> list[list[float]]
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `dt` | `list[dict]` | Detection RLE dicts |
    | `gt` | `list[dict]` | Ground truth RLE dicts |
    | `iscrowd` | `list[bool]` | Per-GT crowd flag |

    **Returns:** 2D list of shape `(len(dt), len(gt))`.

    ```python
    ious = mask.iou(dt_rles, gt_rles, [False] * len(gt_rles))
    ```

=== "Rust"

    ```rust
    fn iou(dt: &[Rle], gt: &[Rle], iscrowd: &[bool]) -> Vec<Vec<f64>>
    ```

    **Returns:** `Vec<Vec<f64>>` of shape D x G.

    ```rust
    let ious = mask::iou(&dt_rles, &gt_rles, &vec![false; gt_rles.len()]);
    ```

When `iscrowd[j]` is `true`, uses `intersection / area(dt)` instead of standard IoU for GT `j`.

---

### `bbox_iou`

Compute pairwise IoU between two lists of bounding boxes.

=== "Python"

    ```python
    bbox_iou(dt: list[list[float]], gt: list[list[float]], iscrowd: list[bool]) -> list[list[float]]
    ```

    Bounding boxes are `[x, y, width, height]`.

    ```python
    ious = mask.bbox_iou(dt_boxes, gt_boxes, [False] * len(gt_boxes))
    ```

    !!! note "camelCase alias"
        Also available as `bboxIou()`.

=== "Rust"

    ```rust
    fn bbox_iou(dt: &[[f64; 4]], gt: &[[f64; 4]], iscrowd: &[bool]) -> Vec<Vec<f64>>
    ```

    ```rust
    let ious = mask::bbox_iou(&dt_boxes, &gt_boxes, &vec![false; gt_boxes.len()]);
    ```

---

### `fr_poly`

Rasterize a polygon to an RLE mask.

=== "Python"

    ```python
    fr_poly(xy: list[float], h: int, w: int) -> dict
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `xy` | `list[float]` | Flat list of coordinates `[x1, y1, x2, y2, ...]` |
    | `h` | `int` | Image height |
    | `w` | `int` | Image width |

    ```python
    rle = mask.fr_poly([10, 10, 50, 10, 50, 50, 10, 50], 100, 100)
    ```

    !!! note "camelCase alias"
        Also available as `frPoly()`.

=== "Rust"

    ```rust
    fn fr_poly(xy: &[f64], h: u32, w: u32) -> Rle
    ```

    ```rust
    let rle = mask::fr_poly(&[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0], 100, 100);
    ```

---

### `fr_bbox`

Convert a bounding box to an RLE mask.

=== "Python"

    ```python
    fr_bbox(bb: list[float], h: int, w: int) -> dict
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `bb` | `list[float]` | Bounding box `[x, y, width, height]` |
    | `h` | `int` | Image height |
    | `w` | `int` | Image width |

    ```python
    rle = mask.fr_bbox([10, 10, 40, 40], 100, 100)
    ```

    !!! note "camelCase alias"
        Also available as `frBbox()`.

=== "Rust"

    ```rust
    fn fr_bbox(bb: &[f64; 4], h: u32, w: u32) -> Rle
    ```

    ```rust
    let rle = mask::fr_bbox(&[10.0, 10.0, 40.0, 40.0], 100, 100);
    ```

---

### `rle_to_string`

Encode an RLE to its compact LEB128 string representation.

=== "Python"

    ```python
    rle_to_string(rle: dict) -> str
    ```

    ```python
    s = mask.rle_to_string(rle)
    ```

    !!! note "camelCase alias"
        Also available as `rleToString()`.

=== "Rust"

    ```rust
    fn rle_to_string(rle: &Rle) -> String
    ```

    ```rust
    let s = mask::rle_to_string(&rle);
    ```

---

### `rle_from_string`

Decode an LEB128 string to an RLE.

=== "Python"

    ```python
    rle_from_string(s: str, h: int, w: int) -> dict
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `s` | `str` | LEB128-encoded RLE string |
    | `h` | `int` | Image height |
    | `w` | `int` | Image width |

    ```python
    rle = mask.rle_from_string(s, 100, 100)
    ```

    !!! note "camelCase alias"
        Also available as `rleFromString()`.

=== "Rust"

    ```rust
    fn rle_from_string(s: &str, h: u32, w: u32) -> Result<Rle, String>
    ```

    ```rust
    let rle = mask::rle_from_string(&s, 100, 100).unwrap();
    ```
