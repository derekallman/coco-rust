# Mask Operations

The `mask` module provides low-level operations on Run-Length Encoded (RLE) binary masks.

## What is RLE?

Run-Length Encoding stores a binary mask as alternating runs of 0s and 1s. Instead of storing every pixel, you store the length of each run. For example, a 1x10 mask `[0,0,0,1,1,1,1,0,0,0]` becomes runs `[3, 4, 3]` (3 zeros, 4 ones, 3 zeros).

COCO uses column-major order (top to bottom, then left to right) and a compact LEB128 string encoding for the `counts` field.

An RLE dict looks like:

```python
{"counts": "oY`0e0Qd04M2N1O1N2O0O2O001N1O1N2O1N1O100O1O001O1N2O...", "size": [480, 640]}
```

## Encoding and decoding

=== "Python"

    ```python
    import numpy as np
    from coco_rs import mask

    # Create a binary mask
    m = np.zeros((100, 100), dtype=np.uint8)
    m[10:50, 20:80] = 1

    # Encode to RLE
    rle = mask.encode(m, 100, 100)
    print(rle["size"])    # [100, 100]
    print(type(rle["counts"]))  # <class 'str'>

    # Decode back to a mask
    decoded = mask.decode(rle)
    assert decoded.shape == (100, 100)
    assert np.array_equal(m, decoded)
    ```

=== "Rust"

    ```rust
    use coco_rs::mask;

    // Create a binary mask (column-major order)
    let mut pixels = vec![0u8; 100 * 100];
    for x in 20..80 {
        for y in 10..50 {
            pixels[y + 100 * x] = 1;  // Column-major: index = y + h * x
        }
    }

    let rle = mask::encode(&pixels, 100, 100);
    let decoded = mask::decode(&rle);
    assert_eq!(pixels, decoded);
    ```

## Area and bounding box

=== "Python"

    ```python
    # Area (number of foreground pixels)
    a = mask.area(rle)
    print(f"Area: {a} pixels")

    # Bounding box [x, y, width, height]
    bbox = mask.to_bbox(rle)
    print(f"Bbox: {bbox}")
    ```

=== "Rust"

    ```rust
    let a = mask::area(&rle);
    println!("Area: {a} pixels");

    let bbox = mask::to_bbox(&rle);
    println!("Bbox: {:?}", bbox);  // [x, y, w, h]
    ```

## Merging masks

Combine multiple masks with union (default) or intersection:

=== "Python"

    ```python
    merged = mask.merge([rle1, rle2])                     # Union
    intersected = mask.merge([rle1, rle2], intersect=True) # Intersection
    ```

=== "Rust"

    ```rust
    let merged = mask::merge(&[rle1.clone(), rle2.clone()], false);  // Union
    let intersected = mask::merge(&[rle1, rle2], true);              // Intersection
    ```

## Computing IoU

Compute pairwise IoU between two lists of masks:

=== "Python"

    ```python
    # Mask IoU — returns a 2D list of shape (len(dt), len(gt))
    ious = mask.iou(dt_rles, gt_rles, [False] * len(gt_rles))
    print(f"IoU between dt[0] and gt[0]: {ious[0][0]:.3f}")

    # Bbox IoU — same interface, but with [x, y, w, h] lists
    ious = mask.bbox_iou(dt_boxes, gt_boxes, [False] * len(gt_boxes))
    ```

=== "Rust"

    ```rust
    let ious = mask::iou(&dt_rles, &gt_rles, &vec![false; gt_rles.len()]);
    println!("IoU between dt[0] and gt[0]: {:.3}", ious[0][0]);

    let ious = mask::bbox_iou(&dt_boxes, &gt_boxes, &vec![false; gt_boxes.len()]);
    ```

The `iscrowd` parameter controls how IoU is computed for crowd annotations. When `iscrowd[j]` is `true`, the IoU for GT `j` uses `intersection / area(dt)` instead of `intersection / union`, which prevents penalizing detections that only partially cover a crowd region.

## Creating masks from geometry

=== "Python"

    ```python
    # From a polygon (flat list of [x1, y1, x2, y2, ...])
    rle = mask.fr_poly([10, 10, 50, 10, 50, 50, 10, 50], 100, 100)

    # From a bounding box [x, y, w, h]
    rle = mask.fr_bbox([10, 10, 40, 40], 100, 100)
    ```

=== "Rust"

    ```rust
    // From a polygon
    let rle = mask::fr_poly(&[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0], 100, 100);

    // From a bounding box
    let rle = mask::fr_bbox(&[10.0, 10.0, 40.0, 40.0], 100, 100);
    ```

## RLE string encoding

Convert between RLE structs and the compact LEB128 string format used in COCO JSON files:

=== "Python"

    ```python
    # RLE to string
    s = mask.rle_to_string(rle)

    # String back to RLE
    rle = mask.rle_from_string(s, 100, 100)
    ```

=== "Rust"

    ```rust
    let s = mask::rle_to_string(&rle);
    let rle = mask::rle_from_string(&s, 100, 100).unwrap();
    ```

## Converting annotations

The `COCO` class provides convenience methods to convert annotations directly:

=== "Python"

    ```python
    from coco_rs import COCO

    coco = COCO("instances_val2017.json")
    ann = coco.load_anns([101])[0]

    rle = coco.ann_to_rle(ann)   # Annotation → RLE dict
    m = coco.ann_to_mask(ann)    # Annotation → numpy array (h, w)
    ```

=== "Rust"

    ```rust
    let ann = &coco.load_anns(&[101])[0];
    let rle = coco.ann_to_rle(ann);    // Option<Rle>
    let m = coco.ann_to_mask(ann);     // Option<Vec<u8>>
    ```

These handle all segmentation formats (polygon, uncompressed RLE, compressed RLE) and look up the image dimensions automatically.
