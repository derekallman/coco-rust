# COCO

Load and query COCO-format datasets.

=== "Python"

    ```python
    from coco_rs import COCO

    coco = COCO("instances_val2017.json")
    ```

=== "Rust"

    ```rust
    use coco_rs::COCO;
    use std::path::Path;

    let coco = COCO::new(Path::new("instances_val2017.json"))?;
    ```

---

## Constructor

=== "Python"

    ```python
    COCO(annotation_file: str | None = None)
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `annotation_file` | <code>str &#124; None</code> | `None` | Path to a COCO JSON annotation file. `None` creates an empty instance. |

=== "Rust"

    ```rust
    COCO::new(annotation_file: &Path) -> Result<Self, Box<dyn Error>>
    COCO::from_dataset(dataset: Dataset) -> Self
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `annotation_file` | `&Path` | Path to a COCO JSON annotation file |
    | `dataset` | `Dataset` | A pre-built `Dataset` struct (for `from_dataset`) |

---

## Properties

### `dataset`

The full dataset with `images`, `annotations`, and `categories`.

=== "Python"

    ```python
    coco = COCO("instances_val2017.json")
    print(len(coco.dataset["images"]))       # 5000
    print(len(coco.dataset["annotations"]))  # 36781
    ```

=== "Rust"

    ```rust
    let coco = COCO::new(Path::new("instances_val2017.json"))?;
    println!("{}", coco.dataset.images.len());       // 5000
    println!("{}", coco.dataset.annotations.len());  // 36781
    ```

---

## Methods

### `get_ann_ids`

Get annotation IDs matching the given filters. All filters are ANDed together.

=== "Python"

    ```python
    get_ann_ids(
        img_ids: list[int] = [],
        cat_ids: list[int] = [],
        area_rng: list[float] | None = None,
        iscrowd: bool | None = None,
    ) -> list[int]
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `img_ids` | `list[int]` | `[]` | Filter by image IDs (empty = all) |
    | `cat_ids` | `list[int]` | `[]` | Filter by category IDs (empty = all) |
    | `area_rng` | <code>list[float] &#124; None</code> | `None` | Filter by area range `[min, max]` |
    | `iscrowd` | <code>bool &#124; None</code> | `None` | Filter by crowd flag |

    ```python
    ann_ids = coco.get_ann_ids(img_ids=[42], cat_ids=[1])
    ```

    !!! note "camelCase alias"
        Also available as `getAnnIds()`.

=== "Rust"

    ```rust
    fn get_ann_ids(
        &self,
        img_ids: &[u64],
        cat_ids: &[u64],
        area_rng: Option<[f64; 2]>,
        is_crowd: Option<bool>,
    ) -> Vec<u64>
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `img_ids` | `&[u64]` | Filter by image IDs (empty = all) |
    | `cat_ids` | `&[u64]` | Filter by category IDs (empty = all) |
    | `area_rng` | `Option<[f64; 2]>` | Filter by area range `[min, max]` |
    | `is_crowd` | `Option<bool>` | Filter by crowd flag |

    ```rust
    let ann_ids = coco.get_ann_ids(&[42], &[1], None, None);
    ```

---

### `get_cat_ids`

Get category IDs matching the given filters.

=== "Python"

    ```python
    get_cat_ids(
        cat_nms: list[str] = [],
        sup_nms: list[str] = [],
        cat_ids: list[int] = [],
    ) -> list[int]
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `cat_nms` | `list[str]` | `[]` | Filter by category names |
    | `sup_nms` | `list[str]` | `[]` | Filter by supercategory names |
    | `cat_ids` | `list[int]` | `[]` | Filter by category IDs |

    ```python
    cat_ids = coco.get_cat_ids(cat_nms=["person", "dog"])
    ```

    !!! note "camelCase alias"
        Also available as `getCatIds()`.

=== "Rust"

    ```rust
    fn get_cat_ids(&self, cat_nms: &[&str], sup_nms: &[&str], cat_ids: &[u64]) -> Vec<u64>
    ```

    ```rust
    let cat_ids = coco.get_cat_ids(&["person", "dog"], &[], &[]);
    ```

---

### `get_img_ids`

Get image IDs matching the given filters.

=== "Python"

    ```python
    get_img_ids(
        img_ids: list[int] = [],
        cat_ids: list[int] = [],
    ) -> list[int]
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `img_ids` | `list[int]` | `[]` | Filter by image IDs |
    | `cat_ids` | `list[int]` | `[]` | Filter by category IDs (images containing these categories) |

    ```python
    img_ids = coco.get_img_ids(cat_ids=[1])
    ```

    !!! note "camelCase alias"
        Also available as `getImgIds()`.

=== "Rust"

    ```rust
    fn get_img_ids(&self, img_ids: &[u64], cat_ids: &[u64]) -> Vec<u64>
    ```

    ```rust
    let img_ids = coco.get_img_ids(&[], &[1]);
    ```

---

### `load_anns`

Load annotations by their IDs.

=== "Python"

    ```python
    load_anns(ids: list[int]) -> list[dict]
    ```

    Returns annotation dicts with keys like `id`, `image_id`, `category_id`, `bbox`, `area`, `segmentation`, `iscrowd`.

    ```python
    anns = coco.load_anns([101, 102, 103])
    print(anns[0]["bbox"])  # [x, y, width, height]
    ```

    !!! note "camelCase alias"
        Also available as `loadAnns()`.

=== "Rust"

    ```rust
    fn load_anns(&self, ids: &[u64]) -> Vec<&Annotation>
    ```

    Returns references to `Annotation` structs.

    ```rust
    let anns = coco.load_anns(&[101, 102, 103]);
    println!("{:?}", anns[0].bbox);  // [x, y, width, height]
    ```

---

### `load_cats`

Load categories by their IDs.

=== "Python"

    ```python
    load_cats(ids: list[int]) -> list[dict]
    ```

    Returns category dicts with keys `id`, `name`, `supercategory`.

    ```python
    cats = coco.load_cats([1, 2, 3])
    print(cats[0]["name"])  # "person"
    ```

    !!! note "camelCase alias"
        Also available as `loadCats()`.

=== "Rust"

    ```rust
    fn load_cats(&self, ids: &[u64]) -> Vec<&Category>
    ```

    ```rust
    let cats = coco.load_cats(&[1, 2, 3]);
    println!("{}", cats[0].name);  // "person"
    ```

---

### `load_imgs`

Load images by their IDs.

=== "Python"

    ```python
    load_imgs(ids: list[int]) -> list[dict]
    ```

    Returns image dicts with keys like `id`, `file_name`, `width`, `height`.

    ```python
    imgs = coco.load_imgs([42])
    print(f"{imgs[0]['width']}x{imgs[0]['height']}")
    ```

    !!! note "camelCase alias"
        Also available as `loadImgs()`.

=== "Rust"

    ```rust
    fn load_imgs(&self, ids: &[u64]) -> Vec<&Image>
    ```

    ```rust
    let imgs = coco.load_imgs(&[42]);
    println!("{}x{}", imgs[0].width, imgs[0].height);
    ```

---

### `load_res`

Load detection results from a JSON file. Returns a new `COCO` object containing the detections, with images and categories copied from the ground truth.

=== "Python"

    ```python
    load_res(res_file: str) -> COCO
    ```

    The results file should be a JSON array of detection dicts:

    ```json
    [
      {"image_id": 42, "category_id": 1, "bbox": [10, 20, 30, 40], "score": 0.95},
      ...
    ]
    ```

    ```python
    coco_dt = coco_gt.load_res("detections.json")
    ```

    !!! note "camelCase alias"
        Also available as `loadRes()`.

=== "Rust"

    ```rust
    fn load_res(&self, res_file: &Path) -> Result<COCO, Box<dyn Error>>
    ```

    ```rust
    let coco_dt = coco_gt.load_res(Path::new("detections.json"))?;
    ```

!!! tip
    `load_res` automatically computes missing fields: `area` from bounding boxes or segmentation masks, and polygon segmentations from bbox results. This matches pycocotools behavior.

---

### `ann_to_rle`

Convert an annotation to RLE format.

=== "Python"

    ```python
    ann_to_rle(ann: dict) -> dict
    ```

    Returns an RLE dict with `"counts"` (str) and `"size"` ([h, w]).

    ```python
    ann = coco.load_anns([101])[0]
    rle = coco.ann_to_rle(ann)
    print(rle.keys())  # dict_keys(['counts', 'size'])
    ```

    !!! note "camelCase alias"
        Also available as `annToRLE()`.

=== "Rust"

    ```rust
    fn ann_to_rle(&self, ann: &Annotation) -> Option<Rle>
    ```

    Returns an `Rle` struct with `h`, `w`, and `counts` fields.

    ```rust
    let ann = &coco.load_anns(&[101])[0];
    if let Some(rle) = coco.ann_to_rle(ann) {
        println!("{}x{}", rle.h, rle.w);
    }
    ```

---

### `ann_to_mask`

Convert an annotation to a binary mask.

=== "Python"

    ```python
    ann_to_mask(ann: dict) -> numpy.ndarray
    ```

    Returns a binary mask of shape (h, w), dtype `uint8`.

    ```python
    ann = coco.load_anns([101])[0]
    mask = coco.ann_to_mask(ann)
    print(mask.shape)  # (height, width)
    ```

    !!! note "camelCase alias"
        Also available as `annToMask()`.

=== "Rust"

    ```rust
    fn ann_to_mask(&self, ann: &Annotation) -> Option<Vec<u8>>
    ```

    Returns a flat `Vec<u8>` in column-major order (h * w pixels).

    ```rust
    let ann = &coco.load_anns(&[101])[0];
    if let Some(mask) = coco.ann_to_mask(ann) {
        println!("pixels: {}", mask.len());
    }
    ```
