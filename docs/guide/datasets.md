# Dataset Operations

hotcoco can do more than evaluate — it can reshape your datasets before evaluation starts.

All operations return a new `COCO` object and leave the original unchanged. They compose naturally,
so you can chain filter → split → sample in a single expression.

!!! tip
    Run `coco.stats()` first to understand your dataset before reshaping it. See the
    [`stats` API reference](../api/coco.md#stats) for the full return structure.

---

## filter

Subset a dataset to a specific set of categories, images, or annotation sizes.
Returns a new `COCO` with matching annotations and — by default — only the images
that have at least one match.

```python
from hotcoco import COCO

coco = COCO("instances_val2017.json")

# Keep only "person" annotations
person_id = coco.get_cat_ids(cat_nms=["person"])[0]
people = coco.filter(cat_ids=[person_id])

print(len(people.dataset["images"]))       # 2693
print(len(people.dataset["annotations"])) # 10777
```

Pass `drop_empty_images=False` to keep all images even if they have no matching
annotations — useful when you need consistent image IDs across filtered splits.

```python
# Same annotations, all 5000 images preserved
people_all_imgs = coco.filter(cat_ids=[person_id], drop_empty_images=False)
```

Filter by annotation area to focus on a size range:

```python
# Medium-to-large objects only (32² – 96² px²)
medium = coco.filter(area_rng=[1024.0, 9216.0])
```

Filters compose — all criteria are ANDed:

```python
medium_people = coco.filter(cat_ids=[person_id], area_rng=[1024.0, 9216.0])
```

---

## split

Split a dataset into train/val (or train/val/test) subsets. Images are shuffled
deterministically and partitioned by fraction. Annotations follow their images;
all splits share the full category list.

```python
# 80/20 train/val split
train, val = coco.split(val_frac=0.2, seed=42)

print(len(train.dataset["images"]))  # 4000
print(len(val.dataset["images"]))    # 1000
```

Add a test set with a second fraction:

```python
train, val, test = coco.split(val_frac=0.15, test_frac=0.15, seed=42)
# train ~70%, val ~15%, test ~15%
```

The same `seed` always produces the same split — important for reproducibility
across experiments:

```python
# These are identical
train_a, val_a = coco.split(val_frac=0.2, seed=42)
train_b, val_b = coco.split(val_frac=0.2, seed=42)
```

A typical eval workflow — filter first, then split:

```python
people = coco.filter(cat_ids=[person_id])
train, val = people.split(val_frac=0.2, seed=42)
```

---

## sample

Draw a random subset of images (with their annotations). Useful for quick
iteration during development without running full-dataset evaluation.

```python
# Sample 500 images
subset = coco.sample(n=500, seed=0)

# Or by fraction
subset = coco.sample(frac=0.1, seed=0)
```

Like `split`, the sample is deterministic for the same seed:

```python
# Always the same 500 images
a = coco.sample(n=500, seed=0)
b = coco.sample(n=500, seed=0)
```

---

## merge

Combine multiple annotation files into one. Common when annotations arrive in
separate batches or from separate labeling jobs.

All datasets must share the same category taxonomy (same names and supercategories).
Image and annotation IDs are remapped automatically to be globally unique.

```python
batch1 = COCO("batch1.json")
batch2 = COCO("batch2.json")

combined = COCO.merge([batch1, batch2])

print(len(combined.dataset["images"]))
# len(batch1.images) + len(batch2.images)
```

Merging a dataset with itself doubles the image and annotation count — useful
for stress-testing:

```python
doubled = COCO.merge([coco, coco])
```

`merge` raises `ValueError` if the datasets have different category sets:

```python
# Raises ValueError: category 'horse' not found in first dataset
COCO.merge([coco_animals, coco_vehicles])
```

---

## save

Write any `COCO` object back to a JSON file. The output format is
standard COCO JSON, readable by any tool that accepts COCO annotations.

```python
merged = COCO.merge([batch1, batch2])
merged.save("combined.json")
```

`save` works at any point in a pipeline:

```python
coco.filter(cat_ids=[person_id]).sample(n=1000, seed=0).save("person_sample.json")
```

---

## CLI

All four operations are available as `coco` subcommands — no Python required
beyond the initial install. See the [CLI reference](../cli.md) for full flag
documentation.

```bash
coco filter instances_val2017.json --cat-ids 1 -o person.json
coco split  person.json --val-frac 0.2 -o splits/person
coco sample person.json --n 500 --seed 0 -o person_sample.json
coco merge  batch1.json batch2.json -o combined.json
```
