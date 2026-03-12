# Plotting

hotcoco includes publication-quality plotting for evaluation results.
Install the optional dependency:

```bash
pip install hotcoco[plot]
```

All plot functions return `(Figure, Axes)` for further customization,
accept an optional `ax` to draw on existing axes, and accept `save_path`
to write directly to disk.

## Quick start

```python
import hotcoco
from hotcoco.plot import pr_curve, per_category_ap

gt = hotcoco.COCO("instances_val2017.json")
dt = gt.load_res("detections.json")
ev = hotcoco.COCOeval(gt, dt, "bbox")
ev.run()

fig, ax = pr_curve(ev, save_path="pr.png")
fig, ax = per_category_ap(ev.results(per_class=True), save_path="ap.png")
```

## Available plots

### Precision-recall curves

```python
from hotcoco.plot import pr_curve

# IoU sweep (default) — one line per IoU threshold
fig, ax = pr_curve(ev)

# Single category
fig, ax = pr_curve(ev, cat_id=1)

# Top 10 categories at IoU=0.50
fig, ax = pr_curve(ev, iou_thr=0.50, top_n=10)
```

### Per-category AP

```python
from hotcoco.plot import per_category_ap

results = ev.results(per_class=True)
fig, ax = per_category_ap(results)
```

Shows horizontal bars sorted by AP with a mean AP reference line.
When there are many categories, the top 20 and bottom 5 are shown
with a visual break.

### Confusion matrix

```python
from hotcoco.plot import confusion_matrix

fig, ax = confusion_matrix(ev.confusion_matrix())
```

For datasets with many categories (>30), the matrix auto-filters to
the 25 most confused categories. You can also aggregate by supercategory:

```python
# Build supercategory groups from the dataset
cats = gt.load_cats(gt.get_cat_ids())
groups = {}
for c in cats:
    groups.setdefault(c["supercategory"], []).append(c["name"])

fig, ax = confusion_matrix(ev.confusion_matrix(), group_by="supercategory", cat_groups=groups)
```

### Top confusions

```python
from hotcoco.plot import top_confusions

fig, ax = top_confusions(ev.confusion_matrix())
```

A bar chart of the most common misclassifications — more readable than
a full heatmap when you have many categories.

### TIDE error breakdown

```python
from hotcoco.plot import tide_errors

fig, ax = tide_errors(ev.tide_errors())
```

Shows the six TIDE error types (Cls, Loc, Both, Dupe, Bkg, Miss)
as horizontal bars with their delta-AP values.

## Bring your own style

Every function accepts `styled=False` to skip hotcoco's visual theme
and return a plain matplotlib figure. This is useful when you want to
apply your own rcParams, seaborn theme, or corporate style guide:

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
fig, ax = pr_curve(ev, styled=False)
```

## Composing plots

Pass an existing `ax` to draw on a subplot:

```python
import matplotlib.pyplot as plt
from hotcoco.plot import pr_curve, per_category_ap

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
pr_curve(ev, ax=ax1)
per_category_ap(ev.results(per_class=True), ax=ax2)
fig.savefig("dashboard.png", dpi=150)
```
