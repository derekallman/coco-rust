# plot

```python
from hotcoco.plot import pr_curve, confusion_matrix, top_confusions, per_category_ap, tide_errors
```

Requires `pip install hotcoco[plot]` (matplotlib >= 3.5).

All functions share these common parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `styled` | `bool` | Apply hotcoco visual style. Default `True`. Set `False` for plain matplotlib. |
| `ax` | <code>Axes &#124; None</code> | Draw on an existing axes. If `None`, creates a new figure. |
| `save_path` | <code>str &#124; Path &#124; None</code> | Save figure to this path (150 DPI). |

All functions return `(Figure, Axes)`.

---

## `pr_curve`

```python
pr_curve(
    coco_eval, *,
    iou_thrs=None, cat_id=None, cat_ids=None,
    iou_thr=None, top_n=10, area_rng="all", max_det=None,
    styled=True, ax=None, save_path=None,
)
```

Plot precision-recall curves. Three modes:

1. **IoU sweep** (default) — one line per IoU threshold, mean precision across categories.
2. **Single category** — set `cat_id` to plot one category.
3. **Multi-category** — set `cat_ids` or `iou_thr` to compare categories at a fixed IoU.

| Parameter | Type | Description |
|-----------|------|-------------|
| `coco_eval` | `COCOeval` | Must have `run()` called first. |
| `iou_thrs` | <code>list[float] &#124; None</code> | IoU thresholds to plot (mode 1). |
| `cat_id` | <code>int &#124; None</code> | Single category to plot (mode 2). |
| `cat_ids` | <code>list[int] &#124; None</code> | Categories to compare (mode 3). |
| `iou_thr` | <code>float &#124; None</code> | Fixed IoU for modes 2/3. Default 0.50. |
| `top_n` | `int` | Top N categories by AP for mode 3. Default 10. |
| `area_rng` | `str` | Area range: `"all"`, `"small"`, `"medium"`, `"large"`. |
| `max_det` | <code>int &#124; None</code> | Max detections index. Default: last in params. |

---

## `confusion_matrix`

```python
confusion_matrix(
    cm_dict, *,
    normalize=True, top_n=None,
    group_by=None, cat_groups=None,
    styled=True, ax=None, save_path=None,
)
```

Plot a confusion matrix heatmap.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cm_dict` | `dict` | Output of `coco_eval.confusion_matrix()`. |
| `normalize` | `bool` | Row-normalize values. Default `True`. |
| `top_n` | <code>int &#124; None</code> | Show only top N categories by confusion mass. Auto-set to 25 when K > 30. |
| `group_by` | <code>str &#124; None</code> | `"supercategory"` to aggregate by group. |
| `cat_groups` | <code>dict &#124; None</code> | Group name → list of category names. Required with `group_by`. |

---

## `top_confusions`

```python
top_confusions(
    cm_dict, *,
    top_n=20,
    styled=True, ax=None, save_path=None,
)
```

Plot the top N misclassifications as horizontal bars. Shows "ground truth → prediction" pairs sorted by count.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cm_dict` | `dict` | Output of `coco_eval.confusion_matrix()`. |
| `top_n` | `int` | Number of confusions to show. Default 20. |

---

## `per_category_ap`

```python
per_category_ap(
    results_dict, *,
    top_n=20, bottom_n=5,
    styled=True, ax=None, save_path=None,
)
```

Plot per-category AP as horizontal bars with a mean AP reference line.

| Parameter | Type | Description |
|-----------|------|-------------|
| `results_dict` | `dict` | Output of `coco_eval.results(per_class=True)`. |
| `top_n` | `int` | Top categories to show. Default 20. |
| `bottom_n` | `int` | Bottom categories to show. Default 5. |

---

## `tide_errors`

```python
tide_errors(
    tide_dict, *,
    styled=True, ax=None, save_path=None,
)
```

Plot TIDE error breakdown as horizontal bars.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tide_dict` | `dict` | Output of `coco_eval.tide_errors()`. |

---

## Color palette

The styled theme uses these constants, available for custom plots:

```python
from hotcoco.plot import SERIES_COLORS, CHROME, SEQUENTIAL
```

- `SERIES_COLORS` — 8 data series colors (warm slate, terracotta, teal, gold, plum, sage, ocean, copper)
- `CHROME` — non-data element colors (text, label, tick, grid, spine, background)
- `SEQUENTIAL` — 4-stop colormap for heatmaps (off-white → gold → terracotta → deep brown)
