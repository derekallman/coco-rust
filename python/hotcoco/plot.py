"""Publication-quality plots for hotcoco evaluation results.

Requires matplotlib (install with ``pip install hotcoco[plot]``).
All functions accept an optional ``ax`` to draw on an existing axes,
and an optional ``save_path`` to save the figure to disk.

Pass ``styled=False`` to any function to get a plain matplotlib figure
with no custom colors, fonts, or styling — useful when applying your own
theme or rcParams.

Example
-------
::

    import hotcoco
    from hotcoco.plot import pr_curve, confusion_matrix

    gt = hotcoco.COCO("annotations.json")
    dt = gt.load_res("detections.json")
    ev = hotcoco.COCOeval(gt, dt, "bbox")
    ev.run()

    fig, ax = pr_curve(ev)
    fig, ax = confusion_matrix(ev.confusion_matrix())

    # Unstyled — bring your own theme
    fig, ax = pr_curve(ev, styled=False)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Palette (from SKILL.md)
# ---------------------------------------------------------------------------

SERIES_COLORS = [
    "#5C7080",  # warm slate
    "#C46B50",  # terracotta
    "#2B7A8C",  # deep teal
    "#C9943E",  # warm gold
    "#8A5A90",  # plum
    "#5B7F63",  # sage
    "#3A7CA5",  # ocean
    "#B07650",  # copper
]

CHROME = {
    "text": "#2C2420",
    "label": "#4A3F38",
    "tick": "#5E544C",
    "grid": "#E8E2DA",
    "spine": "#D4CCC2",
    "background": "#FAF7F4",
    "plot_bg": "#FDF9F6",
}

SEQUENTIAL = ["#FAF7F4", "#C9943E", "#C46B50", "#5C2E1E"]

# ---------------------------------------------------------------------------
# Lazy import helper
# ---------------------------------------------------------------------------

_MPL_ERROR = (
    "matplotlib is required for plotting. "
    "Install with: pip install hotcoco[plot]"
)


def _import_mpl():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        return matplotlib, plt, font_manager
    except ImportError:
        raise ImportError(_MPL_ERROR) from None


# ---------------------------------------------------------------------------
# Font registration
# ---------------------------------------------------------------------------

_FONT_FAMILY: list[str] | None = None


def _resolve_font_family() -> list[str]:
    global _FONT_FAMILY
    if _FONT_FAMILY is not None:
        return _FONT_FAMILY

    _, _, font_manager = _import_mpl()
    fonts_dir = Path(__file__).parent / "_fonts"
    for ttf in fonts_dir.glob("*.ttf"):
        try:
            font_manager.fontManager.addfont(str(ttf))
        except Exception:
            pass

    _FONT_FAMILY = ["Inter", "Helvetica Neue", "DejaVu Sans"]
    return _FONT_FAMILY


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _new_figure(figsize: tuple[float, float], ax=None, styled: bool = True):
    _, plt, _ = _import_mpl()
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    if styled:
        fig.set_facecolor(CHROME["background"])
        ax.set_facecolor(CHROME["plot_bg"])
    return fig, ax


def _apply_style(
    ax,
    title: str | None = None,
    subtitle: str | None = None,
    value_axis: str = "y",
):
    family = _resolve_font_family()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(CHROME["spine"])
        ax.spines[side].set_linewidth(0.75)

    if value_axis == "y":
        ax.yaxis.grid(True, color=CHROME["grid"], linewidth=0.8, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
    elif value_axis == "x":
        ax.xaxis.grid(True, color=CHROME["grid"], linewidth=0.8, zorder=0)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)

    ax.tick_params(
        axis="both",
        colors=CHROME["tick"],
        labelsize=9,
        which="both",
        direction="out",
        length=4,
        width=0.75,
    )
    # No tick marks on the categorical axis of bar charts
    if value_axis == "x":
        ax.tick_params(axis="y", length=0)
    elif value_axis == "y":
        ax.tick_params(axis="x", length=0)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(family)

    if title:
        ax.set_title(
            title,
            fontsize=12,
            fontweight=500,
            color=CHROME["text"],
            fontfamily=family,
            pad=16 if subtitle else 12,
        )
    if subtitle:
        ax.text(
            0.5, 1.005,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color=CHROME["tick"],
            fontfamily=family,
        )

    ax.xaxis.label.set_color(CHROME["label"])
    ax.xaxis.label.set_fontsize(11)
    ax.xaxis.label.set_fontfamily(family)
    ax.xaxis.labelpad = 8
    ax.yaxis.label.set_color(CHROME["label"])
    ax.yaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontfamily(family)
    ax.yaxis.labelpad = 8


def _save_and_return(fig, ax, save_path: str | Path | None):
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, facecolor=fig.get_facecolor())
    return fig, ax


def _annotate_bars(ax, bars, values, *, fmt: str = ".3f", fontsize: float = 9):
    """Add data labels at the end of horizontal bars (styled mode only)."""
    family = _resolve_font_family()
    max_val = max(values) if values else 1
    for bar, val in zip(bars, values):
        text = f"{val:{fmt}}" if isinstance(val, float) else str(val)
        ax.text(
            bar.get_width() + max_val * 0.01,
            bar.get_y() + bar.get_height() / 2,
            text, va="center", fontsize=fontsize,
            color=CHROME["label"], fontfamily=family,
        )


def _nearest_iou_idx(iou_thrs: list[float], target: float) -> int:
    """Return the index of the IoU threshold closest to *target*."""
    return min(range(len(iou_thrs)), key=lambda i: abs(iou_thrs[i] - target))


def _cat_name_lookup(coco_eval) -> dict[int, str]:
    """Build {cat_id: name} from coco_gt."""
    cat_ids = list(coco_eval.params.cat_ids)
    try:
        cats = coco_eval.coco_gt.load_cats(cat_ids)
        return {c["id"]: c["name"] for c in cats}
    except Exception:
        return {cid: str(cid) for cid in cat_ids}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pr_curve(
    coco_eval,
    *,
    iou_thrs: list[float] | None = None,
    cat_id: int | None = None,
    cat_ids: list[int] | None = None,
    iou_thr: float | None = None,
    top_n: int = 10,
    area_rng: str = "all",
    max_det: int | None = None,
    styled: bool = True,
    ax=None,
    save_path: str | Path | None = None,
) -> tuple:
    """Plot precision-recall curves from a COCOeval object.

    Three modes:

    1. **IoU sweep** (default): one line per IoU threshold, mean precision
       across all categories.
    2. **Single category**: ``cat_id=<id>`` plots one category at IoU=0.50.
    3. **Multi-category comparison**: ``cat_ids=[...]`` or ``top_n=N`` plots
       one line per category at a fixed IoU (``iou_thr``, default 0.50).

    Parameters
    ----------
    coco_eval : COCOeval
        Must have ``run()`` (or ``evaluate`` + ``accumulate``) called first.
    iou_thrs : list[float], optional
        IoU thresholds to plot (mode 1). Default: all thresholds in params.
    cat_id : int, optional
        Plot a single category (mode 2).
    cat_ids : list[int], optional
        Plot these categories as separate lines (mode 3).
    iou_thr : float, optional
        Fixed IoU threshold for modes 2/3. Default 0.50.
    top_n : int
        When using mode 3 without explicit ``cat_ids``, plot the top N
        categories by AP. Default 10.
    area_rng : str
        Area range label (``"all"``, ``"small"``, ``"medium"``, ``"large"``).
    max_det : int, optional
        Max detections index. Default: last entry in ``params.max_dets``.
    styled : bool
        Apply hotcoco visual style (default True). Set False for plain
        matplotlib defaults.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    save_path : str or Path, optional
        Save figure to this path.

    Returns
    -------
    (Figure, Axes)
    """
    _, plt, _ = _import_mpl()
    import numpy as np

    eval_data = coco_eval.eval
    if eval_data is None:
        raise ValueError(
            "No accumulated data. Call coco_eval.run() or "
            "coco_eval.evaluate() + coco_eval.accumulate() first."
        )

    precision = eval_data["precision"]  # (T, R, K, A, M)
    params = coco_eval.params

    # Resolve common indices
    all_iou_thrs = list(params.iou_thrs)
    area_labels = list(params.area_rng_lbl)
    max_dets_list = list(params.max_dets)
    all_cat_ids = list(params.cat_ids)

    a_idx = area_labels.index(area_rng) if area_rng in area_labels else 0
    m_idx = (
        max_dets_list.index(max_det)
        if max_det is not None and max_det in max_dets_list
        else len(max_dets_list) - 1
    )

    recall_pts = np.linspace(0.0, 1.0, precision.shape[1])
    name_map = _cat_name_lookup(coco_eval)

    # ---- Mode 3: multi-category comparison ----
    use_multi_cat = cat_ids is not None or (cat_id is None and iou_thr is not None)
    if use_multi_cat:
        fixed_iou = iou_thr if iou_thr is not None else 0.5
        t_idx = _nearest_iou_idx(all_iou_thrs, fixed_iou)

        if cat_ids is not None:
            plot_cat_ids = cat_ids
        else:
            # Pick top_n categories by mean precision (proxy for AP)
            cat_aps = []
            for k_i, cid in enumerate(all_cat_ids):
                p = precision[t_idx, :, k_i, a_idx, m_idx]
                valid = p[p >= 0]
                cat_aps.append((cid, float(np.mean(valid)) if len(valid) else 0.0))
            cat_aps.sort(key=lambda x: x[1], reverse=True)
            plot_cat_ids = [cid for cid, _ in cat_aps[:top_n]]

        fig, ax = _new_figure((6, 6), ax, styled=styled)

        for line_idx, cid in enumerate(plot_cat_ids):
            if cid not in all_cat_ids:
                continue
            k_idx = all_cat_ids.index(cid)
            prec = precision[t_idx, :, k_idx, a_idx, m_idx]
            prec = np.where(prec < 0, np.nan, prec)

            color = SERIES_COLORS[line_idx % len(SERIES_COLORS)] if styled else None
            label = name_map.get(cid, str(cid))

            ax.plot(recall_pts, prec, color=color, linewidth=1.5, label=label)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(frameon=not styled, fontsize=8, loc="lower left")
        if styled:
            _apply_style(
                ax, "Precision-Recall by Category",
                subtitle=f"IoU={all_iou_thrs[t_idx]:.2f}",
                value_axis="y",
            )
        return _save_and_return(fig, ax, save_path)

    # ---- Mode 2: single category ----
    if cat_id is not None:
        if cat_id not in all_cat_ids:
            raise ValueError(f"cat_id {cat_id} not in params.cat_ids")
        k_idx = all_cat_ids.index(cat_id)
        fixed_iou = iou_thr if iou_thr is not None else 0.5
        t_indices = [_nearest_iou_idx(all_iou_thrs, fixed_iou)]
    else:
        k_idx = None
        t_indices = (
            list(range(len(all_iou_thrs)))
            if iou_thrs is None
            else [i for i, t in enumerate(all_iou_thrs) if t in iou_thrs]
        )

    # ---- Mode 1: IoU sweep (or single-cat at one threshold) ----
    fig, ax = _new_figure((6, 6), ax, styled=styled)

    for line_idx, t_idx in enumerate(t_indices):
        if k_idx is not None:
            prec = precision[t_idx, :, k_idx, a_idx, m_idx]
        else:
            prec = precision[t_idx, :, :, a_idx, m_idx]
            valid = prec.copy()
            valid[valid < 0] = np.nan
            prec = np.nanmean(valid, axis=1)

        is_primary = line_idx == 0
        color = SERIES_COLORS[line_idx % len(SERIES_COLORS)] if styled else None
        lw = 2 if is_primary else 1
        label = f"IoU={all_iou_thrs[t_idx]:.2f}"

        ax.plot(recall_pts, prec, color=color, linewidth=lw, label=label)

        if is_primary:
            ax.fill_between(recall_pts, prec, alpha=0.15, color=color)
            f1 = 2 * prec * recall_pts / np.maximum(prec + recall_pts, 1e-8)
            best = np.argmax(f1)
            if prec[best] > 0:
                ax.plot(
                    recall_pts[best], prec[best], "o",
                    color=color, markersize=5, zorder=5,
                )
                font_kw = (
                    {"fontfamily": _resolve_font_family()}
                    if styled else {}
                )
                ax.annotate(
                    f"F1={f1[best]:.2f}",
                    (recall_pts[best], prec[best]),
                    textcoords="offset points",
                    xytext=(8, -8),
                    fontsize=8,
                    color=CHROME["text"] if styled else None,
                    **font_kw,
                )

    if cat_id is not None:
        cat_name = name_map.get(cat_id, str(cat_id))
        title = "Precision-Recall"
        subtitle = cat_name
    else:
        title = "Precision-Recall"
        subtitle = "mean over categories"

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(frameon=not styled, fontsize=9, loc="lower left")
    if styled:
        _apply_style(ax, title, subtitle=subtitle, value_axis="y")

    return _save_and_return(fig, ax, save_path)


def confusion_matrix(
    cm_dict: dict[str, Any],
    *,
    normalize: bool = True,
    top_n: int | None = None,
    group_by: str | None = None,
    cat_groups: dict[str, list[str]] | None = None,
    styled: bool = True,
    ax=None,
    save_path: str | Path | None = None,
) -> tuple:
    """Plot a confusion matrix heatmap.

    Parameters
    ----------
    cm_dict : dict
        Output of ``coco_eval.confusion_matrix()``.
    normalize : bool
        Use row-normalized values (default True).
    top_n : int, optional
        Show only the top N categories by off-diagonal confusion mass.
        Auto-set to 25 when K > 30 and no explicit value is given.
    group_by : str, optional
        ``"supercategory"`` to aggregate into COCO supercategory groups.
        Requires ``cat_groups`` mapping.
    cat_groups : dict[str, list[str]], optional
        Mapping of group name to list of category names for ``group_by``.
        For COCO supercategories, pass the result of building this from
        ``coco_gt.load_cats()``.
    styled : bool
        Apply hotcoco visual style (default True). Set False for plain
        matplotlib defaults.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    save_path : str or Path, optional
        Save figure to this path.

    Returns
    -------
    (Figure, Axes)
    """
    _, plt, _ = _import_mpl()
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    raw_matrix = np.asarray(cm_dict["matrix"], dtype=float)
    norm_matrix = np.asarray(cm_dict["normalized"], dtype=float)
    cat_names = list(cm_dict["cat_names"])
    K = len(cat_names)

    # ---- Supercategory grouping ----
    if group_by == "supercategory" and cat_groups is not None:
        name_to_idx = {n: i for i, n in enumerate(cat_names)}
        group_names = sorted(cat_groups.keys())
        G = len(group_names)

        # Aggregate raw counts into (G+1) x (G+1) — groups + BG
        grouped = np.zeros((G + 1, G + 1), dtype=float)
        group_idx_map = {}  # cat_name -> group index
        for gi, gname in enumerate(group_names):
            for cname in cat_groups[gname]:
                if cname in name_to_idx:
                    group_idx_map[cname] = gi

        for i, iname in enumerate(cat_names):
            gi = group_idx_map.get(iname)
            if gi is None:
                continue
            for j, jname in enumerate(cat_names):
                gj = group_idx_map.get(jname)
                if gj is None:
                    continue
                grouped[gi, gj] += raw_matrix[i, j]
            # GT cat -> BG column (FN)
            grouped[gi, G] += raw_matrix[i, K]
        # BG row -> pred columns (FP)
        for j, jname in enumerate(cat_names):
            gj = group_idx_map.get(jname)
            if gj is not None:
                grouped[G, gj] += raw_matrix[K, j]
        grouped[G, G] = raw_matrix[K, K]

        if normalize:
            row_sums = grouped.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            data = grouped / row_sums
        else:
            data = grouped

        labels = group_names + ["BG"]
    else:
        labels = cat_names + ["BG"]
        data = norm_matrix if normalize else raw_matrix

        # Auto top_n for large matrices
        if top_n is None and K > 30:
            top_n = 25

        if top_n is not None and top_n < K:
            # Rank categories by off-diagonal confusion mass
            row_total = data[:K, :K].sum(axis=1)
            col_total = data[:K, :K].sum(axis=0)[:K]
            diag = np.diag(data[:K, :K])
            confusion_mass = (row_total - diag) + (col_total - diag)
            top_indices = np.argsort(confusion_mass)[::-1][:top_n]
            # Always include BG (last index)
            keep = sorted(top_indices.tolist()) + [len(labels) - 1]
            data = data[np.ix_(keep, keep)]
            labels = [labels[i] for i in keep]

    n = len(labels)
    size = min(max(6, 0.35 * n), 20)
    fig, ax = _new_figure((size, size), ax, styled=styled)

    if styled:
        cmap = LinearSegmentedColormap.from_list("hotcoco_seq", SEQUENTIAL)
    else:
        cmap = "viridis"
    im = ax.imshow(data, cmap=cmap, aspect="equal")

    # Cell annotations
    thresh = data.max() / 2.0
    suppress = 0.01 if normalize else 1
    font_kw = {"fontfamily": _resolve_font_family()} if styled else {}
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            if val < suppress:
                continue
            if styled:
                color = "white" if val > thresh else CHROME["text"]
            else:
                color = "white" if val > thresh else "black"
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color=color, fontsize=max(5, min(9, 100 / n)),
                **font_kw,
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")

    if styled:
        _apply_style(ax, "Confusion Matrix", value_axis=None)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return _save_and_return(fig, ax, save_path)


def top_confusions(
    cm_dict: dict[str, Any],
    *,
    top_n: int = 20,
    styled: bool = True,
    ax=None,
    save_path: str | Path | None = None,
) -> tuple:
    """Plot the top N most common misclassifications as horizontal bars.

    This is the go-to plot for large numbers of categories (>30) where a
    full confusion matrix heatmap is unreadable. Shows only off-diagonal
    mistakes: "ground truth X predicted as Y".

    Parameters
    ----------
    cm_dict : dict
        Output of ``coco_eval.confusion_matrix()``.
    top_n : int
        Number of top confusions to show. Default 20.
    styled : bool
        Apply hotcoco visual style (default True). Set False for plain
        matplotlib defaults.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    save_path : str or Path, optional
        Save figure to this path.

    Returns
    -------
    (Figure, Axes)
    """
    _, plt, _ = _import_mpl()
    import numpy as np

    matrix = np.asarray(cm_dict["matrix"], dtype=int)
    cat_names = list(cm_dict["cat_names"])
    K = len(cat_names)
    labels = cat_names + ["BG"]

    # Collect all off-diagonal (i, j, count) pairs
    pairs = []
    for i in range(K + 1):
        for j in range(K + 1):
            if i == j:
                continue
            count = int(matrix[i, j])
            if count > 0:
                pairs.append((labels[i], labels[j], count))

    pairs.sort(key=lambda x: x[2], reverse=True)
    pairs = pairs[:top_n]

    if not pairs:
        fig, ax = _new_figure((8, 3), ax, styled=styled)
        font_kw = {"fontfamily": _resolve_font_family()} if styled else {}
        ax.text(0.5, 0.5, "No confusions found", ha="center", va="center",
                fontsize=12, color=CHROME["text"] if styled else None,
                transform=ax.transAxes, **font_kw)
        if styled:
            _apply_style(ax, "Top Confusions", value_axis=None)
        return _save_and_return(fig, ax, save_path)

    bar_labels = [f"{gt} \u2192 {pred}" for gt, pred, _ in pairs]
    counts = [c for _, _, c in pairs]
    num_bars = len(bar_labels)

    fig_h = max(4, 0.35 * num_bars)
    fig, ax = _new_figure((8, fig_h), ax, styled=styled)

    y_pos = list(range(num_bars))
    bar_kw = {}
    if styled:
        bar_kw = dict(color=SERIES_COLORS[0], edgecolor=CHROME["spine"], linewidth=0.5)
    bars = ax.barh(y_pos, counts, height=0.7, **bar_kw)

    if styled:
        _annotate_bars(ax, bars, counts, fmt="d", fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count")

    if styled:
        _apply_style(
            ax, "Top Confusions",
            subtitle="ground truth \u2192 prediction",
            value_axis="x",
        )

    return _save_and_return(fig, ax, save_path)


def per_category_ap(
    results_dict: dict[str, Any],
    *,
    top_n: int = 20,
    bottom_n: int = 5,
    styled: bool = True,
    ax=None,
    save_path: str | Path | None = None,
) -> tuple:
    """Plot per-category AP as horizontal bars.

    Parameters
    ----------
    results_dict : dict
        Output of ``coco_eval.results(per_class=True)``.
    top_n : int
        Number of top categories to show.
    bottom_n : int
        Number of bottom categories to show.
    styled : bool
        Apply hotcoco visual style (default True). Set False for plain
        matplotlib defaults.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    save_path : str or Path, optional
        Save figure to this path.

    Returns
    -------
    (Figure, Axes)
    """
    _, plt, _ = _import_mpl()

    per_class = results_dict.get("per_class", {})
    if not per_class:
        raise ValueError(
            "No per-class data. Call results(per_class=True)."
        )

    # per_class is {cat_name: ap_value} from results(per_class=True)
    items = [(k, v) for k, v in per_class.items()]

    items.sort(key=lambda x: x[1], reverse=True)

    # Truncate if needed
    if len(items) > top_n + bottom_n:
        top_items = items[:top_n]
        bottom_items = items[-bottom_n:]
        items = top_items + [("...", None)] + bottom_items

    names = [x[0] for x in items]
    values = [x[1] if x[1] is not None else 0 for x in items]
    num_bars = len(names)

    fig_h = max(4, 0.3 * num_bars)
    fig, ax = _new_figure((8, fig_h), ax, styled=styled)

    y_pos = list(range(num_bars))
    if styled:
        colors = [CHROME["grid"] if n == "..." else SERIES_COLORS[0] for n in names]
        bar_kw = dict(color=colors, edgecolor=CHROME["spine"], linewidth=0.5)
    else:
        bar_kw = {}
    bars = ax.barh(y_pos, values, height=0.7, **bar_kw)

    if styled:
        # Skip the "..." separator row
        label_bars = [(b, v) for b, v, n in zip(bars, values, names) if n != "..."]
        if label_bars:
            bs, vs = zip(*label_bars)
            _annotate_bars(ax, bs, vs, fmt=".2f", fontsize=7.5)

    # Mean AP reference line
    real_values = [v for n, v in zip(names, values) if n != "..."]
    mean_ap = sum(real_values) / len(real_values) if real_values else 0
    ax.axvline(
        mean_ap,
        color=SERIES_COLORS[1] if styled else None,
        linestyle="--", linewidth=1,
        label=f"Mean AP: {mean_ap:.3f}",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("AP")
    ax.legend(frameon=not styled, fontsize=9, loc="lower right")

    if styled:
        _apply_style(ax, "Per-Category AP", value_axis="x")

    return _save_and_return(fig, ax, save_path)


def tide_errors(
    tide_dict: dict[str, Any],
    *,
    styled: bool = True,
    ax=None,
    save_path: str | Path | None = None,
) -> tuple:
    """Plot TIDE error breakdown as horizontal bars.

    Parameters
    ----------
    tide_dict : dict
        Output of ``coco_eval.tide_errors()``.
    styled : bool
        Apply hotcoco visual style (default True). Set False for plain
        matplotlib defaults.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    save_path : str or Path, optional
        Save figure to this path.

    Returns
    -------
    (Figure, Axes)
    """
    _, plt, _ = _import_mpl()

    delta_ap = tide_dict["delta_ap"]
    ap_base = tide_dict["ap_base"]

    error_types = ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]
    values = [delta_ap.get(e, 0.0) for e in error_types]

    fig, ax = _new_figure((8, 4), ax, styled=styled)

    y_pos = list(range(len(error_types)))
    bar_kw = {}
    if styled:
        bar_kw = dict(color=SERIES_COLORS[0], edgecolor=CHROME["spine"], linewidth=0.5)
    bars = ax.barh(y_pos, values, height=0.6, **bar_kw)

    if styled:
        _annotate_bars(ax, bars, values, fmt=".3f", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(error_types)
    ax.invert_yaxis()
    ax.set_xlabel("\u0394AP")

    if styled:
        _apply_style(
            ax, "TIDE Error Breakdown",
            subtitle=f"baseline AP={ap_base:.3f}",
            value_axis="x",
        )

    return _save_and_return(fig, ax, save_path)
