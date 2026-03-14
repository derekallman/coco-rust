"""Public plot functions: pr_curve, confusion_matrix, top_confusions, per_category_ap, tide_errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .core import (
    _annotate_bars,
    _annotate_f1_peak,
    _cat_name_lookup,
    _configure_axes,
    _import_mpl,
    _nearest_iou_idx,
    _new_figure,
    _resolve_pr_params,
    _save_and_return,
)
from .theme import _build_rc


def pr_curve_iou_sweep(
    coco_eval,
    *,
    iou_thrs: list[float] | None = None,
    area_rng: str = "all",
    max_det: int | None = None,
    theme: str = "warm-slate",
    paper_mode: bool = False,
    ax=None,
    save_path: str | Path | None = None,
) -> tuple:
    """Plot mean precision-recall across all IoU thresholds.

    One line per IoU threshold, precision averaged across all categories.
    The primary line (lowest IoU) gets an under-fill and F1 peak annotation.

    Parameters
    ----------
    coco_eval : COCOeval
        Must have ``run()`` called first.
    iou_thrs : list[float], optional
        IoU thresholds to include. Default: all thresholds in params.
    area_rng : str
        Area range label. Default ``"all"``.
    max_det : int, optional
        Max detections. Default: last entry in ``params.max_dets``.
    theme : str
        ``"warm-slate"`` (default), ``"scientific-blue"``, or ``"ember"``.
    paper_mode : bool
        White figure and axes background for PDF/LaTeX inclusion.
    ax : matplotlib.axes.Axes, optional
    save_path : str or Path, optional

    Returns
    -------
    (Figure, Axes)
    """
    import numpy as np

    mpl, _, _ = _import_mpl()
    precision, all_iou_thrs, _, a_idx, m_idx, recall_pts = _resolve_pr_params(coco_eval, area_rng, max_det)

    t_indices = (
        list(range(len(all_iou_thrs)))
        if iou_thrs is None
        else [i for i, t in enumerate(all_iou_thrs) if t in iou_thrs]
    )

    with mpl.rc_context(_build_rc(theme, paper_mode)):
        fig, ax = _new_figure((6, 6), ax)

        for line_idx, t_idx in enumerate(t_indices):
            p = precision[t_idx, :, :, a_idx, m_idx].copy()
            p[p < 0] = np.nan
            prec = np.nanmean(p, axis=1)
            lw = 2 if line_idx == 0 else 1
            (line,) = ax.plot(recall_pts, prec, linewidth=lw, label=f"IoU={all_iou_thrs[t_idx]:.2f}")
            if line_idx == 0:
                _annotate_f1_peak(ax, recall_pts, prec, line)

        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="Recall", ylabel="Precision")
        ax.legend(fontsize=9, loc="lower left")
        _configure_axes(ax, "Precision-Recall", subtitle="mean over categories", value_axis="y")
    return _save_and_return(fig, ax, save_path)


def pr_curve_by_category(
    coco_eval,
    cat_id: int,
    *,
    iou_thr: float = 0.5,
    area_rng: str = "all",
    max_det: int | None = None,
    theme: str = "warm-slate",
    paper_mode: bool = False,
    ax=None,
    save_path: str | Path | None = None,
) -> tuple:
    """Plot the precision-recall curve for a single category.

    Parameters
    ----------
    coco_eval : COCOeval
        Must have ``run()`` called first.
    cat_id : int
        Category ID to plot.
    iou_thr : float
        IoU threshold. Default 0.50.
    area_rng : str
        Area range label. Default ``"all"``.
    max_det : int, optional
        Max detections. Default: last entry in ``params.max_dets``.
    theme : str
        ``"warm-slate"`` (default), ``"scientific-blue"``, or ``"ember"``.
    paper_mode : bool
        White figure and axes background for PDF/LaTeX inclusion.
    ax : matplotlib.axes.Axes, optional
    save_path : str or Path, optional

    Returns
    -------
    (Figure, Axes)
    """
    import numpy as np

    mpl, _, _ = _import_mpl()
    precision, all_iou_thrs, all_cat_ids, a_idx, m_idx, recall_pts = _resolve_pr_params(coco_eval, area_rng, max_det)

    if cat_id not in all_cat_ids:
        raise ValueError(f"cat_id {cat_id} not in params.cat_ids")

    k_idx = all_cat_ids.index(cat_id)
    t_idx = _nearest_iou_idx(all_iou_thrs, iou_thr)
    prec = precision[t_idx, :, k_idx, a_idx, m_idx].copy()
    prec = np.where(prec < 0, np.nan, prec)
    cat_name = _cat_name_lookup(coco_eval).get(cat_id, str(cat_id))

    with mpl.rc_context(_build_rc(theme, paper_mode)):
        fig, ax = _new_figure((6, 6), ax)
        (line,) = ax.plot(recall_pts, prec, linewidth=2)
        _annotate_f1_peak(ax, recall_pts, prec, line)

        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="Recall", ylabel="Precision")
        _configure_axes(ax, "Precision-Recall", subtitle=cat_name, value_axis="y")
    return _save_and_return(fig, ax, save_path)


def pr_curve_top_n(
    coco_eval,
    *,
    cat_ids: list[int] | None = None,
    top_n: int = 10,
    iou_thr: float = 0.5,
    area_rng: str = "all",
    max_det: int | None = None,
    theme: str = "warm-slate",
    paper_mode: bool = False,
    ax=None,
    save_path: str | Path | None = None,
) -> tuple:
    """Plot precision-recall curves for multiple categories.

    Parameters
    ----------
    coco_eval : COCOeval
        Must have ``run()`` called first.
    cat_ids : list[int], optional
        Categories to plot. If omitted, the top ``top_n`` by AP are selected.
    top_n : int
        Number of top categories when ``cat_ids`` is not given. Default 10.
    iou_thr : float
        IoU threshold. Default 0.50.
    area_rng : str
        Area range label. Default ``"all"``.
    max_det : int, optional
        Max detections. Default: last entry in ``params.max_dets``.
    theme : str
        ``"warm-slate"`` (default), ``"scientific-blue"``, or ``"ember"``.
    paper_mode : bool
        White figure and axes background for PDF/LaTeX inclusion.
    ax : matplotlib.axes.Axes, optional
    save_path : str or Path, optional

    Returns
    -------
    (Figure, Axes)
    """
    import numpy as np

    mpl, _, _ = _import_mpl()
    precision, all_iou_thrs, all_cat_ids, a_idx, m_idx, recall_pts = _resolve_pr_params(coco_eval, area_rng, max_det)
    t_idx = _nearest_iou_idx(all_iou_thrs, iou_thr)
    name_map = _cat_name_lookup(coco_eval)

    if cat_ids is None:
        cat_aps = []
        for k_i, cid in enumerate(all_cat_ids):
            p = precision[t_idx, :, k_i, a_idx, m_idx]
            valid = p[p >= 0]
            cat_aps.append((cid, float(np.mean(valid)) if len(valid) else 0.0))
        cat_aps.sort(key=lambda x: x[1], reverse=True)
        cat_ids = [cid for cid, _ in cat_aps[:top_n]]

    cat_id_to_k = {cid: k for k, cid in enumerate(all_cat_ids)}

    with mpl.rc_context(_build_rc(theme, paper_mode)):
        fig, ax = _new_figure((6, 6), ax)

        for cid in cat_ids:
            k_idx = cat_id_to_k.get(cid)
            if k_idx is None:
                continue
            prec = precision[t_idx, :, k_idx, a_idx, m_idx]
            prec = np.where(prec < 0, np.nan, prec)
            ax.plot(recall_pts, prec, linewidth=1.5, label=name_map.get(cid, str(cid)))

        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="Recall", ylabel="Precision")
        ax.legend(fontsize=8, loc="lower left")
        _configure_axes(ax, "Precision-Recall by Category", subtitle=f"IoU={all_iou_thrs[t_idx]:.2f}", value_axis="y")
    return _save_and_return(fig, ax, save_path)


def pr_curve(coco_eval, *, iou_thrs=None, cat_id=None, cat_ids=None, iou_thr=None, top_n=10,
             area_rng="all", max_det=None, theme="warm-slate", paper_mode=False, ax=None, save_path=None):
    """Dispatch to the appropriate named function.

    Prefer calling directly: ``pr_curve_iou_sweep``, ``pr_curve_by_category``,
    or ``pr_curve_top_n``.
    """
    if cat_id is not None:
        return pr_curve_by_category(coco_eval, cat_id, iou_thr=iou_thr or 0.5,
                                    area_rng=area_rng, max_det=max_det, theme=theme, paper_mode=paper_mode,
                                    ax=ax, save_path=save_path)
    if cat_ids is not None or iou_thr is not None:
        return pr_curve_top_n(coco_eval, cat_ids=cat_ids, top_n=top_n, iou_thr=iou_thr or 0.5,
                              area_rng=area_rng, max_det=max_det, theme=theme, paper_mode=paper_mode,
                              ax=ax, save_path=save_path)
    return pr_curve_iou_sweep(coco_eval, iou_thrs=iou_thrs, area_rng=area_rng,
                              max_det=max_det, theme=theme, paper_mode=paper_mode, ax=ax, save_path=save_path)


def confusion_matrix(
    cm_dict: dict[str, Any],
    *,
    normalize: bool = True,
    top_n: int | None = None,
    group_by: str | None = None,
    cat_groups: dict[str, list[str]] | None = None,
    theme: str = "warm-slate",
    paper_mode: bool = False,
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
    theme : str
        ``"warm-slate"`` (default), ``"scientific-blue"``, or ``"ember"``.
    paper_mode : bool
        White figure and axes background for PDF/LaTeX inclusion.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    save_path : str or Path, optional
        Save figure to this path.

    Returns
    -------
    (Figure, Axes)
    """
    import numpy as np

    mpl, _, _ = _import_mpl()

    raw_matrix = np.asarray(cm_dict["matrix"], dtype=float)
    norm_matrix = np.asarray(cm_dict["normalized"], dtype=float)
    cat_names = list(cm_dict["cat_names"])
    K = len(cat_names)

    # ---- Supercategory grouping ----
    if group_by == "supercategory" and cat_groups is not None:
        name_to_idx = {n: i for i, n in enumerate(cat_names)}
        group_names = sorted(cat_groups.keys())
        G = len(group_names)

        grouped = np.zeros((G + 1, G + 1), dtype=float)
        group_idx_map = {}
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
            grouped[gi, G] += raw_matrix[i, K]
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

        if top_n is None and K > 30:
            top_n = 25

        if top_n is not None and top_n < K:
            row_total = data[:K, :K].sum(axis=1)
            col_total = data[:K, :K].sum(axis=0)[:K]
            diag = np.diag(data[:K, :K])
            confusion_mass = (row_total - diag) + (col_total - diag)
            top_indices = np.argsort(confusion_mass)[::-1][:top_n]
            keep = sorted(top_indices.tolist()) + [len(labels) - 1]
            data = data[np.ix_(keep, keep)]
            labels = [labels[i] for i in keep]

    n = len(labels)
    size = min(max(6, 0.35 * n), 20)
    rc = _build_rc(theme, paper_mode)
    with mpl.rc_context(rc):
        fig, ax = _new_figure((size, size), ax)

        im = ax.imshow(data, aspect="equal", vmin=0)  # cmap from active rcParams; vmin=0 anchors zeros to white

        thresh = data.max() / 2.0
        suppress = 0.01 if normalize else 1
        for i in range(n):
            for j in range(n):
                val = data[i, j]
                if val < suppress:
                    continue
                color = "white" if val > thresh else mpl.rcParams["text.color"]
                text = f"{val:.2f}" if normalize else f"{int(val)}"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=max(5, min(9, 100 / n)))

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground truth")
        ax.set_frame_on(False)
        _configure_axes(ax, "Confusion Matrix", value_axis=None)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return _save_and_return(fig, ax, save_path)


def top_confusions(
    cm_dict: dict[str, Any], *, top_n: int = 20, theme: str = "warm-slate", paper_mode: bool = False,
    ax=None, save_path: str | Path | None = None
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
    theme : str
        ``"warm-slate"`` (default), ``"scientific-blue"``, or ``"ember"``.
    paper_mode : bool
        White figure and axes background for PDF/LaTeX inclusion.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    save_path : str or Path, optional
        Save figure to this path.

    Returns
    -------
    (Figure, Axes)
    """
    import numpy as np

    mpl, _, _ = _import_mpl()

    matrix = np.asarray(cm_dict["matrix"], dtype=int)
    cat_names = list(cm_dict["cat_names"])
    K = len(cat_names)
    labels = cat_names + ["BG"]

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

    with mpl.rc_context(_build_rc(theme, paper_mode)):
        if not pairs:
            fig, ax = _new_figure((8, 3), ax)
            ax.text(0.5, 0.5, "No confusions found", ha="center", va="center", fontsize=12, transform=ax.transAxes)
            _configure_axes(ax, "Top Confusions", value_axis=None)
            return _save_and_return(fig, ax, save_path)

        bar_labels = [f"{gt} \u2192 {pred}" for gt, pred, _ in pairs]
        counts = [c for _, _, c in pairs]
        num_bars = len(bar_labels)

        fig, ax = _new_figure((8, max(4, 0.35 * num_bars)), ax)
        bars = ax.barh(range(num_bars), counts, height=0.7)
        _annotate_bars(ax, bars, counts, fmt="d", fontsize=8)

        ax.set_yticks(range(num_bars))
        ax.set_yticklabels(bar_labels)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        _configure_axes(ax, "Top Confusions", subtitle="ground truth \u2192 prediction", value_axis="x")

    return _save_and_return(fig, ax, save_path)


def per_category_ap(
    results_dict: dict[str, Any],
    *,
    top_n: int = 20,
    bottom_n: int = 5,
    theme: str = "warm-slate",
    paper_mode: bool = False,
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
    theme : str
        ``"warm-slate"`` (default), ``"scientific-blue"``, or ``"ember"``.
    paper_mode : bool
        White figure and axes background for PDF/LaTeX inclusion.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    save_path : str or Path, optional
        Save figure to this path.

    Returns
    -------
    (Figure, Axes)
    """
    mpl, _, _ = _import_mpl()

    per_class = results_dict.get("per_class", {})
    if not per_class:
        raise ValueError("No per-class data. Call results(per_class=True).")

    items = sorted(per_class.items(), key=lambda x: x[1], reverse=True)

    if len(items) > top_n + bottom_n:
        items = items[:top_n] + [("...", None)] + items[-bottom_n:]

    names = [x[0] for x in items]
    values = [x[1] if x[1] is not None else 0 for x in items]
    num_bars = len(names)

    with mpl.rc_context(_build_rc(theme, paper_mode)):
        fig, ax = _new_figure((8, max(4, 0.3 * num_bars)), ax)

        bar_color = next(iter(mpl.rcParams["axes.prop_cycle"]))["color"]
        sep_color = mpl.rcParams.get("grid.color", "#cccccc")
        colors = [sep_color if n == "..." else bar_color for n in names]
        bars = ax.barh(range(num_bars), values, height=0.7, color=colors)

        label_bars = [(b, v) for b, v, n in zip(bars, values, names) if n != "..."]
        if label_bars:
            bs, vs = zip(*label_bars)
            _annotate_bars(ax, bs, vs, fmt=".2f", fontsize=7.5)

        real_values = [v for n, v in zip(names, values) if n != "..."]
        mean_ap = sum(real_values) / len(real_values) if real_values else 0
        ax.axvline(mean_ap, linestyle="--", linewidth=1, label=f"Mean AP: {mean_ap:.3f}")

        ax.set_yticks(range(num_bars))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("AP")
        ax.legend(fontsize=9, loc="lower right")
        _configure_axes(ax, "Per-Category AP", value_axis="x")

    return _save_and_return(fig, ax, save_path)


def tide_errors(
    tide_dict: dict[str, Any], *, theme: str = "warm-slate", paper_mode: bool = False,
    ax=None, save_path: str | Path | None = None
) -> tuple:
    """Plot TIDE error breakdown as horizontal bars.

    Parameters
    ----------
    tide_dict : dict
        Output of ``coco_eval.tide_errors()``.
    theme : str
        ``"warm-slate"`` (default), ``"scientific-blue"``, or ``"ember"``.
    paper_mode : bool
        White figure and axes background for PDF/LaTeX inclusion.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    save_path : str or Path, optional
        Save figure to this path.

    Returns
    -------
    (Figure, Axes)
    """
    mpl, _, _ = _import_mpl()

    delta_ap = tide_dict["delta_ap"]
    ap_base = tide_dict["ap_base"]

    error_types = ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]
    values = [delta_ap.get(e, 0.0) for e in error_types]

    with mpl.rc_context(_build_rc(theme, paper_mode)):
        fig, ax = _new_figure((8, 4), ax)
        bars = ax.barh(range(len(error_types)), values, height=0.6)
        _annotate_bars(ax, bars, values, fmt=".3f", fontsize=9)

        ax.set_yticks(range(len(error_types)))
        ax.set_yticklabels(error_types)
        ax.invert_yaxis()
        ax.set_xlabel("\u0394AP")
        _configure_axes(ax, "TIDE Error Breakdown", subtitle=f"baseline AP={ap_base:.3f}", value_axis="x")

    return _save_and_return(fig, ax, save_path)
