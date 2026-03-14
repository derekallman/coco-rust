"""Matplotlib import helpers, figure utilities, and shared plot primitives."""

from __future__ import annotations

from pathlib import Path

_MPL_ERROR = "matplotlib is required for plotting. Install with: pip install hotcoco[plot]"


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
    fonts_dir = Path(__file__).parent.parent / "_fonts"
    for ttf in fonts_dir.glob("*.ttf"):
        try:
            font_manager.fontManager.addfont(str(ttf))
        except Exception:
            pass

    _FONT_FAMILY = ["Inter", "Helvetica Neue", "DejaVu Sans"]
    return _FONT_FAMILY


# ---------------------------------------------------------------------------
# Figure / axes helpers
# ---------------------------------------------------------------------------


def _new_figure(figsize: tuple[float, float], ax=None):
    _, plt, _ = _import_mpl()
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return fig, ax


def _configure_axes(ax, title: str | None = None, subtitle: str | None = None, value_axis: str = "y"):
    """Set grid direction, title, and subtitle. Colors come from active rcParams."""
    if value_axis == "y":
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)
        ax.tick_params(axis="x", length=0)
    elif value_axis == "x":
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        ax.tick_params(axis="y", length=0)
    else:
        ax.grid(False)

    if title:
        ax.set_title(title, fontsize=12, fontweight=500, pad=16 if subtitle else 12)
    if subtitle:
        ax.text(0.5, 1.005, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=9)


def _save_and_return(fig, ax, save_path):
    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, facecolor=fig.get_facecolor())
    return fig, ax


def _annotate_bars(ax, bars, values, *, fmt: str = ".3f", fontsize: float = 9):
    """Add data labels at the end of horizontal bars."""
    max_val = max(values) if values else 1
    for bar, val in zip(bars, values):
        text = f"{val:{fmt}}" if isinstance(val, float) else str(val)
        ax.text(
            bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height() / 2, text, va="center", fontsize=fontsize
        )


def _nearest_iou_idx(iou_thrs: list[float], target: float) -> int:
    return min(range(len(iou_thrs)), key=lambda i: abs(iou_thrs[i] - target))


def _cat_name_lookup(coco_eval) -> dict[int, str]:
    cat_ids = list(coco_eval.params.cat_ids)
    try:
        cats = coco_eval.coco_gt.load_cats(cat_ids)
        return {c["id"]: c["name"] for c in cats}
    except Exception:
        return {cid: str(cid) for cid in cat_ids}


def _resolve_pr_params(coco_eval, area_rng: str, max_det):
    """Resolve precision array and axis indices from a COCOeval object."""
    import numpy as np

    if coco_eval.eval is None:
        raise ValueError("Call coco_eval.run() first.")
    precision = coco_eval.eval["precision"]  # (T, R, K, A, M)
    params = coco_eval.params
    iou_thrs = list(params.iou_thrs)
    area_labels = list(params.area_rng_lbl)
    max_dets = list(params.max_dets)
    cat_ids = list(params.cat_ids)
    a_idx = area_labels.index(area_rng) if area_rng in area_labels else 0
    m_idx = max_dets.index(max_det) if max_det in max_dets else len(max_dets) - 1
    recall_pts = np.linspace(0.0, 1.0, precision.shape[1])
    return precision, iou_thrs, cat_ids, a_idx, m_idx, recall_pts


def _annotate_f1_peak(ax, recall_pts, prec, line):
    """Fill under a PR curve and mark the F1 peak."""
    import numpy as np

    color = line.get_color()
    ax.fill_between(recall_pts, prec, alpha=0.15, color=color)
    f1 = 2 * prec * recall_pts / np.maximum(prec + recall_pts, 1e-8)
    best = int(np.nanargmax(f1))
    if prec[best] > 0:
        ax.plot(recall_pts[best], prec[best], "o", color=color, markersize=5, zorder=5)
        ax.annotate(
            f"F1={f1[best]:.2f}", (recall_pts[best], prec[best]), textcoords="offset points", xytext=(5, 5), fontsize=8
        )
