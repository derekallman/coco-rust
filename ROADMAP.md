# Roadmap

Planned features and improvements, organized by theme.

## Architecture

All core logic lives in the Rust library. The Python package and Rust CLI are thin wrappers.

```
                       ┌──→ PyO3 ──→ hotcoco (Python library + CLI)
Rust Core (all logic) ─┤
                       └──→ hotcoco-cli (Rust CLI, eval only)
```

- **Rust core** — types, masks, eval, dataset ops, format conversion, streaming
- **Python CLI** (primary) — all subcommands: `$ coco eval`, `$ coco stats`, `$ coco merge`, `$ coco plot`, etc. Rich formatting, plots via matplotlib/plotly.
- **Rust CLI** (`hotcoco-cli`) — evaluation only. JSON/CSV/markdown output, no plots, no Python. New features added on request.

| Registry | Package | Contents |
|----------|---------|----------|
| PyPI | `hotcoco` | Python library + Python CLI + compiled Rust core |
| crates.io | `hotcoco` | Rust library |
| crates.io | `hotcoco-cli` | Rust CLI binary (eval only, no Python) |

## Dataset Statistics

**Shipped.** Quick health check for any COCO dataset — annotation counts per category, image size distributions, area distributions, crowd/iscrowd breakdown. Available as `coco.stats()` in Python and `coco stats` in the Python CLI.

## Dataset Operations

**Shipped.** Split, merge, filter, and sample COCO datasets:

- ~~**Filter** — subset by category, image ID, area range, or custom predicate~~
- ~~**Merge** — combine multiple annotation files (e.g., separate labeling batches)~~
- ~~**Split** — reproducible train/val/test split with deterministic shuffle~~
- ~~**Sample** — random or deterministic subset for quick iteration~~

All implemented in Rust core, exposed via Python CLI and Python API.

## Extended Dataset Support

Support for large-scale datasets that use the COCO annotation format but require evaluation tweaks. These are high-value targets because their scale makes pycocotools particularly painful — the speed advantage compounds.

### Objects365

Standard COCO evaluation protocol over 365 categories and ~2M images. Likely works today since it's standard COCO format; needs verification and explicit documentation. The main story here is scale: at O365 size, hotcoco's speed advantage is most dramatic.

### LVIS (Large Vocabulary Instance Segmentation)

~1,200 category long-tail dataset requiring **federated AP** evaluation — per-category results are computed independently across the subset of images that contain each category, rather than globally. Using standard pycocotools on LVIS gives subtly wrong numbers. Increasingly common in foundation model benchmarking (SAM, DINO, CLIP-based detectors).

### CrowdPose

Keypoint dataset for crowded scenes. Uses a modified OKS matching algorithm with a crowd factor to handle overlapping people. Requires a custom matching pass in the evaluator. Lower priority than LVIS/O365 but rounds out pose evaluation coverage.

## Format Conversion

COCO ↔ YOLO, COCO ↔ Pascal VOC, COCO ↔ CVAT. Everyone has a slightly broken converter script — a correct, fast, well-tested one has high value. High surface area though, so this should be scoped carefully (start with YOLO, the most requested).

## Streaming Evaluation

Evaluate datasets that don't fit in memory. Process annotations in chunks without loading the full ground truth and detection sets upfront. Critical for production datasets with millions of annotations.

## Confusion Matrices

Per-category confusion matrix generation to identify systematic misclassifications.

## TIDE Error Analysis

Error decomposition following [TIDE](https://github.com/dbolya/tide) — classification, localization, duplicate, background, and missed errors.

## Hierarchical Evaluation

Open Images-style evaluation with category hierarchies, where a detection of a parent category is not penalized against a child.

## Video Sequence Analysis

Lightweight per-sequence metric breakdowns for video object detection, surfacing high-level trends like which clips perform worst. Not a substitute for dedicated tracking suites (MOT, HOTA) — just a quick diagnostic view.

## CI/CD

GitHub Actions CI (shipped):

- ~~`cargo test` / `cargo clippy` / `cargo fmt --check`~~
- ~~Cross-platform matrix (Linux/macOS/Windows)~~
- ~~Python smoke test (`maturin develop` + inline import/eval check)~~
- ~~Automated release publishing to crates.io and PyPI~~
