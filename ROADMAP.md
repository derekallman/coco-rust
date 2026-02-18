# Roadmap

Planned features and improvements for coco-rust, organized by theme.

## Confusion Matrices

Per-category confusion matrix generation to identify systematic misclassifications.

## TIDE Error Analysis

Error decomposition following [TIDE](https://github.com/dbolya/tide) — classification, localization, duplicate, background, and missed errors.

## Hierarchical Evaluation

Open Images-style evaluation with category hierarchies, where a detection of a parent category is not penalized against a child.

## Video Sequence Analysis

Lightweight per-sequence metric breakdowns for video object detection, surfacing high-level trends like which clips perform worst. Not a substitute for dedicated tracking suites (MOT, HOTA) — just a quick diagnostic view.

## CI/CD

GitHub Actions for:

- `cargo test`
- `cargo clippy`
- `cargo fmt --check`
- Python binding tests (`maturin develop` + pytest)
- Cross-platform matrix (Linux/macOS/Windows)
- Automated release publishing to crates.io and PyPI
