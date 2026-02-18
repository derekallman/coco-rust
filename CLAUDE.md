# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

coco-rust is a pure Rust port of [pycocotools](https://github.com/ppwwyyxx/cocoapi) with PyO3 Python bindings. It provides 11-26x speedups over pycocotools for bbox, segmentation, and keypoint evaluation.

- **Primary language:** Rust. All core logic lives in `coco-rs`.
- **Python bindings:** PyO3/maturin in `coco-pyo3`, exposed as the `coco_rs` Python package.
- **CLI:** `coco-cli` binary wrapping the Rust library.

## Workspace Structure

```
crates/coco-rs/      # Pure Rust library — types, mask ops, COCO API, evaluation
crates/coco-cli/     # CLI binary
crates/coco-pyo3/    # PyO3 Python bindings (cdylib, built with maturin)
```

### Key Architecture

- `coco-pyo3` uses `coco-core` as the Cargo dependency alias for `coco-rs` to avoid name collision with the `coco_rs` Python module name
- Python bindings return plain dicts (not wrapped Rust structs) matching pycocotools conventions
- Mask operations handle numpy row-major <-> Rust column-major transposition in the PyO3 layer
- `cargo build --workspace` will fail at link time for coco-pyo3 (expected — cdylib needs Python). Use `cargo check` instead, or build via maturin.

## Metric Parity

All 12 COCO evaluation metrics (AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl) must match pycocotools. Keypoints has 10 metrics (no small area range).

- **Always ensure exact parity when modifying evaluation logic.** Run `cargo test` after Rust changes.
- Verified on val2017: bbox exact, segm within 0.003, keypoints exact.
- When in doubt, run differential tests against pycocotools on real COCO data before declaring a task complete.

## Benchmarking

- **Use wall clock time**, not CPU time.
- **Only scale detections** when creating synthetic benchmarks (never scale ground truth).
- Format benchmark tables consistently: columns are `[Eval Type | pycocotools | faster-coco-eval | coco-rust]`, times in seconds with 2 decimal places, speedups in parentheses vs pycocotools.
- Always verify all 12 metrics still match before reporting timing results.

## Build Commands

```bash
cargo build                    # Build all crates
cargo test                     # Run all tests
cargo test -p coco-rs          # Run library tests only
cargo check -p coco-pyo3       # Check pyo3 crate (can't link without Python)
cargo clippy                   # Lint
cargo fmt --all                # Format (use --all, not --workspace)
cargo fmt --all -- --check     # Check formatting

# Python bindings
cd crates/coco-pyo3
maturin develop --release      # Build + install into active Python env
```

## Git Workflow

- When committing and pushing, always verify the current git status first to avoid trying to commit already-committed changes. Check `git status` and `git log --oneline -3` before any commit/push operation.
- Main branch: `main`.
- Run `cargo test` and `cargo clippy` before committing.
