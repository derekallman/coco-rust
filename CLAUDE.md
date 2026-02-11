# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

coco-rust is a Rust implementation of [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools), the Python API for the COCO (Common Objects in Context) dataset. It provides 11-26x speedups over pycocotools for bbox, segmentation, and keypoint evaluation.

## Workspace Structure

```
crates/coco-rs/      # Pure Rust library — all core logic
crates/coco-cli/     # CLI binary
crates/coco-pyo3/    # PyO3 Python bindings (cdylib, built with maturin)
```

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

## Key Architecture Notes

- `coco-pyo3` uses `coco-core` as the dependency alias for `coco-rs` to avoid name collision with the `coco_rs` Python module name
- Python bindings return plain dicts (not wrapped Rust structs) matching pycocotools conventions
- Mask operations handle numpy row-major ↔ Rust column-major transposition in the PyO3 layer
- `cargo build --workspace` will fail at link time for coco-pyo3 (expected — cdylib needs Python). Use `cargo check` instead, or build via maturin.
