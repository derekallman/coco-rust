# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

coco-rust is a Rust implementation of [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools), the Python API for the COCO (Common Objects in Context) dataset. COCO is a standard benchmark for object detection, segmentation, and captioning evaluation in computer vision.

## Build Commands

```bash
cargo build            # Build the project
cargo test             # Run all tests
cargo test <test_name> # Run a single test
cargo clippy           # Lint
cargo fmt              # Format code
cargo fmt -- --check   # Check formatting without modifying
```

## Status

This project is in its initial state — no Cargo.toml or source code exists yet. The repository needs to be bootstrapped with a Cargo manifest and module structure.
