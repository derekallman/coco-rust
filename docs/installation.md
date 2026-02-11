# Installation

## Python

```bash
pip install coco-rs
```

!!! note "numpy dependency"
    coco-rs requires numpy, which is installed automatically. If you need a specific numpy version, install it first.

Verify the installation:

```python
from coco_rs import COCO
coco = COCO("annotations.json")
print(coco.get_img_ids())
```

### From source

```bash
git clone https://github.com/derekallman/coco-rust.git
cd coco-rust/crates/coco-pyo3
pip install maturin
maturin develop --release
```

This builds the `coco_rs` Python module and installs it into your active environment.

## CLI

```bash
cargo install coco-cli
```

This installs the `coco-eval` binary.

### From source

```bash
git clone https://github.com/derekallman/coco-rust.git
cd coco-rust
cargo build --release
# Binary is at target/release/coco-eval
```

## Rust library

Add `coco-rs` to your project:

```bash
cargo add coco-rs
```

Or add it manually to your `Cargo.toml`:

```toml
[dependencies]
coco-rs = "0.1"
```
