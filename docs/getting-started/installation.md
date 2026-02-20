# Installation

## Python

```bash
pip install coco-rs
```

Verify the installation:

```python
from coco_rs import COCO
print("coco-rs installed successfully")
```

!!! note "numpy"
    coco-rs requires numpy, which is installed automatically. If you need a specific numpy version, install it first.

??? info "Build from source"
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

??? info "Build from source"
    ```bash
    git clone https://github.com/derekallman/coco-rust.git
    cd coco-rust
    cargo build --release
    # Binary is at target/release/coco-eval
    ```

## Rust library

```bash
cargo add coco-rs
```

Or add it manually to your `Cargo.toml`:

```toml
[dependencies]
coco-rs = "0.1"
```

Full API documentation is on [docs.rs](https://docs.rs/coco-rs).
