# Installation

## Python

```bash
pip install hotcoco
```

Verify the installation:

```python
from hotcoco import COCO
print("hotcoco installed successfully")
```

!!! note "numpy"
    hotcoco requires numpy, which is installed automatically. If you need a specific numpy version, install it first.

??? info "Build from source"
    ```bash
    git clone https://github.com/derekallman/hotcoco.git
    cd hotcoco/crates/hotcoco-pyo3
    pip install maturin
    maturin develop --release
    ```
    This builds the `hotcoco` Python module and installs it into your active environment.

## CLI

```bash
cargo install hotcoco-cli
```

This installs the `coco-eval` binary.

??? info "Build from source"
    ```bash
    git clone https://github.com/derekallman/hotcoco.git
    cd hotcoco
    cargo build --release
    # Binary is at target/release/coco-eval
    ```

## Rust library

```bash
cargo add hotcoco
```

Or add it manually to your `Cargo.toml`:

```toml
[dependencies]
hotcoco = "0.1"
```

Full API documentation is on [docs.rs](https://docs.rs/hotcoco).
