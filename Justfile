# Build the Python extension (required before running any Python scripts)
build:
    uv run maturin develop --release

# Run all tests: Rust unit tests + Python property-based parity tests
test: build
    cargo test
    uv run pytest scripts/test_parity.py -v -x --tb=short

# Verify metric parity vs pycocotools on COCO val2017
parity: build
    uv run python scripts/parity.py

# Run performance benchmarks
bench: build
    uv run python scripts/bench.py

# Lint (warnings are errors, matches CI)
lint:
    cargo clippy --workspace --all-targets -- -D warnings

# Format all Rust code
fmt:
    cargo fmt --all

# Check formatting without modifying (matches CI)
fmt-check:
    cargo fmt --all -- --check
