"""Convert Objects365 HuggingFace parquet (validation split) to COCO JSON.

Downloads parquet files from jxu124/objects365 on HuggingFace,
reassembles them into a standard COCO annotation JSON.

Output: data/annotations/zhiyuan_objv2_val.json

Usage:
    uv run python scripts/download_o365.py [--out PATH]
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

try:
    import polars as pl
except ImportError:
    sys.exit("polars not installed — run: uv pip install polars")



HF_REPO = "jxu124/objects365"
SPLIT = "validation"
OUT_DEFAULT = "data/annotations/zhiyuan_objv2_val.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT_DEFAULT, help="Output COCO JSON path")
    p.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace cache dir (default: ~/.cache/huggingface)",
    )
    return p.parse_args()


def download_parquets(repo: str, split: str, cache_dir: str | None) -> list[Path]:
    """Download all parquet shards for the split via HF API + urllib."""
    api_url = f"https://huggingface.co/api/datasets/{repo}/parquet/default/{split}"
    print("Fetching parquet file list from HF API ...")
    with urllib.request.urlopen(api_url) as resp:
        urls = json.load(resp)  # list of URL strings
    print(f"  Found {len(urls)} parquet shard(s) for '{split}' split")

    cache = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "hf_objects365"
    cache.mkdir(parents=True, exist_ok=True)

    paths = []
    for url in urls:
        fname = url.split("/")[-1]
        dest = cache / f"{split}_{fname}"
        if dest.exists():
            print(f"  {fname} (cached)")
        else:
            print(f"  Downloading {fname} ...", flush=True)
            urllib.request.urlretrieve(url, dest)
            print(f"  Saved to {dest}")
        paths.append(dest)
    return paths


def build_coco_json(parquet_paths: list[Path]) -> dict:
    """Read parquet shards and assemble a COCO-format annotation dict."""
    print("Reading parquet shards with polars ...")
    df = pl.concat([pl.read_parquet(p) for p in parquet_paths])
    print(f"  Loaded {len(df):,} rows (images)")

    # ── Images ──────────────────────────────────────────────────────────────
    # image_info is a struct column: {file_name, height, id, license, url, width}
    img_df = (
        df.select("image_info")
        .unnest("image_info")
        .select(["id", "file_name", "width", "height"])
        .rename({"id": "id"})
    )
    images = img_df.to_dicts()
    print(f"  Images: {len(images):,}")

    # ── Annotations ─────────────────────────────────────────────────────────
    # anns_info is a list-of-structs; explode to get one row per annotation
    ann_df = (
        df.select("anns_info")
        .explode("anns_info")
        .unnest("anns_info")
        .select(["id", "image_id", "category_id", "bbox", "area", "iscrowd"])
    )

    # bbox in HF is XYXY [x1, y1, x2, y2]; COCO expects XYWH [x, y, w, h]
    ann_df = ann_df.with_columns(
        pl.struct(
            x=pl.col("bbox").list.get(0),
            y=pl.col("bbox").list.get(1),
            w=pl.col("bbox").list.get(2) - pl.col("bbox").list.get(0),
            h=pl.col("bbox").list.get(3) - pl.col("bbox").list.get(1),
        )
        .map_elements(
            lambda s: [s["x"], s["y"], s["w"], s["h"]], return_dtype=pl.List(pl.Float64)
        )
        .alias("bbox")
    )

    # Ensure annotation IDs are globally unique (they should be, but just in case)
    if ann_df["id"].n_unique() < len(ann_df):
        print("  WARNING: duplicate annotation IDs detected — reassigning")
        ann_df = ann_df.with_row_index("id")

    annotations = ann_df.to_dicts()
    print(f"  Annotations: {len(annotations):,}")

    # ── Categories ──────────────────────────────────────────────────────────
    # Extract unique (category_id, category_name) from annotations
    cat_df = (
        df.select("anns_info")
        .explode("anns_info")
        .unnest("anns_info")
        .select(["category_id", "category"])
        .unique()
        .sort("category_id")
    )
    categories = [
        {"id": row["category_id"], "name": row["category"], "supercategory": ""}
        for row in cat_df.to_dicts()
    ]
    print(f"  Categories: {len(categories)}")

    return {
        "info": {
            "description": "Objects365 v2 validation",
            "url": "https://www.objects365.org",
            "version": "2.0",
            "year": 2020,
            "contributor": "Objects365 Consortium",
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_paths = download_parquets(HF_REPO, SPLIT, args.cache_dir)
    coco = build_coco_json(parquet_paths)

    print(f"Writing COCO JSON to {out_path} ...")
    with open(out_path, "w") as f:
        json.dump(coco, f)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"Done — {size_mb:.1f} MB")
    print()
    print(f"Run benchmark with:")
    print(f"  uv run python scripts/bench_objects365.py --gt {out_path}")


if __name__ == "__main__":
    main()
