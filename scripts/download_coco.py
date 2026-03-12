"""Download COCO val2017 annotations and generate parity result files.

Downloads the official COCO 2017 validation annotations (~240 MB) and generates
three deterministic synthetic detection files used by `just parity`.

Usage:
    uv run python scripts/download_coco.py
    just download-coco

Flags:
    --data-dir PATH   Root data directory (default: data/)
    --force           Overwrite files that already exist
    --skip-download   Skip annotation download (generate result files only)
    --skip-generate   Skip result file generation (download annotations only)
"""

import argparse
import json
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
NEEDED_ANNOTATION_FILES = {
    "annotations/instances_val2017.json",
    "annotations/person_keypoints_val2017.json",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _make_progress(label: str):
    """Return a urllib reporthook that prints a single updating line."""

    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / 1_048_576
            total_mb = total_size / 1_048_576
            print(f"\r  {label}: {mb:.1f} / {total_mb:.1f} MB  ({pct}%)", end="", flush=True)
        else:
            mb = downloaded / 1_048_576
            print(f"\r  {label}: {mb:.1f} MB", end="", flush=True)

    return hook


def download_annotations(data_dir: Path, force: bool) -> bool:
    """Download and extract val2017 annotation files. Returns True if anything changed."""
    ann_dir = data_dir / "annotations"
    targets = [ann_dir / Path(f).name for f in NEEDED_ANNOTATION_FILES]

    if not force and all(t.exists() for t in targets):
        for t in targets:
            print(f"  exists: {t}  ({t.stat().st_size / 1_048_576:.1f} MB)")
        return False

    ann_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "annotations_trainval2017.zip"

    if force or not zip_path.exists():
        print(f"Downloading {ANNOTATIONS_URL}")
        urllib.request.urlretrieve(ANNOTATIONS_URL, zip_path, _make_progress("download"))
        print()  # newline after progress
    else:
        print(f"  zip cached: {zip_path}")

    print("Extracting needed files from zip...")
    with zipfile.ZipFile(zip_path) as zf:
        for member in NEEDED_ANNOTATION_FILES:
            dest = ann_dir / Path(member).name
            if not force and dest.exists():
                print(f"  exists: {dest}")
                continue
            print(f"  extracting {member} → {dest}")
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            print(f"  wrote {dest.stat().st_size / 1_048_576:.1f} MB")

    return True


# ---------------------------------------------------------------------------
# Detection generation
# ---------------------------------------------------------------------------


def _gen_bbox(gt: dict) -> list:
    random.seed(42)
    results = []
    for ann in gt["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        bbox = ann["bbox"]
        noisy_bbox = [
            bbox[0] + random.gauss(0, 2),
            bbox[1] + random.gauss(0, 2),
            max(1, bbox[2] + random.gauss(0, 3)),
            max(1, bbox[3] + random.gauss(0, 3)),
        ]
        score = min(1.0, max(0.01, random.gauss(0.7, 0.2)))
        results.append({
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "bbox": [round(x, 2) for x in noisy_bbox],
            "score": round(score, 4),
            "area": round(noisy_bbox[2] * noisy_bbox[3], 2),
        })
        if random.random() < 0.2:
            fp_bbox = [
                random.uniform(0, 400),
                random.uniform(0, 400),
                random.uniform(10, 100),
                random.uniform(10, 100),
            ]
            results.append({
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "bbox": [round(x, 2) for x in fp_bbox],
                "score": round(random.uniform(0.01, 0.5), 4),
                "area": round(fp_bbox[2] * fp_bbox[3], 2),
            })
    return results


def _gen_segm(gt: dict) -> list:
    random.seed(42)
    img_dims = {img["id"]: (img["width"], img["height"]) for img in gt["images"]}
    results = []
    for ann in gt["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        seg = ann.get("segmentation")
        if not seg or not isinstance(seg, list):
            continue
        noisy_seg = [
            [round(c + random.gauss(0, 1.5), 2) for c in poly]
            for poly in seg
        ]
        bbox = ann["bbox"]
        noisy_bbox = [
            bbox[0] + random.gauss(0, 2),
            bbox[1] + random.gauss(0, 2),
            max(1, bbox[2] + random.gauss(0, 3)),
            max(1, bbox[3] + random.gauss(0, 3)),
        ]
        score = min(1.0, max(0.01, random.gauss(0.7, 0.2)))
        results.append({
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "segmentation": noisy_seg,
            "bbox": [round(x, 2) for x in noisy_bbox],
            "score": round(score, 4),
            "area": round(noisy_bbox[2] * noisy_bbox[3], 2),
        })
        if random.random() < 0.2:
            img_w, img_h = img_dims.get(ann["image_id"], (640, 480))
            cx = random.uniform(50, img_w - 50)
            cy = random.uniform(50, img_h - 50)
            size = random.uniform(10, 50)
            fp_poly = [
                round(cx - size + random.gauss(0, 5), 2), round(cy - size + random.gauss(0, 5), 2),
                round(cx + size + random.gauss(0, 5), 2), round(cy - size + random.gauss(0, 5), 2),
                round(cx + size + random.gauss(0, 5), 2), round(cy + size + random.gauss(0, 5), 2),
                round(cx - size + random.gauss(0, 5), 2), round(cy + size + random.gauss(0, 5), 2),
            ]
            fp_bbox = [round(cx - size, 2), round(cy - size, 2), round(2 * size, 2), round(2 * size, 2)]
            results.append({
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "segmentation": [fp_poly],
                "bbox": fp_bbox,
                "score": round(random.uniform(0.01, 0.5), 4),
                "area": round(4 * size * size, 2),
            })
    return results


def _gen_kpt(gt: dict) -> list:
    random.seed(42)
    img_dims = {img["id"]: (img["width"], img["height"]) for img in gt["images"]}
    results = []
    for ann in gt["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        kpts = ann.get("keypoints")
        if not kpts or len(kpts) < 51:
            continue
        noisy_kpts = []
        for i in range(0, len(kpts), 3):
            x, y, v = kpts[i], kpts[i + 1], kpts[i + 2]
            if v > 0:
                noisy_kpts.extend([round(x + random.gauss(0, 3), 2), round(y + random.gauss(0, 3), 2), 1])
            else:
                noisy_kpts.extend([0, 0, 0])
        bbox = ann["bbox"]
        noisy_bbox = [
            bbox[0] + random.gauss(0, 2),
            bbox[1] + random.gauss(0, 2),
            max(1, bbox[2] + random.gauss(0, 3)),
            max(1, bbox[3] + random.gauss(0, 3)),
        ]
        score = min(1.0, max(0.01, random.gauss(0.7, 0.2)))
        results.append({
            "image_id": ann["image_id"],
            "category_id": 1,
            "keypoints": noisy_kpts,
            "bbox": [round(x, 2) for x in noisy_bbox],
            "score": round(score, 4),
            "area": round(noisy_bbox[2] * noisy_bbox[3], 2),
        })
        if random.random() < 0.2:
            img_w, img_h = img_dims.get(ann["image_id"], (640, 480))
            cx = random.uniform(50, img_w - 50)
            cy = random.uniform(50, img_h - 50)
            fp_kpts = [v for _ in range(17) for v in [round(cx + random.gauss(0, 30), 2), round(cy + random.gauss(0, 30), 2), 1]]
            results.append({
                "image_id": ann["image_id"],
                "category_id": 1,
                "keypoints": fp_kpts,
                "bbox": [round(cx - 30, 2), round(cy - 50, 2), 60.0, 100.0],
                "score": round(random.uniform(0.01, 0.5), 4),
                "area": round(60.0 * 100.0, 2),
            })
    return results


def generate_results(data_dir: Path, force: bool) -> bool:
    """Generate synthetic detection result files. Returns True if anything was written."""
    ann_dir = data_dir / "annotations"
    tasks = [
        ("bbox", data_dir / "bbox_val2017_results.json", ann_dir / "instances_val2017.json", _gen_bbox),
        ("segm", data_dir / "segm_val2017_results.json", ann_dir / "instances_val2017.json", _gen_segm),
        ("kpt",  data_dir / "kpt_val2017_results.json",  ann_dir / "person_keypoints_val2017.json", _gen_kpt),
    ]

    wrote_any = False
    for label, out_path, ann_path, gen_fn in tasks:
        if not force and out_path.exists():
            print(f"  exists: {out_path}  ({out_path.stat().st_size / 1_048_576:.1f} MB)")
            continue
        if not ann_path.exists():
            print(f"  ERROR: annotation file missing: {ann_path}", file=sys.stderr)
            print(f"         Run without --skip-download to fetch it first.", file=sys.stderr)
            sys.exit(1)
        print(f"  generating {label} results from {ann_path.name}...", end=" ", flush=True)
        with open(ann_path) as f:
            gt = json.load(f)
        results = gen_fn(gt)
        with open(out_path, "w") as f:
            json.dump(results, f)
        print(f"{len(results):,} detections → {out_path.stat().st_size / 1_048_576:.1f} MB")
        wrote_any = True
    return wrote_any


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data", metavar="PATH", help="Root data directory (default: data/)")
    parser.add_argument("--force", action="store_true", help="Overwrite files that already exist")
    parser.add_argument("--skip-download", action="store_true", help="Skip annotation download")
    parser.add_argument("--skip-generate", action="store_true", help="Skip result file generation")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print("── Annotations ──────────────────────────────")
        download_annotations(data_dir, args.force)

    if not args.skip_generate:
        print("── Detection results ─────────────────────────")
        generate_results(data_dir, args.force)

    print("Done. Run `just parity` to verify.")


if __name__ == "__main__":
    main()
