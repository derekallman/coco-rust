"""Benchmark faster-coco-eval vs hotcoco on Objects365 val.

Generates synthetic detections (capped at MAX_DET_PER_IMAGE per image)
from the ground truth annotations and writes them to a temp file for reuse.

pycocotools is excluded (DNF at O365 scale — needs a beefier machine).

Usage:
    uv run python scripts/bench_objects365.py [--gt PATH] [--max-det N]

Defaults:
    --gt      <workspace>/data/annotations/zhiyuan_objv2_val.json
    --max-det 100
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil

_WORKSPACE = Path(__file__).resolve().parents[1]
_DATA = _WORKSPACE / "data"
_BIN_NAME = "coco-eval.exe" if sys.platform == "win32" else "coco-eval"
RUST_BIN = str(_WORKSPACE / "target/release" / _BIN_NAME)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gt",
        default=str(_DATA / "annotations/zhiyuan_objv2_val.json"),
        help="Path to Objects365 val annotation JSON",
    )
    p.add_argument(
        "--max-det",
        type=int,
        default=100,
        help="Max synthesized detections per image (default: 100)",
    )
    p.add_argument(
        "--dt",
        default=None,
        help="Pre-generated detections JSON (skip generation if provided)",
    )
    return p.parse_args()


def generate_detections(gt_path, max_det_per_image, out_path):
    """Generate noisy synthetic detections from GT, capped per image."""
    print(f"Loading GT from {gt_path} ...")
    with open(gt_path) as f:
        gt = json.load(f)

    rng = random.Random(42)

    # Group annotations by image_id
    by_image: dict[int, list] = {}
    for ann in gt["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        by_image.setdefault(ann["image_id"], []).append(ann)

    results = []
    for img_id, anns in by_image.items():
        # Shuffle so the cap is random, not biased toward first categories
        rng.shuffle(anns)
        for ann in anns[:max_det_per_image]:
            x, y, w, h = ann["bbox"]
            noisy = [
                x + rng.gauss(0, 2),
                y + rng.gauss(0, 2),
                max(1.0, w + rng.gauss(0, 3)),
                max(1.0, h + rng.gauss(0, 3)),
            ]
            results.append(
                {
                    "image_id": img_id,
                    "category_id": ann["category_id"],
                    "bbox": [round(v, 2) for v in noisy],
                    "score": round(min(1.0, max(0.01, rng.gauss(0.7, 0.2))), 4),
                }
            )

    print(
        f"Generated {len(results):,} detections across {len(by_image):,} images "
        f"(cap={max_det_per_image}/image)"
    )
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"Saved detections to {out_path}")
    return out_path


def peak_memory_thread(proc, result):
    """Poll a subprocess's memory (including children) in a background thread.

    result is [peak_physical, peak_committed].
    On Windows, uses peak_wset (OS-tracked, no polling gaps) for physical and
    pagefile usage for total committed (physical + swap).  Falls back to RSS
    polling on other platforms.
    Sums across the process tree to capture real memory usage when the
    subprocess is a wrapper/shim that spawns child processes.
    """
    peak_phys = 0
    peak_commit = 0
    try:
        p = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                procs = [p] + p.children(recursive=True)
                phys = 0
                commit = 0
                for child in procs:
                    try:
                        mi = child.memory_info()
                        if sys.platform == "win32":
                            phys += getattr(mi, "peak_wset", mi.rss)
                            commit += getattr(mi, "pagefile", 0)
                        else:
                            phys += mi.rss
                    except psutil.NoSuchProcess:
                        continue
                if phys > peak_phys:
                    peak_phys = phys
                if commit > peak_commit:
                    peak_commit = commit
            except psutil.NoSuchProcess:
                break
            time.sleep(0.05)
    except psutil.NoSuchProcess:
        pass
    result[0] = peak_phys
    result[1] = peak_commit


_FASTER_RUNNER = """
import sys, time
from faster_coco_eval import COCO, COCOeval_faster
gt_file, dt_file = sys.argv[1], sys.argv[2]
gt = COCO(gt_file)
dt = gt.loadRes(dt_file)
ev = COCOeval_faster(gt, dt, "bbox")
ev.evaluate()
ev.accumulate()
ev.summarize()
"""

_PYCOCOTOOLS_RUNNER = """
import sys, time
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
gt_file, dt_file = sys.argv[1], sys.argv[2]
gt = COCO(gt_file)
dt = gt.loadRes(dt_file)
ev = COCOeval(gt, dt, "bbox")
ev.evaluate()
ev.accumulate()
ev.summarize()
"""


def _bench_python_runner(script, gt_file, dt_file, label):
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        runner = f.name

    try:
        proc = subprocess.Popen(
            [sys.executable, runner, gt_file, dt_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        peak_result = [0, 0]
        monitor = threading.Thread(target=peak_memory_thread, args=(proc, peak_result))
        monitor.start()

        t0 = time.perf_counter()
        stdout, stderr = proc.communicate()
        elapsed = time.perf_counter() - t0
        monitor.join()
    finally:
        os.unlink(runner)

    if proc.returncode != 0:
        print(f"  {label} failed:\n{stderr}", file=sys.stderr)
        return None, None, None

    return elapsed, peak_result[0], peak_result[1]


def bench_faster_coco_eval(gt_file, dt_file):
    return _bench_python_runner(_FASTER_RUNNER, gt_file, dt_file, "faster-coco-eval")


def bench_pycocotools(gt_file, dt_file):
    return _bench_python_runner(_PYCOCOTOOLS_RUNNER, gt_file, dt_file, "pycocotools")


def bench_hotcoco(gt_file, dt_file):
    if not os.path.exists(RUST_BIN):
        print(f"  hotcoco binary not found at {RUST_BIN} — skipping")
        print("  Run: cargo build -p hotcoco-cli --release")
        return None, None, None

    proc = subprocess.Popen(
        [RUST_BIN, "--gt", gt_file, "--dt", dt_file, "--iou-type", "bbox"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    peak_result = [0, 0]
    monitor = threading.Thread(target=peak_memory_thread, args=(proc, peak_result))
    monitor.start()

    t0 = time.perf_counter()
    stdout, stderr = proc.communicate()
    elapsed = time.perf_counter() - t0
    monitor.join()

    if proc.returncode != 0:
        print(f"  hotcoco failed:\n{stderr}", file=sys.stderr)
        return None, None, None

    return elapsed, peak_result[0], peak_result[1]


def fmt_time(t):
    if t is None:
        return "     N/A"
    return f"{t:>7.2f}s"


def fmt_mem(b):
    if b is None or b <= 0:
        return "     N/A"
    return f"{b / 1e9:>6.2f} GB"


def fmt_speedup(base, t):
    if t is None or base is None:
        return "   N/A"
    return f"{base / t:>5.1f}x"


def main():
    args = parse_args()

    if not os.path.exists(args.gt):
        print(f"ERROR: GT file not found: {args.gt}", file=sys.stderr)
        sys.exit(1)

    # Generate or use existing detections
    if args.dt:
        dt_file = args.dt
        print(f"Using pre-generated detections: {dt_file}")
    else:
        dt_path = str(_DATA / f"objects365_val_synth_det_{args.max_det}per.json")
        if os.path.exists(dt_path):
            print(f"Using cached detections: {dt_path}")
            dt_file = dt_path
        else:
            dt_file = generate_detections(args.gt, args.max_det, dt_path)

    print()
    print("=" * 65)
    print("Objects365 val — bbox evaluation")
    print("=" * 65)

    pc_time, pc_phys, pc_commit = None, None, None
    fc_time, fc_phys, fc_commit = None, None, None
    hc_time, hc_phys, hc_commit = None, None, None

    print("Running hotcoco ...", flush=True)
    hc_time, hc_phys, hc_commit = bench_hotcoco(args.gt, dt_file)
    if hc_time is not None:
        print(f"  done in {hc_time:.2f}s, peak RAM {fmt_mem(hc_phys)}, committed {fmt_mem(hc_commit)}")

    print("Running faster-coco-eval ...", flush=True)
    try:
        fc_time, fc_phys, fc_commit = bench_faster_coco_eval(args.gt, dt_file)
        if fc_time is not None:
            print(f"  done in {fc_time:.2f}s, peak RAM {fmt_mem(fc_phys)}, committed {fmt_mem(fc_commit)}")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("Running pycocotools ...", flush=True)
    try:
        pc_time, pc_phys, pc_commit = bench_pycocotools(args.gt, dt_file)
        if pc_time is not None:
            print(f"  done in {pc_time:.2f}s, peak RAM {fmt_mem(pc_phys)}, committed {fmt_mem(pc_commit)}")
    except Exception as e:
        print(f"  FAILED: {e}")

    print()
    print("=" * 80)
    print(f"{'Library':<20} {'Time':>8}  {'Peak RAM':>9}  {'Committed':>9}  {'Speedup':>9}")
    print("-" * 80)
    print(f"{'pycocotools':<20} {fmt_time(pc_time)}  {fmt_mem(pc_phys)}  {fmt_mem(pc_commit)}  {'baseline':>9}")
    print(f"{'faster-coco-eval':<20} {fmt_time(fc_time)}  {fmt_mem(fc_phys)}  {fmt_mem(fc_commit)}  {fmt_speedup(pc_time, fc_time):>9}")
    print(f"{'hotcoco':<20} {fmt_time(hc_time)}  {fmt_mem(hc_phys)}  {fmt_mem(hc_commit)}  {fmt_speedup(pc_time, hc_time):>9}")
    print("=" * 80)
    if sys.platform == "win32":
        print("Peak RAM = peak working set; Committed = pagefile (phys + swap)")


if __name__ == "__main__":
    main()
