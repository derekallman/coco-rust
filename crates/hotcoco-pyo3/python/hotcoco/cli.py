"""
hotcoco command-line interface.

Usage:
    coco stats <annotation_file>
"""

import argparse
import os
import sys


def cmd_stats(args):
    try:
        from hotcoco import COCO
    except ImportError:
        print("error: hotcoco is not installed", file=sys.stderr)
        sys.exit(1)

    try:
        coco = COCO(args.annotation_file)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    s = coco.stats()
    filename = os.path.basename(args.annotation_file)

    ann_count = s["annotation_count"]
    crowd_count = s["crowd_count"]
    crowd_pct = 100.0 * crowd_count / ann_count if ann_count > 0 else 0.0

    print(filename)
    print()
    print(f"  Images:      {s['image_count']:>6,}")
    print(f"  Annotations: {ann_count:>6,}")
    print(f"  Categories:  {s['category_count']:>6,}")
    print(f"  Crowd:       {crowd_count:>6,}  ({crowd_pct:.1f}%)")

    per_cat = s["per_category"]
    if per_cat:
        shown = per_cat if args.all_cats else per_cat[:20]
        max_name_len = max(len(c["name"]) for c in shown)
        max_name_len = max(max_name_len, 8)
        print()
        label = "Per-cat"
        if not args.all_cats and len(per_cat) > 20:
            label += f" (top 20 of {len(per_cat)}, use --all-cats for full list)"
        print(f"{label}:")
        for c in shown:
            name = c["name"].ljust(max_name_len)
            print(
                f"  {name}  {c['ann_count']:>6,} anns   {c['img_count']:>5,} imgs"
            )

    w = s["image_width"]
    h = s["image_height"]
    print()
    print("Image dimensions:")
    print(
        f"  width   min={w['min']:.0f}    max={w['max']:.0f}"
        f"   mean={w['mean']:.1f}  median={w['median']:.1f}"
    )
    print(
        f"  height  min={h['min']:.0f}    max={h['max']:.0f}"
        f"   mean={h['mean']:.1f}  median={h['median']:.1f}"
    )

    a = s["annotation_area"]
    print()
    print("Annotation areas:")
    print(
        f"  min={a['min']:.1f}   max={a['max']:.1f}"
        f"   mean={a['mean']:.1f}   median={a['median']:.1f}"
    )


def main():
    parser = argparse.ArgumentParser(
        prog="coco",
        description="hotcoco command-line tools for COCO datasets",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    stats_parser = subparsers.add_parser(
        "stats",
        help="print dataset health-check statistics",
    )
    stats_parser.add_argument(
        "annotation_file",
        help="path to COCO annotation JSON file",
    )
    stats_parser.add_argument(
        "--all-cats",
        action="store_true",
        help="show all categories instead of top 20",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()
