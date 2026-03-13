"""Generate an evaluation report PDF for COCO val2017."""

import argparse

import hotcoco
from hotcoco.plot import report

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="bbox", choices=["bbox", "segm", "kpt"])
parser.add_argument("--out", default="report.pdf")
args = parser.parse_args()

if args.type == "kpt":
    gt_path = "data/annotations/person_keypoints_val2017.json"
else:
    gt_path = "data/annotations/instances_val2017.json"
dt_path = f"data/{args.type}_val2017_results.json"

gt = hotcoco.COCO(gt_path)
dt = gt.load_res(dt_path)
ev = hotcoco.COCOeval(gt, dt, args.type)
ev.run()
report(ev, save_path=args.out, gt_path=gt_path, dt_path=dt_path)
print(f"Saved {args.out}")
