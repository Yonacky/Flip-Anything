"""Read-only: locate the green target can in saved frames and report its image geometry."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def analyse(run, index):
    folder = ROOT.parent / 'logs' / 'experiments' / 'ds_control' / run
    img = cv2.imread(str(folder / ("frame_%06d_color.png" % index)))
    depth = np.load(folder / ("frame_%06d_depth_raw.npy" % index)).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 60, 50), (90, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA]) if n > 1 else None
    res = {"frame": index}
    if best is not None:
        comp = labels == best
        x, y, w, h, area = stats[best]
        res.update(area=int(area), bbox=[int(x), int(y), int(w), int(h)],
                   centroid=[round(float(cents[best][0]), 1), round(float(cents[best][1]), 1)],
                   fill=round(area / float(w * h), 3))
        d = depth[comp]
        dv = d[d > 0]
        if dv.size:
            res["depth_med_on_can"] = round(float(np.median(dv)), 1)
            res["depth_valid_frac"] = round(float(dv.size / comp.sum()), 3)
        # 罐顶边缘：该连通域最上方一行
        ys = np.where(comp.any(axis=1))[0]
        res["top_row"] = int(ys[0])
        res["bottom_row"] = int(ys[-1])
        row = np.where(comp[ys[0]])[0]
        res["top_row_span_px"] = [int(row[0]), int(row[-1])]
    print(json.dumps(res, ensure_ascii=False))
    return res


if __name__ == "__main__":
    run = sys.argv[1]
    for idx in [int(a) for a in sys.argv[2:]]:
        analyse(run, idx)
