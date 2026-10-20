"""Read-only: report measured depth on the target region and nearest valid surface."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
run, index = sys.argv[1], int(sys.argv[2])
folder = ROOT.parent / 'logs' / 'experiments' / 'ds_control' / run
img = cv2.imread(str(folder / ("frame_%06d_color.png" % index)))
depth = np.load(folder / ("frame_%06d_depth_raw.npy" % index)).astype(np.float32)
h, w = depth.shape
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
green = cv2.inRange(hsv, (35, 60, 50), (90, 255, 255)) > 0
green = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_OPEN, np.ones((11, 11), np.uint8)) > 0
green = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8)) > 0
n, labels, stats, cents = cv2.connectedComponentsWithStats(green.astype(np.uint8), 8)
best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
comp = labels == best
x, y, ww, hh, area = stats[best]
print("can comp bbox=(%d,%d,%dx%d) area=%d centroid=(%.0f,%.0f)" % (x, y, ww, hh, area, cents[best][0], cents[best][1]))
d = depth[comp]
dv = d[d > 0]
print("depth on can: valid %d/%d" % (dv.size, comp.sum()))
if dv.size:
    print("   p10=%.0f p50=%.0f p90=%.0f" % tuple(np.percentile(dv, q) for q in (10, 50, 90)))
# 罐体中心竖线上的深度剖面
cx = int(cents[best][0])
print("vertical profile at x=%d:" % cx)
for py in range(max(0, y), min(h, y + hh), 40):
    z = depth[py, cx]
    print("   y=%3d depth=%s" % (py, "invalid" if z <= 0 else "%.0f" % z))
# 图像中最近的有效表面
valid = depth > 100
if valid.any():
    yy, xx = np.where(valid)
    zz = depth[valid]
    order = np.argsort(zz)[:400]
    print("nearest valid surface points (top 400 median): %.0f mm" % np.median(zz[order]))
    print("   their image region x %d..%d y %d..%d" % (xx[order].min(), xx[order].max(), yy[order].min(), yy[order].max()))
    # 排除罐体区域后再看最近表面
    mask2 = valid & (~comp)
    if mask2.any():
        z2 = depth[mask2]
        o2 = np.argsort(z2)[:800]
        yy2, xx2 = np.where(mask2)
        print("nearest non-can surface median: %.0f mm at x %d..%d y %d..%d"
              % (np.median(z2[o2]), xx2[o2].min(), xx2[o2].max(), yy2[o2].min(), yy2[o2].max()))
