"""针对当前帧：近处桌面平面、罐体几何、标定板可见性。"""
import sys
from pathlib import Path
import cv2
import numpy as np

run = Path(r"G:/Flip Anything/logs/experiments/legacy/ds_flip_20261014_01")
frame = sys.argv[1] if len(sys.argv) > 1 else "frame_000002"
K = {"fx": 691.4324951171875, "fy": 691.7568359375, "cx": 643.6455078125, "cy": 358.8835144042969}

color = cv2.imread(str(run / (frame + "_color.png")), cv2.IMREAD_COLOR)
depth = np.load(run / (frame + "_depth_raw.npy")).astype(np.float32)
h, w = depth.shape
vv, uu = np.mgrid[0:h, 0:w]
valid = depth > 0
Z = depth
X = (uu - K["cx"]) * Z / K["fx"]
Y = (vv - K["cy"]) * Z / K["fy"]

# 罐体
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
green = cv2.inRange(hsv, (35, 60, 50), (95, 255, 255))
green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
n, lab, stats, cent = cv2.connectedComponentsWithStats(green)
i = 1 + int(np.argmax(stats[1:, 4]))
x, y, bw, bh, area = stats[i]
m = (lab == i) & valid
print("CAN bbox x[%d..%d] y[%d..%d] area %d" % (x, x + bw - 1, y, y + bh - 1, area))
print("CAN 3D centroid", np.round([X[m].mean(), Y[m].mean(), Z[m].mean()], 1))
print("CAN depth p5/p50/p95", np.percentile(Z[m], [5, 50, 95]).round(1))
# 每行的水平宽度 -> 圆柱侧面；估计轴（列方向最小二乘）
rows = np.unique(np.nonzero(m)[0])
rowc = np.array([[np.nonzero(m[r])[0].mean(), r] for r in rows if m[r].sum() > 8])
if len(rowc) > 10:
    A = np.polyfit(rowc[:, 1], rowc[:, 0], 1)
    print("CAN image axis: u = %.4f*v + %.1f (slope deg %.2f)" % (A[0], A[1], np.degrees(np.arctan(A[0]))))
    top, bot = rowc[0], rowc[-1]
    print("CAN visible top row %d (u=%.0f), bottom row %d (u=%.0f)" % (top[1], top[0], bot[1], bot[0]))
    for r in (int(top[1]) + 2, int((top[1] + bot[1]) / 2), int(bot[1]) - 2):
        cols = np.nonzero(m[r])[0]
        zz = Z[r, cols]
        zz = zz[zz > 0]
        print("  row %d: u %d..%d  width_px %d  depth_med %.1f" % (r, cols.min(), cols.max(), cols.max() - cols.min(), np.median(zz)))

# 近处桌面：图像下半部 + 深度 < 1200
near = valid & (Z < 1200) & (vv > 560)
pts = np.stack([X[near], Y[near], Z[near]], 1)
print("near-table candidate pixels", len(pts))
rng = np.random.default_rng(2)
best = None
S = pts[rng.choice(len(pts), min(40000, len(pts)), replace=False)]
for _ in range(2000):
    a, b, c = S[rng.choice(len(S), 3, replace=False)]
    nv = np.cross(b - a, c - a)
    L = np.linalg.norm(nv)
    if L < 1e-6:
        continue
    nv = nv / L
    d = -nv @ a
    cnt = int((np.abs(S @ nv + d) < 2.0).sum())
    if best is None or cnt > best[0]:
        best = (cnt, nv, d)
cnt, nv, d = best
if nv[1] > 0:
    nv, d = -nv, -d
print("near table plane: n", nv.round(4), "d %.2f mm  inliers %d" % (d, cnt))
print("  camera height above this plane: %.1f mm" % abs(d))

# 标定板区域（右下方白色高亮纸面）
paper = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 150) & (uu > 800) & (vv > 430) & valid
if paper.sum() > 500:
    print("BOARD-like pixels", int(paper.sum()),
          "3D centroid", np.round([X[paper].mean(), Y[paper].mean(), Z[paper].mean()], 1),
          "depth p5/p50/p95", np.percentile(Z[paper], [5, 50, 95]).round(1),
          "u range", int(uu[paper].min()), int(uu[paper].max()),
          "v range", int(vv[paper].min()), int(vv[paper].max()))

gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
for size in ((9, 6), (8, 6), (7, 6), (7, 5), (6, 5), (6, 4), (5, 4), (4, 3), (11, 7), (10, 7), (9, 7), (12, 9), (11, 8)):
    ok, c = cv2.findChessboardCorners(gray, size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok:
        c = c.reshape(-1, 2)
        zz = Z[np.clip(c[:, 1], 0, h - 1).astype(int), np.clip(c[:, 0], 0, w - 1).astype(int)]
        print("CHESS", size, "corners", len(c), "u range", int(c[:, 0].min()), int(c[:, 0].max()),
              "v range", int(c[:, 1].min()), int(c[:, 1].max()),
              "depth valid", int((zz > 0).sum()), "median", round(float(np.median(zz[zz > 0])), 1) if (zz > 0).any() else None)
