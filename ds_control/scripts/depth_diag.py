"""深度诊断：分块深度统计 + 点云投影图 + 深度栅格着色图（不依赖 matplotlib）。"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flip_anything.geometry import pose_matrix  # noqa: E402

RUN = Path(sys.argv[1]).resolve()
FRAME = int(sys.argv[2])


def intrinsics(run):
    for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e["kind"] == "camera_started":
            return e["payload"]["metadata"]["factory_camera_param"]["color_intrinsic"]
    raise RuntimeError("no camera_started")


def pose_of(run, frame):
    last = None
    for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e["kind"] == "robot_sample":
            last = e["payload"]
        elif e["kind"] == "observation" and e["payload"].get("camera"):
            if Path(e["payload"]["camera"]["color_file"]).stem.endswith(f"{frame:06d}"):
                return last["pose"]
    return None


intr = intrinsics(RUN)
bgr = cv2.imread(str(RUN / f"frame_{FRAME:06d}_color.png"), cv2.IMREAD_COLOR)
depth = np.load(RUN / f"frame_{FRAME:06d}_depth_raw.npy").astype(np.float64)
pose = pose_of(RUN, FRAME)
h, w = depth.shape
print("pose", pose)
print("depth stats: valid", float(((depth > 1) & (depth < 6000)).mean()))
grid = []
for r0 in range(0, h, 120):
    row = []
    for c0 in range(0, w, 160):
        blk = depth[r0:r0 + 120, c0:c0 + 160]
        m = (blk > 1) & (blk < 6000)
        row.append(None if m.sum() < 50 else int(np.median(blk[m])))
    grid.append(row)
print("median depth per 160x120 block (mm), rows top->bottom:")
for i, row in enumerate(grid):
    print(f"  y={i*120:3d}", row)

# 深度着色图
vis = np.zeros_like(bgr)
d8 = np.clip(depth / 4000.0, 0, 1)
vis[..., 1] = (255 * (1 - d8)).astype(np.uint8)
vis[..., 2] = (255 * d8).astype(np.uint8)
vis[(depth <= 1) | (depth >= 6000)] = 0
cv2.imwrite(str(RUN / f"diag_depth_{FRAME:06d}.png"), vis)

# 点云投影：相机系 X-Z 侧视 + X-Y 俯视
gu, gv = np.meshgrid(np.arange(0, w, 4), np.arange(0, h, 4))
z = depth[gv, gu]
m = (z > 1) & (z < 6000)
u = gu[m].astype(float); v = gv[m].astype(float); zz = z[m]
X = (u - intr["cx"]) * zz / intr["fx"]
Y = (v - intr["cy"]) * zz / intr["fy"]
Z = zz
for name, (ax, ay) in {"xz": (X, Z), "xy": (X, Y)}.items():
    canvas = np.full((600, 1200, 3), 255, np.uint8)
    lim = 3000.0
    px = ((ax + lim) / (2 * lim) * 1199).astype(int)
    py = (599 - (ay + lim) / (2 * lim) * 599).astype(int)
    ok = (px >= 0) & (px < 1200) & (py >= 0) & (py < 600)
    canvas[py[ok], px[ok]] = (0, 0, 0)
    cv2.imwrite(str(RUN / f"diag_cloud_{name}_{FRAME:06d}.png"), canvas)
print("wrote diagnostics into", RUN)