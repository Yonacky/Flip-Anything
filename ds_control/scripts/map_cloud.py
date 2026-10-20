"""把最近一帧的深度点云投到基座系俯视图与侧视图，用于判读场景布局。"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flip_anything.geometry import pose_matrix  # noqa: E402

RUN = Path(sys.argv[1]).resolve()
FRAME = int(sys.argv[2]) if len(sys.argv) > 2 else -1
TCAM = float(sys.argv[3]) if len(sys.argv) > 3 else 110.0


def events():
    return [json.loads(l) for l in (RUN / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def intr():
    for e in events():
        if e["kind"] == "camera_started":
            return e["payload"]["metadata"]["factory_camera_param"]["color_intrinsic"]
    raise RuntimeError("no camera_started")


def obs():
    out, last = [], None
    for e in events():
        if e["kind"] == "robot_sample":
            last = e["payload"]
        elif e["kind"] == "observation" and e["payload"].get("camera"):
            p = dict(e["payload"])
            p["pose"] = list(last["pose"])
            out.append(p)
    return out


I = intr()
obs_list = obs()
p = obs_list[FRAME]
idx = int(Path(p["camera"]["color_file"]).stem.split("_")[1])
pose = np.array(p["pose"], float)
R = np.array(pose_matrix(pose))[:3, :3]
C = pose[:3] + R @ np.array([0.0, 0.0, TCAM])
depth = np.load(RUN / f"frame_{idx:06d}_depth_raw.npy").astype(np.float64)
h, w = depth.shape
gu, gv = np.meshgrid(np.arange(0, w, 4), np.arange(0, h, 4))
z = depth[gv, gu]
m = (z > 1) & (z < 4000)
u, v, zz = gu[m].astype(float), gv[m].astype(float), z[m]
Xc = np.stack([(u - I["cx"]) * zz / I["fx"], (v - I["cy"]) * zz / I["fy"], zz], axis=1)
Xb = (R @ Xc.T).T + C
print("frame", idx, "pose", np.round(pose, 2).tolist())
print("camera center", np.round(C, 1).tolist())
print("point count", len(Xb))
for name, ax, ay, lim in (("top_xy", 0, 1, 2000.0), ("side_xz", 0, 2, 2000.0), ("side_yz", 1, 2, 2000.0)):
    canvas = np.full((800, 800, 3), 255, np.uint8)
    px = ((Xb[:, ax] + lim) / (2 * lim) * 799).astype(int)
    py = (799 - (Xb[:, ay] + lim) / (2 * lim) * 799).astype(int)
    ok = (px >= 0) & (px < 800) & (py >= 0) & (py < 800)
    shade = np.clip(255 - Xb[ok, 2] / 4000 * 255, 0, 255).astype(np.uint8)
    canvas[py[ok], px[ok], 0] = shade
    canvas[py[ok], px[ok], 1] = shade
    canvas[py[ok], px[ok], 2] = 255
    # 基座原点
    ox = int((0 + lim) / (2 * lim) * 799)
    oy = int(799 - (0 + lim) / (2 * lim) * 799)
    cv2.drawMarker(canvas, (ox, oy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(canvas, f"{name} lim+-{lim:.0f}mm frame{idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(str(RUN / f"map_{name}_{idx:06d}.png"), canvas)
    print("wrote map_%s_%06d.png" % (name, idx))