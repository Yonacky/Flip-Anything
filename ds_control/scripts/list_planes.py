"""列出每帧前几个平面（含基座系法向），并在彩色图上标出内点，便于判读桌面。"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flip_anything.geometry import pose_matrix  # noqa: E402

RUN = Path(sys.argv[1]).resolve()
FRAMES = [int(v) for v in sys.argv[2:]] or [6]


def events():
    return [json.loads(l) for l in (RUN / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def intr():
    for e in events():
        if e["kind"] == "camera_started":
            return e["payload"]["metadata"]["factory_camera_param"]["color_intrinsic"]
    raise RuntimeError("no camera_started")


def pose_of(idx):
    last = None
    for e in events():
        if e["kind"] == "robot_sample":
            last = e["payload"]
        elif e["kind"] == "observation" and e["payload"].get("camera"):
            if Path(e["payload"]["camera"]["color_file"]).stem == f"frame_{idx:06d}_color":
                return list(last["pose"])
    return None


I = intr()
for idx in FRAMES:
    pose = pose_of(idx)
    R = np.array(pose_matrix(pose))[:3, :3]
    depth = np.load(RUN / f"frame_{idx:06d}_depth_raw.npy").astype(np.float64)
    bgr = cv2.imread(str(RUN / f"frame_{idx:06d}_color.png"), cv2.IMREAD_COLOR)
    h, w = depth.shape
    gu, gv = np.meshgrid(np.arange(0, w, 3), np.arange(0, h, 3))
    z = depth[gv, gu]
    m = (z > 1) & (z < 6000)
    u, v, zz = gu[m].astype(float), gv[m].astype(float), z[m]
    pts = np.stack([(u - I["cx"]) * zz / I["fx"], (v - I["cy"]) * zz / I["fy"], zz], axis=1)
    rng = np.random.default_rng(11)
    found = []
    remaining = pts.copy()
    for _ in range(6):
        best = None
        for _ in range(2500):
            i3 = rng.choice(len(remaining), 3, replace=False)
            p0, p1, p2 = remaining[i3]
            nrm = np.cross(p1 - p0, p2 - p0)
            ln = np.linalg.norm(nrm)
            if ln < 1e-6:
                continue
            nrm = nrm / ln
            d = -nrm @ p0
            dist = np.abs(remaining @ nrm + d)
            inl = dist < 6.0
            c = int(inl.sum())
            if best is None or c > best[0]:
                best = (c, nrm, d, inl)
        if best is None or best[0] < 150:
            break
        c, nrm, d, inl = best
        sel = remaining[inl]
        cen = sel.mean(axis=0)
        _, _, vh = np.linalg.svd(sel - cen)
        nrm = vh[-1]
        d = -nrm @ cen
        dist = np.abs(pts @ nrm + d)
        inl_all = dist < 6.0
        nb = R @ nrm
        found.append({"n": int(inl_all.sum()), "normal_base": [round(float(x), 3) for x in nb],
                      "plane_dist_mm": round(float(abs(d)), 1),
                      "centroid_cam": [round(float(x), 1) for x in cen],
                      "depth_range_mm": [round(float(sel[:, 2].min()), 1), round(float(sel[:, 2].max()), 1)]})
        remaining = remaining[~inl]
        if len(remaining) < 200:
            break
    print(f"--- frame {idx} pose {[round(v,2) for v in pose]} ---")
    for f in found:
        print("   ", json.dumps(f))
    # 标注第一个平面
    vis = bgr.copy()
    if found:
        nrm = None
        # 重新取第一个平面的内点做可视化
        best = None
        for _ in range(4000):
            i3 = rng.choice(len(pts), 3, replace=False)
            p0, p1, p2 = pts[i3]
            nn = np.cross(p1 - p0, p2 - p0)
            ln = np.linalg.norm(nn)
            if ln < 1e-6:
                continue
            nn = nn / ln
            dd = -nn @ p0
            dist = np.abs(pts @ nn + dd)
            inl = dist < 6.0
            c = int(inl.sum())
            if best is None or c > best[0]:
                best = (c, nn, dd, inl)
        sel = best[3]
        vis[gv[m][sel], gu[m][sel]] = (0, 0, 255)
        cv2.imwrite(str(RUN / f"diag_plane_{idx:06d}.png"), vis)
        print("    wrote diag_plane_%06d.png (top plane inliers in red)" % idx)