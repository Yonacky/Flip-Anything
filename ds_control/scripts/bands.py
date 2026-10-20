"""平面检测 + 相机安装姿态判决：
对每帧在近/中/远深度带各拟合平面，然后对候选相机姿态检查
「同一物理面（桌面）在各帧的基座系高度是否一致」。"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flip_anything.geometry import pose_matrix  # noqa: E402

RUN = Path(sys.argv[1]).resolve()


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


def fit(points, tol=5.0, iters=1500, seed=1):
    rng = np.random.default_rng(seed)
    best = None
    n = len(points)
    if n < 100:
        return None
    for _ in range(iters):
        i3 = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[i3]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        d = -nrm @ p0
        dist = np.abs(points @ nrm + d)
        inl = dist < tol
        c = int(inl.sum())
        if best is None or c > best[0]:
            best = (c, nrm, d, inl)
    if best is None or best[0] < 80:
        return None
    c, nrm, d, inl = best
    for _ in range(3):
        sel = points[inl]
        cen = sel.mean(axis=0)
        _, _, vh = np.linalg.svd(sel - cen)
        nrm = vh[-1]
        d = -nrm @ cen
        inl = np.abs(points @ nrm + d) < tol
    return {"n": int(inl.sum()), "normal": nrm, "d": float(d),
            "rms": float(np.sqrt((np.abs(points @ nrm + d)[inl] ** 2).mean())),
            "centroid": points[inl].mean(axis=0)}


def main():
    I = intr()
    frames = []
    for p in obs():
        stem = Path(p["camera"]["color_file"]).stem
        idx = int(stem.split("_")[1])
        frames.append({"idx": idx, "pose": np.array(p["pose"], float),
                       "R": np.array(pose_matrix(p["pose"]))[:3, :3],
                       "depth": np.load(RUN / f"frame_{idx:06d}_depth_raw.npy").astype(np.float64)})
    Rct = None
    for name in ("camera_rotation.json", "camera_rotation_epipolar.json"):
        p = RUN / name
        if p.exists():
            try:
                Rct = np.array(json.loads(p.read_text(encoding="utf-8"))["R_tcp_cam"])
                print("using rotation from", name)
                break
            except Exception:
                pass
    if Rct is None:
        Rct = np.eye(3)
        print("using identity rotation")
    planes = {}
    for f in frames:
        depth = f["depth"]
        h, w = depth.shape
        gu, gv = np.meshgrid(np.arange(0, w, 3), np.arange(0, h, 3))
        z = depth[gv, gu]
        m = (z > 1) & (z < 4000)
        u, v, zz = gu[m].astype(float), gv[m].astype(float), z[m]
        pts = np.stack([(u - I["cx"]) * zz / I["fx"], (v - I["cy"]) * zz / I["fy"], zz], axis=1)
        row = []
        for lo, hi in ((150, 450), (450, 900), (900, 1800)):
            band = pts[(pts[:, 2] >= lo) & (pts[:, 2] < hi)]
            if len(band) < 300:
                continue
            res = fit(band)
            if res is None:
                continue
            n_cam = res["normal"]
            nb = f["R"] @ Rct @ n_cam
            if nb[2] < 0:
                nb = -nb
            row.append({"band": [lo, hi], "n": res["n"], "rms": round(res["rms"], 2),
                        "normal_base": [round(float(x), 3) for x in nb],
                        "dist_along_axis_mm": round(float(abs(res["d"])), 1),
                        "centroid_cam": [round(float(x), 0) for x in res["centroid"]]})
        planes[f["idx"]] = row
        print(f"frame {f['idx']} pose {np.round(f['pose'],1).tolist()}")
        for r in row:
            print("   ", json.dumps(r))
    (RUN / "plane_bands.json").write_text(json.dumps(planes, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()