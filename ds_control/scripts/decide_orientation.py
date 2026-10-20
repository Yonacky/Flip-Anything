"""判决性检验：不同相机安装姿态下，用桌面平面约束解光心偏移，
正确姿态应使各帧的桌面平面在基座系中落在同一高度（残差最小）。"""
import json
import math
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


def plane_near(pts, uv, tol=5.0, iters=3000, seed=3, min_nz=None, R=None):
    rng = np.random.default_rng(seed)
    best = None
    n = len(pts)
    for _ in range(iters):
        i3 = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[i3]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        if min_nz is not None and R is not None and abs((R @ nrm)[2]) < min_nz:
            continue
        d = -nrm @ p0
        dist = np.abs(pts @ nrm + d)
        inl = dist < tol
        c = int(inl.sum())
        if best is None or c > best[0]:
            best = (c, nrm, d, inl)
    if best is None or best[0] < 200:
        return None
    c, nrm, d, inl = best
    for _ in range(4):
        sel = pts[inl]
        cen = sel.mean(axis=0)
        _, _, vh = np.linalg.svd(sel - cen)
        nrm = vh[-1]
        d = -nrm @ cen
        dist = np.abs(pts @ nrm + d)
        inl = dist < tol
    return {"n": int(inl.sum()), "normal_cam": nrm, "d": float(d), "inliers": inl,
            "rms": float(np.sqrt((np.abs(pts @ nrm + d)[inl] ** 2).mean()))}


def main():
    I = intr()
    frames = []
    for p in obs():
        stem = Path(p["camera"]["color_file"]).stem
        idx = int(stem.split("_")[1])
        depth = np.load(RUN / f"frame_{idx:06d}_depth_raw.npy").astype(np.float64)
        frames.append({"idx": idx, "pose": np.array(p["pose"], float),
                       "R": np.array(pose_matrix(p["pose"]))[:3, :3], "depth": depth})
    K = I
    planes = []
    for f in frames:
        depth = f["depth"]
        h, w = depth.shape
        gu, gv = np.meshgrid(np.arange(0, w, 3), np.arange(0, h, 3))
        z = depth[gv, gu]
        m = (z > 1) & (z < 3000)
        u, v, zz = gu[m].astype(float), gv[m].astype(float), z[m]
        pts = np.stack([(u - K["cx"]) * zz / K["fx"], (v - K["cy"]) * zz / K["fy"], zz], axis=1)
        # 近处优先：按深度分位裁剪
        lo, hi = np.percentile(pts[:, 2], [0, 45])
        near = pts[(pts[:, 2] >= lo) & (pts[:, 2] <= hi)]
        res = plane_near(near, None, R=f["R"], min_nz=0.75)
        if res is None:
            res = plane_near(near, None)
        planes.append(res)
        print(f"frame {f['idx']}: " + (f"n={res['n']} rms={res['rms']:.2f} normal_cam={np.round(res['normal_cam'],3).tolist()} d={res['d']:.1f}" if res else "no plane"))
    cands = {}
    for name, path, key in (("sift", RUN / "camera_rotation.json", "R_tcp_cam"),
                            ("epipolar", RUN / "camera_rotation_epipolar.json", "R_tcp_cam"),
                            ("identity", None, None)):
        if path is None:
            cands[name] = np.eye(3)
        elif path.exists():
            cands[name] = np.array(json.loads(path.read_text(encoding="utf-8"))[key])
    variants = {}
    for name, R_ct in cands.items():
        variants[name] = R_ct
        variants[name + "_T"] = R_ct.T
    idxs = [i for i, p in enumerate(planes) if p is not None]
    for name, R_ct in variants.items():
        A, b = [], []
        for i in idxs:
            f = frames[i]
            n_cam = planes[i]["normal_cam"]
            d_cam = planes[i]["d"]
            n_b = f["R"] @ n_cam
            if n_b[2] > 0:
                n_b, d_cam = -n_b, -d_cam
            # (n_b^T R_i) t_c 约束
            A.append(n_b @ f["R"])
            b.append(-d_cam - n_b @ f["pose"][:3])
        A = np.array(A); b = np.array(b)
        if len(A) < 3:
            print(name, "insufficient planes")
            continue
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid = A @ sol - b
        print(f"{name:16s} n_planes={len(A)} rms_resid={np.sqrt((resid**2).mean()):8.2f} mm  "
              f"t_c={np.round(sol,1).tolist()} |t|={np.linalg.norm(sol):.1f}  "
              f"h(基座高度)={np.mean([A[k] @ sol + b[k] + 2*0 for k in range(len(A))]):.0f}")
        # 桌面高度：h_b = n_b^T (p_i + R_i t_c)
        hs = [float((frames[i]['R'] @ planes[i]['normal_cam'] if False else planes[i]['normal_cam']) @ np.zeros(3)) for i in idxs]
    out = {}
    print("\n各候选下的平面法向一致性（基座系 z 分量应接近 ±1）：")
    for i in idxs:
        f = frames[i]
        row = [f["idx"]]
        for name in ("identity", "sift", "epipolar"):
            R_ct = variants[name]
            R_cam = f["R"] @ R_ct
            n_cam = planes[i]["normal_cam"]
            if n_cam @ (R_cam[:, 2]) > 0:
                n_cam = -n_cam
            nb = R_cam @ n_cam
            row.append(f"{name}:{np.round(nb,2).tolist()}")
        print("  ", row)


if __name__ == "__main__":
    main()