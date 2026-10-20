"""用手腕扫描时检测到的同一物理水平面（远带天花板）在不同帧的约束，
联合解出相机→TCP 的旋转候选与光心偏移 t_c（平面高度作为未知量一并求解）。"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flip_anything.geometry import pose_matrix  # noqa: E402

RUNS = [Path(p).resolve() for p in sys.argv[1:]]


def events(run):
    return [json.loads(l) for l in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def intr(run):
    for e in events(run):
        if e["kind"] == "camera_started":
            return e["payload"]["metadata"]["factory_camera_param"]["color_intrinsic"]
    raise RuntimeError("no camera_started")


def obs(run):
    out, last = [], None
    for e in events(run):
        if e["kind"] == "robot_sample":
            last = e["payload"]
        elif e["kind"] == "observation" and e["payload"].get("camera"):
            p = dict(e["payload"])
            p["pose"] = list(last["pose"])
            out.append(p)
    return out


def plane_far(points, tol=6.0, iters=2000, seed=3):
    """在 900-2500mm 深度带内拟合平面，返回法向/距离/内点数。"""
    rng = np.random.default_rng(seed)
    n = len(points)
    if n < 200:
        return None
    best = None
    for _ in range(iters):
        i3 = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[i3]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        d = -nrm @ p0
        inl = np.abs(points @ nrm + d) < tol
        c = int(inl.sum())
        if best is None or c > best[0]:
            best = (c, nrm, d, inl)
    if best is None or best[0] < 150:
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
            "rms": float(np.sqrt((np.abs(points @ nrm + d)[inl] ** 2).mean()))}


def collect():
    per_frame = []
    for run in RUNS:
        I = intr(run)
        for p in obs(run):
            idx = int(Path(p["camera"]["color_file"]).stem.split("_")[1])
            depth = np.load(run / f"frame_{idx:06d}_depth_raw.npy").astype(np.float64)
            h, w = depth.shape
            gu, gv = np.meshgrid(np.arange(0, w, 3), np.arange(0, h, 3))
            z = depth[gv, gu]
            m = (z > 900) & (z < 3000)
            if m.sum() < 400:
                continue
            u, v, zz = gu[m].astype(float), gv[m].astype(float), z[m]
            pts = np.stack([(u - I["cx"]) * zz / I["fx"], (v - I["cy"]) * zz / I["fy"], zz], axis=1)
            res = plane_far(pts)
            if res is None:
                continue
            per_frame.append({"run": run.name, "idx": idx, "pose": np.array(p["pose"], float),
                              "R": np.array(pose_matrix(p["pose"]))[:3, :3], "plane": res})
    return per_frame


def solve(per_frame, R_ct, label):
    A, b = [], []
    for f in per_frame:
        R_cam = f["R"] @ R_ct
        n_cam = f["plane"]["normal"]
        d_cam = f["plane"]["d"]
        n_b = R_cam @ n_cam
        # 平面在基座系：n_b·X = h ；相机光心 C = p + R_cam t_c 在平面上方 h
        # 约束: n_b·(p + R_cam t_c) = h  =>  n_b·(R_cam t_c) - h = -n_b·p
        A.append(np.concatenate([n_b @ R_cam, [-1.0]]))
        b.append(-float(n_b @ f["pose"][:3]))
    A = np.array(A); b = np.array(b)
    if len(A) < 4:
        return None
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = A @ sol - b
    return {"label": label, "n_frames": len(A), "t_c": sol[:3].tolist(), "h_mm": float(sol[3]),
            "rms_mm": float(np.sqrt((resid ** 2).mean())), "max_mm": float(np.abs(resid).max()),
            "sing": np.linalg.svd(A, compute_uv=False).tolist(), "resid": resid.tolist()}


def main():
    per_frame = collect()
    print("frames with far plane:", [(f["run"][-8:], f["idx"], f["plane"]["n"]) for f in per_frame])
    cands = {}
    for run in RUNS:
        for name, key in (("camera_rotation.json", "R_tcp_cam"), ("camera_rotation_epipolar.json", "R_tcp_cam")):
            p = run / name
            if p.exists():
                cands[f"{run.name[-8:]}:{name}"] = np.array(json.loads(p.read_text(encoding="utf-8"))[key])
    cands["identity"] = np.eye(3)
    for name, R_ct in cands.items():
        r = solve(per_frame, R_ct, name)
        if r is None:
            print(name, "insufficient frames")
            continue
        print(f"{name:34s} frames={r['n_frames']} rms={r['rms_mm']:8.2f} mm  max={r['max_mm']:8.2f}  "
              f"t_c={np.round(r['t_c'],1).tolist()} |t|={np.linalg.norm(r['t_c']):6.1f}  h={r['h_mm']:.0f} mm  "
              f"sing_min={min(r['sing']):.3f}")
    (RUNS[0] / "plane_solution.json").write_text(json.dumps(
        {name: solve(per_frame, R_ct, name) for name, R_ct in cands.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()