"""在给定相机姿态 R_ct 下，用本质矩阵平移方向 + 深度变化求光心偏移 t_c（工具系）。"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flip_anything.geometry import pose_matrix  # noqa: E402

RUNS = [Path(p).resolve() for p in sys.argv[1:]]
# 相机 z（光轴）指向工具 -X，相机 x 指向工具 +Z，相机 y 指向工具 +Y
R_PERP = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


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


def collect(R_ct):
    rows = []      # 深度变化方程系数
    rhs = []
    tgt = []       # 本质矩阵方向方程 (A t = 0)
    info = []
    for run in RUNS:
        I = intr(run)
        K = np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1]], float)
        Kinv = np.linalg.inv(K)
        frames = []
        for p in obs(run):
            idx = int(Path(p["camera"]["color_file"]).stem.split("_")[1])
            frames.append({"idx": idx,
                           "g": cv2.imread(str(run / f"frame_{idx:06d}_color.png"), cv2.IMREAD_GRAYSCALE),
                           "pose": np.array(p["pose"], float),
                           "R": np.array(pose_matrix(p["pose"]))[:3, :3],
                           "depth": np.load(run / f"frame_{idx:06d}_depth_raw.npy").astype(np.float64)})
        sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.01)
        feats = [sift.detectAndCompute(f["g"], None) for f in frames]
        for i in range(len(frames) - 1):
            a, b = frames[i], frames[i + 1]
            ka, da = feats[i]
            kb, db = feats[i + 1]
            if da is None or db is None:
                continue
            ms = cv2.BFMatcher().knnMatch(da, db, k=2)
            good = [m for m, n in (pr for pr in ms if len(pr) == 2) if m.distance < 0.8 * n.distance]
            if len(good) < 20:
                continue
            x1 = np.float64([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            x2 = np.float64([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            E, mask = cv2.findEssentialMat(x1, x2, K, method=cv2.RANSAC, prob=0.9999, threshold=1.5)
            if E is None or mask is None or int(mask.sum()) < 12:
                continue
            n_in, R_meas, t_meas, _ = cv2.recoverPose(E, x1, x2, K, mask=mask)
            R_tool = b["R"] @ a["R"].T
            # 方向方程：t_meas ∝ R_ct^T (R_tool - I) R_ct t_tool  =>  skew(t_meas) R_ct^T (R_tool-I) R_ct t = 0
            t = t_meas.reshape(3)
            S = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
            M = (R_tool - np.eye(3))
            tgt.append(S @ (R_ct.T @ M @ R_ct))
            # 深度变化：Δz ≈ n1 · (R_ct^T (R_tool^T - I) R_ct) t
            keep = mask.reshape(-1).astype(bool)
            k1 = x1.reshape(-1, 2)[keep]
            k2 = x2.reshape(-1, 2)[keep]
            z1 = np.array([a["depth"][int(round(v)), int(round(u))] for u, v in k1])
            z2 = np.array([b["depth"][int(round(v)), int(round(u))] for u, v in k2])
            ok = (z1 > 50) & (z1 < 2500) & (z2 > 50) & (z2 < 2500)
            k1, z1, z2 = k1[ok], z1[ok], z2[ok]
            n1 = (Kinv @ np.hstack([k1, np.ones((len(k1), 1))]).T).T
            n1 = n1 / np.linalg.norm(n1, axis=1, keepdims=True)
            Cm = R_ct.T @ (R_tool.T - np.eye(3)) @ R_ct
            for k in range(len(k1)):
                rows.append(n1[k] @ Cm)
                rhs.append(z2[k] - z1[k])
            info.append((run.name, a["idx"], b["idx"], int(mask.sum()),
                         round(math.degrees(math.acos(np.clip((np.trace(R_tool) - 1) / 2, -1, 1))), 1),
                         round(float(np.linalg.norm(b["pose"][:3] - a["pose"][:3])), 1)))
    return np.array(rows), np.array(rhs), np.array(tgt), info


def main():
    rows, rhs, tgt, info = collect(R_PERP)
    print("pairs:")
    for r in info:
        print("   ", r)
    tgt = tgt.reshape(-1, 3)
    A = np.vstack([rows * 1.0, tgt * 8.0])
    b = np.concatenate([rhs, np.zeros(len(tgt))])
    w = np.ones(len(A))
    for _ in range(30):
        sol, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
        resid = A @ sol - b
        s = 1.4826 * np.median(np.abs(resid)) + 1e-9
        w = 1.0 / np.sqrt(1 + (resid / (2.5 * s)) ** 2)
    print("\nt_c (tool frame) =", np.round(sol, 1).tolist(), "|t| =", round(float(np.linalg.norm(sol)), 1))
    print("resid(rms) =", round(float(np.sqrt((resid ** 2).mean())), 3))
    print("singular values:", np.round(np.linalg.svd(A, compute_uv=False), 4).tolist())
    (RUNS[0] / "offset_perp.json").write_text(json.dumps(
        {"R_tcp_cam": R_PERP.tolist(), "t_tcp_cam_mm": sol.tolist()}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()