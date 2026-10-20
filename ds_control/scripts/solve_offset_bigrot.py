"""大角度纯旋转下的光心偏移求解：深度变化法 + 本质矩阵平移方向法。"""
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


def main():
    I = intr()
    K = np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1]], float)
    frames = []
    for p in obs():
        stem = Path(p["camera"]["color_file"]).stem
        idx = int(stem.split("_")[1])
        g = cv2.imread(str(RUN / f"frame_{idx:06d}_color.png"), cv2.IMREAD_GRAYSCALE)
        frames.append({"idx": idx, "g": g, "pose": np.array(p["pose"], float),
                       "R": np.array(pose_matrix(p["pose"]))[:3, :3],
                       "depth": np.load(RUN / f"frame_{idx:06d}_depth_raw.npy").astype(np.float64)})
    base = frames[0]
    print("base pose", np.round(base["pose"], 2).tolist())
    A_rows, b_vals = [], []
    em_rows = []
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.008)
    kb, db = sift.detectAndCompute(base["g"], None)
    for f in frames[1:]:
        dpose = f["pose"][:3] - base["pose"][:3]
        if np.linalg.norm(dpose) > 3.0:
            print(f"skip frame {f['idx']}: translation {np.linalg.norm(dpose):.1f} mm")
            continue
        kf, df = sift.detectAndCompute(f["g"], None)
        matches = cv2.BFMatcher().knnMatch(db, df, k=2)
        good = [m for m, n in (pr for pr in matches if len(pr) == 2) if m.distance < 0.8 * n.distance]
        x1 = np.float64([kb[m.queryIdx].pt for m in good]).reshape(-1, 2)
        x2 = np.float64([kf[m.trainIdx].pt for m in good]).reshape(-1, 2)
        R_tool = f["R"] @ base["R"].T
        M = R_tool.T - np.eye(3)
        used = 0
        for (u1, v1), (u2, v2) in zip(x1, x2):
            z1 = base["depth"][int(round(v1)), int(round(u1))]
            z2 = f["depth"][int(round(v2)), int(round(u2))]
            if not (30 < z1 < 3000 and 30 < z2 < 3000):
                continue
            n1 = np.array([(u1 - I["cx"]) / I["fx"], (v1 - I["cy"]) / I["fy"], 1.0])
            n1 = n1 / np.linalg.norm(n1)
            A_rows.append(n1 @ M)
            b_vals.append(z2 - z1)
            used += 1
        # 本质矩阵方向法
        if len(x1) >= 15:
            E, mask = cv2.findEssentialMat(x1.reshape(-1, 1, 2), x2.reshape(-1, 1, 2), K,
                                           method=cv2.RANSAC, prob=0.9999, threshold=1.5)
            if E is not None and mask is not None:
                n_in, R_rel, t_rel, _ = cv2.recoverPose(E, x1.reshape(-1, 1, 2), x2.reshape(-1, 1, 2), K, mask=mask)
                t = t_rel.reshape(3)
                S = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
                # 期望 t_rel ∝ R_ct^T (R_tool^T (C_f - C_base)) = R_ct^T M t_c  (M = R_tool^T - I)
                # 用 t_rel 的 3 个分量作为观测量，未知 t_c：R_ct^T M t_c = λ t_rel
                # 先忽略 λ，用叉乘：skew(t_rel) R_ct^T M t_c = 0
                Rct = np.array(json.loads((RUN / "camera_rotation.json").read_text(encoding="utf-8"))["R_tcp_cam"]) \
                    if (RUN / "camera_rotation.json").exists() else np.eye(3)
                em_rows.append(S @ (Rct.T @ M))
                print(f"frame {f['idx']}: matches={len(x1)} depth_rows={used} inliers={int(mask.sum())} "
                      f"angle={math.degrees(math.acos(np.clip((np.trace(R_tool)-1)/2,-1,1))):.1f} t_rel={np.round(t,3).tolist()}")
    A = np.array(A_rows); b = np.array(b_vals)
    print("depth rows:", len(A))
    if len(A) >= 6:
        w = np.ones(len(A))
        for _ in range(30):
            sol, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
            resid = A @ sol - b
            s = 1.4826 * np.median(np.abs(resid)) + 1e-6
            w = 1.0 / np.sqrt(1 + (resid / (2.0 * s)) ** 2)
        print("depth-change solution t_c =", np.round(sol, 2).tolist(), "|t| =", round(float(np.linalg.norm(sol)), 2))
        print("  residual median/rms mm:", round(float(np.median(np.abs(resid))), 3), round(float(np.sqrt((resid**2).mean())), 2))
        print("  singular values of A:", np.round(np.linalg.svd(A, compute_uv=False), 3).tolist())
    if em_rows:
        A2 = np.vstack(em_rows)
        u, s2, vt = np.linalg.svd(A2)
        print("epipolar-direction solution t_c =", np.round(vt[-1], 4).tolist(), "sv:", np.round(s2, 4).tolist())


if __name__ == "__main__":
    main()