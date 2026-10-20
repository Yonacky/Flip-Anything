"""正确模型的深度变化最小二乘：z' - z = (R_rel^T n1)_z * (-R_cam^T dC)，
其中 dC = (R_tool - I) R_cam t_tool（相机固连工具时的光心位移）。"""
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
    frames = []
    for p in obs():
        stem = Path(p["camera"]["color_file"]).stem
        idx = int(stem.split("_")[1])
        frames.append({"idx": idx,
                       "g": cv2.imread(str(RUN / f"frame_{idx:06d}_color.png"), cv2.IMREAD_GRAYSCALE),
                       "pose": np.array(p["pose"], float),
                       "R": np.array(pose_matrix(p["pose"]))[:3, :3],
                       "depth": np.load(RUN / f"frame_{idx:06d}_depth_raw.npy").astype(np.float64)})
    Rct = np.array(json.loads((RUN / "camera_rotation.json").read_text(encoding="utf-8"))["R_tcp_cam"])
    Rct = Rct / np.linalg.det(Rct) ** (1 / 3)
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.01)
    base = frames[0]
    kb, db = sift.detectAndCompute(base["g"], None)
    A_all, b_all, meta = [], [], []
    for f in frames[1:]:
        if np.linalg.norm(f["pose"][:3] - base["pose"][:3]) > 3.0:
            continue
        kf, df = sift.detectAndCompute(f["g"], None)
        ms = cv2.BFMatcher().knnMatch(db, df, k=2)
        good = [m for m, n in (pr for pr in ms if len(pr) == 2) if m.distance < 0.8 * n.distance]
        R_tool = f["R"] @ base["R"].T
        # 相机系下的相对旋转: R_rel = Rct^T R_tool' Rct ；帧1->帧2 用 R_cam 的相对
        R_rel = Rct.T @ R_tool @ Rct
        M = (R_tool - np.eye(3)) @ Rct        # dC = M t_tool
        rows, rhs = [], []
        for m in good:
            u1, v1 = kb[m.queryIdx].pt
            u2, v2 = kf[m.trainIdx].pt
            z1 = base["depth"][int(round(v1)), int(round(u1))]
            z2 = f["depth"][int(round(v2)), int(round(u2))]
            if not (50 < z1 < 2000 and 50 < z2 < 2000):
                continue
            n1 = np.array([(u1 - I["cx"]) / I["fx"], (v1 - I["cy"]) / I["fy"], 1.0])
            n1 = n1 / np.linalg.norm(n1)
            coef = (R_rel.T @ n1)[2] * (-Rct.T @ M)
            rows.append(coef)
            rhs.append(z2 - z1)
        if len(rows) < 20:
            continue
        A_all.append(np.array(rows)); b_all.append(np.array(rhs))
        meta.append((base["idx"], f["idx"], len(rows), np.round(np.median(rhs), 1)))
        print(f"pair {base['idx']}->{f['idx']}: rows={len(rows)} dz_median={np.median(rhs):.1f}")
    A = np.vstack(A_all); b = np.concatenate(b_all)
    # 两阶段稳健最小二乘
    w = np.ones(len(A))
    for _ in range(40):
        Aw = A * w[:, None]
        # 加入小的岭项以避免退化
        sol, *_ = np.linalg.lstsq(np.vstack([Aw, 1e-6 * np.eye(3)]), np.concatenate([b * w, np.zeros(3)]), rcond=None)
        resid = A @ sol - b
        s = 1.4826 * np.median(np.abs(resid)) + 1e-6
        w = 1.0 / np.sqrt(1 + (resid / (2.0 * s)) ** 2)
    print("t_cam_in_base (mm) =", np.round(sol, 2).tolist(), "|t| =", round(float(np.linalg.norm(sol)), 2))
    t_tool = Rct.T @ sol
    print("t_cam_in_tool (mm) =", np.round(t_tool, 2).tolist())
    print("residual median/rms (mm):", round(float(np.median(np.abs(resid))), 2), round(float(np.sqrt((resid ** 2).mean())), 2))
    print("singular values:", np.round(np.linalg.svd(A, compute_uv=False), 4).tolist())
    (RUN / "camera_offset_depth.json").write_text(json.dumps(
        {"t_cam_base_mm": sol.tolist(), "t_cam_tool_mm": t_tool.tolist(),
         "residual_median_mm": float(np.median(np.abs(resid))),
         "singular_values": np.linalg.svd(A, compute_uv=False).tolist(),
         "pairs": meta}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()