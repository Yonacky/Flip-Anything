"""用两段已知纯平移（沿工具 Z 与工具 X）的深度变化 + 光流，解光心在工具系的偏移 t_c。"""
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
    obs_list = obs()
    frames = []
    for p in obs_list:
        stem = Path(p["camera"]["color_file"]).stem
        idx = int(stem.split("_")[1])
        g = cv2.imread(str(RUN / f"frame_{idx:06d}_color.png"), cv2.IMREAD_GRAYSCALE)
        frames.append({"idx": idx, "g": g, "pose": np.array(p["pose"], float),
                       "R": np.array(pose_matrix(p["pose"]))[:3, :3],
                       "depth": np.load(RUN / f"frame_{idx:06d}_depth_raw.npy").astype(np.float64)})
    print("observations:", [(f["idx"], [round(v, 1) for v in f["pose"][:3]]) for f in frames])

    A_rows, b_vals, notes = [], [], []
    pairs_used = []
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        d_base = b["pose"][:3] - a["pose"][:3]
        if np.linalg.norm(d_base) < 5.0:
            continue
        d_tool = a["R"].T @ d_base
        # 只保留近似单轴纯平移
        if np.max(np.abs(d_tool)) > 1.15 * np.linalg.norm(d_tool):
            pairs_used.append((a["idx"], b["idx"], "pure"))
        else:
            pairs_used.append((a["idx"], b["idx"], "mixed"))
        p0 = cv2.goodFeaturesToTrack(a["g"], maxCorners=3000, qualityLevel=0.005, minDistance=7, blockSize=7)
        if p0 is None:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(a["g"], b["g"], p0, None, winSize=(21, 21), maxLevel=4)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(b["g"], a["g"], p1, None, winSize=(21, 21), maxLevel=4)
        fb = np.linalg.norm(back - p0, axis=2).reshape(-1)
        ok = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 1.0)
        pts = p0.reshape(-1, 2)[ok]
        nxt = p1.reshape(-1, 2)[ok]
        n_rows = 0
        for (u, v), (u2, v2) in zip(pts, nxt):
            z1 = a["depth"][int(round(v)), int(round(u))]
            z2 = b["depth"][int(round(v2)), int(round(u2))]
            if not (30 < z1 < 3000 and 30 < z2 < 3000):
                continue
            # 单位方向（相机系）
            n1 = np.array([(u - I["cx"]) / I["fx"], (v - I["cy"]) / I["fy"], 1.0])
            n1u = n1 / np.linalg.norm(n1)
            # (||t_c + d|| - ||t_c||) * n1u ~= z2 - z1
            for k in range(3):
                row = np.zeros(3)
                row[k] = n1u[k]
                A_rows.append(row)
                b_vals.append(z2 - z1 - n1u @ d_tool)
                n_rows += 1
        print(f"pair {a['idx']}->{b['idx']} d_tool={np.round(d_tool,2).tolist()} rows={n_rows}")
    A = np.array(A_rows); b = np.array(b_vals)
    if len(A) < 6:
        raise SystemExit("insufficient rows")
    # 稳健迭代加权最小二乘
    w = np.ones(len(A))
    for _ in range(20):
        Aw = A * w[:, None]
        sol, *_ = np.linalg.lstsq(Aw, b * w, rcond=None)
        resid = A @ sol - b
        s = 1.4826 * np.median(np.abs(resid)) + 1e-6
        w = 1.0 / np.sqrt(1 + (resid / (2.5 * s)) ** 2)
    print("t_c (tool frame) =", np.round(sol, 2).tolist(), "|t| =", round(float(np.linalg.norm(sol)), 2))
    print("residual median/rms mm:", round(float(np.median(np.abs(resid))), 3), round(float(np.sqrt((resid**2).mean())), 3))
    print("rows:", len(A))
    (RUN / "camera_offset_solve.json").write_text(json.dumps(
        {"t_tcp_cam_mm": sol.tolist(), "residual_median_mm": float(np.median(np.abs(resid))),
         "pairs": pairs_used}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()