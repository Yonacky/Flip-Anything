"""多对帧的深度变化最小二乘解光心偏移（模型：Δz = n1·((R_tool^T - I) t_c)），
同时对工具系偏移与'相对位姿平移'成立；并给出奇异值以判断可观测性。"""
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
    sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.01)
    feats = [sift.detectAndCompute(f["g"], None) for f in frames]
    A_all, b_all, group = [], [], []
    gid = 0
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        if np.linalg.norm(b["pose"][:3] - a["pose"][:3]) > 3.0:
            continue
        ka, da = feats[i]
        kb, db = feats[i + 1]
        ms = cv2.BFMatcher().knnMatch(da, db, k=2)
        good = [m for m, n in (pr for pr in ms if len(pr) == 2) if m.distance < 0.8 * n.distance]
        if len(good) < 25:
            continue
        R_tool = b["R"] @ a["R"].T
        N = (R_tool.T - np.eye(3))
        rows, rhs = [], []
        for m in good:
            u1, v1 = ka[m.queryIdx].pt
            u2, v2 = kb[m.trainIdx].pt
            z1 = a["depth"][int(round(v1)), int(round(u1))]
            z2 = b["depth"][int(round(v2)), int(round(u2))]
            if not (50 < z1 < 2500 and 50 < z2 < 2500):
                continue
            n1 = np.array([(u1 - I["cx"]) / I["fx"], (v1 - I["cy"]) / I["fy"], 1.0])
            n1 = n1 / np.linalg.norm(n1)
            rows.append(n1 @ N)
            rhs.append(z2 - z1)
        if len(rows) < 25:
            continue
        d = np.linalg.norm(b["pose"][:3] - a["pose"][:3])
        rot = np.degrees(np.arccos(np.clip((np.trace(R_tool) - 1) / 2, -1, 1)))
        print(f"pair {a['idx']}->{b['idx']}: rows={len(rows)} translation={d:.1f}mm rotation={rot:.1f}deg")
        A_all.append(np.array(rows)); b_all.append(np.array(rhs)); group.append(gid)
        gid += 1
    A = np.vstack(A_all); b = np.concatenate(b_all)
    w = np.ones(len(A))
    for _ in range(40):
        sol, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
        resid = A @ sol - b
        s = 1.4826 * np.median(np.abs(resid)) + 1e-6
        w = 1.0 / np.sqrt(1 + (resid / (2.0 * s)) ** 2)
    print("t_c (tool frame) =", np.round(sol, 2).tolist(), "|t| =", round(float(np.linalg.norm(sol)), 2))
    print("residual median/rms mm:", round(float(np.median(np.abs(resid))), 2), round(float(np.sqrt((resid**2).mean())), 2))
    print("A singular values:", np.round(np.linalg.svd(A, compute_uv=False), 3).tolist())
    # 按组残差
    pos = 0
    for g, (Ai, bi) in enumerate(zip(A_all, b_all)):
        n = len(Ai)
        seg = resid[pos:pos + n]; pos += n
        print(f"  group {g}: median |resid| {np.median(np.abs(seg)):.1f} mm")
    (RUN / "camera_offset_depth_all.json").write_text(json.dumps(
        {"t_cam_tool_mm": sol.tolist(), "residual_median_mm": float(np.median(np.abs(resid))),
         "singular_values": np.linalg.svd(A, compute_uv=False).tolist()}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()