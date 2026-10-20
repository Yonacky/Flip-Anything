"""由纯旋转帧对的本质矩阵平移方向解光心偏移 t_c（工具系），
t_tool_i ∝ R_i^T R_ct t_c；对多个旋转轴的最小二乘解。"""
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
                       "R": np.array(pose_matrix(p["pose"]))[:3, :3]})
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.008)
    feats = [sift.detectAndCompute(f["g"], None) for f in frames]
    bf = cv2.BFMatcher()
    rows, rhs, details = [], [], []
    Rct = np.array(json.loads((RUN / "camera_rotation.json").read_text(encoding="utf-8"))["R_tcp_cam"])
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        if np.linalg.norm(b["pose"][:3] - a["pose"][:3]) > 3.0:
            continue
        ka, da = feats[i]
        kb, db = feats[i + 1]
        matches = bf.knnMatch(da, db, k=2)
        good = [m for m, n in (pr for pr in matches if len(pr) == 2) if m.distance < 0.8 * n.distance]
        if len(good) < 20:
            continue
        x1 = np.float64([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        x2 = np.float64([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        E, mask = cv2.findEssentialMat(x1, x2, K, method=cv2.RANSAC, prob=0.9999, threshold=1.5)
        if E is None or mask is None:
            continue
        n_in, R_rel, t_rel, _ = cv2.recoverPose(E, x1, x2, K, mask=mask)
        R_tool = b["R"] @ a["R"].T
        # t_tool = a["R"].T @ (C_{i+1} - C_i)，且 t_tool = (R_tool - I) t_c
        M = R_tool - np.eye(3)
        # 方程 M t_c ∝ t_rel
        # 跨乘消去尺度：skew(t_rel) M t_c = 0
        t = t_rel.reshape(3)
        S = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
        A = S @ M
        rows.append(A)
        n = len(A)
        rhs.append(np.zeros(n))
        details.append({"pair": (a["idx"], b["idx"]), "inliers": int(mask.sum()),
                        "angle": round(math.degrees(math.acos(np.clip((np.trace(R_tool) - 1) / 2, -1, 1))), 2),
                        "t_rel": np.round(t, 3).tolist()})
        print(f"pair {a['idx']}->{b['idx']} inliers={int(mask.sum())} angle={details[-1]['angle']} "
              f"t_rel={np.round(t,3).tolist()} |M|={np.round(np.linalg.norm(M,axis=0),3).tolist()}")
    if not rows:
        raise SystemExit("no rows")
    A = np.vstack(rows)
    u, s, vt = np.linalg.svd(A)
    t_c = vt[-1]
    print("singular values:", np.round(s, 4).tolist())
    print("t_c direction (tool frame):", np.round(t_c, 4).tolist())
    # 用深度变化解出的规模定标（前一步 46mm 量级）仅作参考；此处输出方向与最小奇异值
    out = {"t_c_direction_tool": t_c.tolist(), "singular_values": s.tolist(), "pairs": details}
    (RUN / "camera_offset_direction.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()