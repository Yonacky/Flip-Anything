"""独立验证：对候选 (R_ct, t_c) 用 DLT 三角化 SIFT 匹配点，
要求基座系重建点投影回两帧、且与深度反投影一致。"""
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


def build_pairs():
    I = intr()
    K = np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1]], float)
    frames = []
    for p in obs():
        stem = Path(p["camera"]["color_file"]).stem
        idx = int(stem.split("_")[1])
        bgr = cv2.imread(str(RUN / f"frame_{idx:06d}_color.png"), cv2.IMREAD_COLOR)
        depth = np.load(RUN / f"frame_{idx:06d}_depth_raw.npy").astype(np.float64)
        frames.append({"idx": idx, "gray": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), "depth": depth,
                       "pose": np.array(p["pose"], float), "R": np.array(pose_matrix(p["pose"]))[:3, :3]})
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.006)
    feats = [sift.detectAndCompute(f["gray"], None) for f in frames]
    bf = cv2.BFMatcher()
    pairs = []
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        if np.linalg.norm(b["pose"][:3] - a["pose"][:3]) > 3.0:
            continue
        ka, da = feats[i]
        kb, db = feats[i + 1]
        matches = bf.knnMatch(da, db, k=2)
        good = [m for m, n in (pr for pr in matches if len(pr) == 2) if m.distance < 0.8 * n.distance]
        if len(good) < 15:
            continue
        x1 = np.float64([ka[m.queryIdx].pt for m in good]).reshape(-1, 2)
        x2 = np.float64([kb[m.trainIdx].pt for m in good]).reshape(-1, 2)
        E, mask = cv2.findEssentialMat(x1.reshape(-1, 1, 2), x2.reshape(-1, 1, 2), K, method=cv2.RANSAC,
                                       prob=0.9999, threshold=1.5)
        if E is None or mask is None:
            continue
        keep = mask.reshape(-1).astype(bool)
        x1, x2 = x1[keep], x2[keep]
        d1 = np.array([a["depth"][int(round(v)), int(round(u))] if (0 <= int(round(v)) < 720 and 0 <= int(round(u)) < 1280) else 0 for u, v in x1], float)
        d2 = np.array([b["depth"][int(round(v)), int(round(u))] if (0 <= int(round(v)) < 720 and 0 <= int(round(u)) < 1280) else 0 for u, v in x2], float)
        ok = (d1 > 1) & (d1 < 3000) & (d2 > 1) & (d2 < 3000)
        x1, x2, d1, d2 = x1[ok], x2[ok], d1[ok], d2[ok]
        if len(x1) < 8:
            continue
        pairs.append({"pair": (a["idx"], b["idx"]), "x1": x1, "x2": x2, "d1": d1, "d2": d2,
                      "R_i": a["R"], "R_j": b["R"], "p_i": a["pose"][:3], "p_j": b["pose"][:3]})
        print(f"pair {a['idx']}->{b['idx']}: {len(x1)} inliers")
    return I, K, pairs


def cam_pose(R_ct, t_c, R_tool_i, p_i):
    R_cam = R_tool_i @ R_ct
    C = p_i + R_tool_i @ t_c
    return R_cam, C


def dlt_triangulate(Rc1, C1, Rc2, C2, x1, x2):
    """返回 3D 点、两帧重投影误差（px）。"""
    out, errs = [], []
    for (u1, v1), (u2, v2) in zip(x1, x2):
        rows = []
        # 相机1：x = K [Rc1|C1] X ；用叉乘消去齐次尺度
        P1 = np.hstack([Rc1, C1.reshape(3, 1)])
        P2 = np.hstack([Rc2, C2.reshape(3, 1)])
        # K 已在像素坐标下，直接构造投影矩阵 K[R|t]
        out.append((P1, P2))
    return out


def main():
    I, K, pairs = build_pairs()
    if not pairs:
        raise SystemExit("no pairs")
    fits = json.loads((RUN / "camera_rotation.json").read_text(encoding="utf-8"))
    R_sift = np.array(fits["R_tcp_cam"])
    candidates = {
        "identity_t110": (np.eye(3), np.array([0.0, 0.0, 110.0])),
        "identity_tm85": (np.eye(3), np.array([0.0, 0.0, -85.0])),
        "sift_rot_t110": (R_sift, np.array([0.0, 0.0, 110.0])),
        "sift_rot_tm85": (R_sift, np.array([0.0, 0.0, -85.0])),
        "sift_rot_t0": (R_sift, np.array([0.0, 0.0, 0.0])),
    }
    out = {}
    for name, (R_ct, t_c) in candidates.items():
        errs = []
        for P in pairs:
            Rc1, C1 = cam_pose(R_ct, t_c, P["R_i"], P["p_i"])
            Rc2, C2 = cam_pose(R_ct, t_c, P["R_j"], P["p_j"])
            P1 = K @ np.hstack([Rc1, C1.reshape(3, 1)])
            P2 = K @ np.hstack([Rc2, C2.reshape(3, 1)])
            for (u1, v1), (u2, v2) in zip(P["x1"], P["x2"]):
                A = np.vstack([u1 * P1[2] - P1[0], v1 * P1[2] - P1[1],
                               u2 * P2[2] - P2[0], v2 * P2[2] - P2[1]])
                _, _, vh = np.linalg.svd(A)
                X = vh[-1]
                X = X[:3] / X[3]
                # 深度（相机系 Z）
                Xc1 = Rc1.T @ (X - C1)
                Xc2 = Rc2.T @ (X - C2)
                if Xc1[2] <= 0 or Xc2[2] <= 0:
                    continue
                errs.append(abs(Xc1[2] - P["d1"][0]) if False else 0)
            # 用逐点深度比较
        # 逐点比较（重新遍历，记录深度差）
        diffs = []
        for P in pairs:
            Rc1, C1 = cam_pose(R_ct, t_c, P["R_i"], P["p_i"])
            Rc2, C2 = cam_pose(R_ct, t_c, P["R_j"], P["p_j"])
            P1 = K @ np.hstack([Rc1, C1.reshape(3, 1)])
            P2 = K @ np.hstack([Rc2, C2.reshape(3, 1)])
            for (u1, v1), (u2, v2), dd1, dd2 in zip(P["x1"], P["x2"], P["d1"], P["d2"]):
                A = np.vstack([u1 * P1[2] - P1[0], v1 * P1[2] - P1[1],
                               u2 * P2[2] - P2[0], v2 * P2[2] - P2[1]])
                _, _, vh = np.linalg.svd(A)
                X = vh[-1]
                if abs(X[3]) < 1e-12:
                    continue
                X = X[:3] / X[3]
                Xc1 = Rc1.T @ (X - C1)
                Xc2 = Rc2.T @ (X - C2)
                if Xc1[2] <= 0 or Xc2[2] <= 0:
                    continue
                diffs.append((abs(Xc1[2] - dd1), abs(Xc2[2] - dd2)))
        diffs = np.array(diffs, float)
        if len(diffs) == 0:
            out[name] = {"n": 0}
            continue
        stat = {"n": int(len(diffs)),
                "median_dz1_mm": round(float(np.median(diffs[:, 0])), 1),
                "median_dz2_mm": round(float(np.median(diffs[:, 1])), 1),
                "within_20mm_frac": round(float(((diffs[:, 0] < 20) & (diffs[:, 1] < 20)).mean()), 3)}
        out[name] = stat
        print(name, json.dumps(stat))
    (RUN / "camera_candidate_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()