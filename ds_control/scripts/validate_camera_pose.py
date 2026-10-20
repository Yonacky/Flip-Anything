"""对候选相机安装姿态做独立验证：用位姿三角化 SIFT 匹配点，
要求三角化深度与深度传感器读数一致（不一致即姿态错误）。"""
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


def euler_mat(rx, ry, rz):
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def depth_consistency(R_ct, t_c, I, pairs):
    """对每对帧的匹配点，在基座系三角化并比较两帧的深度，返回深度差统计。"""
    errs = []
    for P in pairs:
        Ri, Rj = P["R_i"], P["R_j"]
        pi, pj = P["p_i"], P["p_j"]
        x1, x2, d1, d2 = P["n1"], P["n2"], P["d1"], P["d2"]
        C1 = pi + Ri @ t_c
        C2 = pj + Rj @ t_c
        # 相机系方向（单位向量）
        v1 = np.hstack([x1, np.ones((len(x1), 1))]) @ R_ct.T
        v2 = np.hstack([x2, np.ones((len(x2), 1))]) @ R_ct.T
        v1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
        v2 = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
        for k in range(len(v1)):
            A = np.stack([v1[k], -v2[k]], axis=1)
            b = C2 - C1
            sol, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            if rank < 2:
                continue
            a1, b2 = sol[0], sol[1]
            if a1 <= 0 or b2 <= 0:
                continue
            errs.append((abs(a1 - d1[k]), abs(b2 - d2[k])))
    errs = np.array(errs, float)
    e1 = errs[:, 0][~np.isnan(errs[:, 0])]
    e2 = errs[:, 1][~np.isnan(errs[:, 1])]
    return {"n": len(errs), "median_depth_err_1": float(np.median(e1)) if len(e1) else None,
            "median_depth_err_2": float(np.median(e2)) if len(e2) else None,
            "p90_depth_err_1": float(np.percentile(e1, 90)) if len(e1) else None}


def main():
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
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.008)
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
        good = [m for m, n in (pr for pr in matches if len(pr) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 20:
            continue
        x1 = np.float64([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        x2 = np.float64([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        E, mask = cv2.findEssentialMat(x1, x2, K, method=cv2.RANSAC, prob=0.999, threshold=2.0)
        if E is None:
            continue
        keep = mask.reshape(-1).astype(bool)
        x1, x2 = x1[keep].reshape(-1, 2), x2[keep].reshape(-1, 2)
        d1 = np.array([a["depth"][int(round(v)), int(round(u))] for u, v in x1], float)
        d2 = np.array([b["depth"][int(round(v)), int(round(u))] for u, v in x2], float)
        ok = (d1 > 1) & (d1 < 3000) & (d2 > 1) & (d2 < 3000)
        x1, x2, d1, d2 = x1[ok], x2[ok], d1[ok], d2[ok]
        if len(x1) < 8:
            continue
        R_tool = b["R"] @ a["R"].T
        n1 = ((np.linalg.inv(K) @ np.hstack([x1, np.ones((len(x1), 1))]).T).T)[:, :2]
        n2 = ((np.linalg.inv(K) @ np.hstack([x2, np.ones((len(x2), 1))]).T).T)[:, :2]
        pairs.append({"pair": (a["idx"], b["idx"]), "n1": n1, "n2": n2, "d1": d1, "d2": d2,
                      "R_i": a["R"], "R_j": b["R"], "p_i": a["pose"][:3], "p_j": b["pose"][:3],
                      "R_tool": R_tool})
        print(f"pair {a['idx']}->{b['idx']}: {len(x1)} depth-valid inliers")
    if not pairs:
        raise SystemExit("no pairs")
    fits = json.loads((RUN / "camera_rotation.json").read_text(encoding="utf-8"))
    R_sift = np.array(fits["R_tcp_cam"])
    candidates = {
        "identity": (np.eye(3), np.array([0.0, 0.0, 110.0])),
        "axis_+Z_back": (np.eye(3), np.array([0.0, 0.0, -85.0])),
        "sift_solved": (R_sift, np.array([0.0, 0.0, 110.0])),
    }
    out = {}
    for name, (R_ct, t_c) in candidates.items():
        stat = depth_consistency(R_ct, t_c, I, pairs)
        out[name] = stat
        print(name, json.dumps(stat))
    # 对 sift 解做平移的粗搜索，看深度一致性是否改善
    best = None
    for z in np.arange(-140, 141, 20):
        for x in np.arange(-80, 81, 40):
            t_c = np.array([float(x), 0.0, float(z)])
            st = depth_consistency(R_sift, t_c, I, pairs)
            if st["median_depth_err_1"] is None:
                continue
            score = st["median_depth_err_1"] + (st["median_depth_err_2"] or 0)
            if best is None or score < best[0]:
                best = (score, t_c.copy(), st)
    print("best translation for sift rotation:", np.round(best[1], 1).tolist(), json.dumps(best[2]))
    (RUN / "camera_candidate_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()