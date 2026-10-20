"""联合求解相机相对 TCP 的完整变换 (R_ct, t_c)：
- 纯旋转对：||R_meas - R_ct^T R_tool R_ct||_F 小
- 纯平移对：t_meas 方向 ≈ normalize((R_j - R_i) t_c)
- 全部对：匹配点深度变化与模型预测一致（作为权重较低的项）
自写 Levenberg-Marquardt（6 参数：旋转向量 + 平移）。"""
import json
import math
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


def rodrigues(v):
    R, _ = cv2.Rodrigues(np.asarray(v, float).reshape(3, 1))
    return R


def collect():
    data = []
    for run in RUNS:
        I = intr(run)
        K = np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1]], float)
        frames = []
        for p in obs(run):
            stem = Path(p["camera"]["color_file"]).stem
            idx = int(stem.split("_")[1])
            frames.append({"idx": idx,
                           "g": cv2.imread(str(run / f"frame_{idx:06d}_color.png"), cv2.IMREAD_GRAYSCALE),
                           "pose": np.array(p["pose"], float),
                           "R": np.array(pose_matrix(p["pose"]))[:3, :3],
                           "depth": np.load(run / f"frame_{idx:06d}_depth_raw.npy").astype(np.float64)})
        sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.01)
        feats = [sift.detectAndCompute(f["g"], None) for f in frames]
        for i in range(len(frames) - 1):
            a, b = frames[i], frames[i + 1]
            d_base = b["pose"][:3] - a["pose"][:3]
            R_tool = b["R"] @ a["R"].T
            ang = math.degrees(math.acos(np.clip((np.trace(R_tool) - 1) / 2, -1, 1)))
            trans = float(np.linalg.norm(d_base))
            ka, da = feats[i]
            kb, db = feats[i + 1]
            ms = cv2.BFMatcher().knnMatch(da, db, k=2)
            good = [m for m, n in (pr for pr in ms if len(pr) == 2) if m.distance < 0.8 * n.distance]
            if len(good) < 25:
                continue
            x1 = np.float64([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            x2 = np.float64([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            E, mask = cv2.findEssentialMat(x1, x2, K, method=cv2.RANSAC, prob=0.9999, threshold=1.5)
            if E is None or mask is None or int(mask.sum()) < 12:
                continue
            n_in, R_meas, t_meas, _ = cv2.recoverPose(E, x1, x2, K, mask=mask)
            # 深度变化数据（像素坐标 + 深度）
            k1 = x1.reshape(-1, 2)[mask.reshape(-1).astype(bool)]
            k2 = x2.reshape(-1, 2)[mask.reshape(-1).astype(bool)]
            z1 = np.array([a["depth"][int(round(v)), int(round(u))] for u, v in k1])
            z2 = np.array([b["depth"][int(round(v)), int(round(u))] for u, v in k2])
            ok = (z1 > 50) & (z1 < 2500) & (z2 > 50) & (z2 < 2500)
            k1, k2, z1, z2 = k1[ok], k2[ok], z1[ok], z2[ok]
            n1 = (np.linalg.inv(K) @ np.hstack([k1, np.ones((len(k1), 1))]).T).T[:, :2]
            u1 = np.hstack([n1, np.ones((len(n1), 1))])
            u1 = u1 / np.linalg.norm(u1, axis=1, keepdims=True)
            data.append({"R_tool": R_tool, "R_meas": R_meas, "t_meas": t_meas.reshape(3), "ang": ang,
                         "trans": trans, "R_i": a["R"], "R_j": b["R"], "n1": u1, "z1": z1, "z2": z2,
                         "idx": (run.name, a["idx"], b["idx"]), "inliers": int(mask.sum())})
            print(f"{run.name} pair {a['idx']}->{b['idx']}: inliers={int(mask.sum())} ang={ang:.1f} trans={trans:.1f}")
    return data


def residuals(params, data, w_rot=1.0, w_trans=1.0, w_depth=0.02):
    R_ct = rodrigues(params[0:3])
    t_c = params[3:6]
    res = []
    for d in data:
        # 旋转一致性
        pred_R = R_ct.T @ d["R_tool"] @ R_ct
        dR = pred_R.T @ d["R_meas"]
        ang_err = math.acos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        res.append(w_rot * ang_err)
        # 平移方向一致性
        base_t = (d["R_j"] - d["R_i"]) @ t_c
        n = np.linalg.norm(base_t)
        if n > 1e-6:
            dir_pred = base_t / n
            # 相机系下的期望方向：R_ct^T @ dir_pred
            cam_pred = R_ct.T @ dir_pred
            res.extend((w_trans * (cam_pred - d["t_meas"])).tolist())
        else:
            res.extend([0.0, 0.0, 0.0])
        # 深度变化
        R_rel = R_ct.T @ d["R_tool"].T @ R_ct   # 帧 j 相机系表达 R_i->? 用 (R_tool^T - I)
        coef = (R_ct.T @ (d["R_tool"].T - np.eye(3)) @ R_ct).T
        pred_dz = (d["n1"] @ coef.T)  # 近似：Δz = n1·((R_tool^T - I) t_tool)
        res.extend((w_depth * (pred_dz - (d["z2"] - d["z1"]))).tolist())
    return np.array(res)


def main():
    data = collect()
    if len(data) < 3:
        raise SystemExit("not enough pairs")
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 110.0])
    r = residuals(x, data)
    cost = float((r ** 2).mean())
    lam = 1e-3
    for it in range(60):
        J = np.zeros((len(r), 6))
        for k in range(6):
            dx = np.zeros(6)
            dx[k] = 1e-5
            J[:, k] = (residuals(x + dx, data) - r) / 1e-5
        A = J.T @ J + lam * np.eye(6)
        try:
            step = np.linalg.solve(A, -J.T @ r)
        except np.linalg.LinAlgError:
            break
        xn = x + step
        rn = residuals(xn, data)
        cn = float((rn ** 2).mean())
        if cn < cost:
            x, r, cost = xn, rn, cn
            lam = max(lam * 0.3, 1e-10)
        else:
            lam *= 4
            if lam > 1e8:
                break
    R_ct = rodrigues(x[0:3])
    t_c = x[3:6]
    print("\n=== 联合解 ===")
    print("R_tcp_cam=\n", np.round(R_ct, 4))
    print("optical axis in tool frame:", np.round(R_ct[:, 2], 4).tolist())
    print("t_cam_in_tcp_mm:", np.round(t_c, 2).tolist(), "|t| =", round(float(np.linalg.norm(t_c)), 2))
    print("cost rms:", round(math.sqrt(cost), 5))
    out = {"R_tcp_cam": R_ct.tolist(), "t_tcp_cam_mm": t_c.tolist(),
           "rms": math.sqrt(cost), "pairs": [d["idx"] for d in data]}
    (RUNS[0] / "handeye_joint.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()