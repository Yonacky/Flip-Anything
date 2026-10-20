"""用 SIFT 匹配 + 5 点法解相机相对工具的安装姿态（对纯姿态变化的帧对）。"""
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
        bgr = cv2.imread(str(RUN / f"frame_{idx:06d}_color.png"), cv2.IMREAD_COLOR)
        frames.append({"idx": idx, "gray": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                       "pose": np.array(p["pose"], float), "R": np.array(pose_matrix(p["pose"]))[:3, :3]})
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.008)
    feats = []
    for f in frames:
        kp, des = sift.detectAndCompute(f["gray"], None)
        feats.append((kp, des))
        print("frame", f["idx"], "keypoints", len(kp))
    bf = cv2.BFMatcher()
    results = []
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        if np.linalg.norm(b["pose"][:3] - a["pose"][:3]) > 3.0:
            continue
        ka, da = feats[i]
        kb, db = feats[i + 1]
        if da is None or db is None:
            continue
        matches = bf.knnMatch(da, db, k=2)
        good = [m for m, n in (pair for pair in matches if len(pair) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 20:
            print(f"pair {a['idx']}->{b['idx']}: {len(good)} good matches (skip)")
            continue
        x1 = np.float64([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        x2 = np.float64([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        E, mask = cv2.findEssentialMat(x1, x2, K, method=cv2.RANSAC, prob=0.999, threshold=2.0)
        if E is None:
            print(f"pair {a['idx']}->{b['idx']}: essential failed")
            continue
        n_in, R_rel, t_rel, mask2 = cv2.recoverPose(E, x1, x2, K, mask=mask)
        R_tool = b["R"] @ a["R"].T
        ang_meas = math.degrees(math.acos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
        ang_cmd = math.degrees(math.acos(np.clip((np.trace(R_tool) - 1) / 2, -1, 1)))
        results.append({"pair": (a["idx"], b["idx"]), "matches": len(good), "inliers": int(mask2.sum()),
                        "R_rel": R_rel, "R_tool": R_tool, "ang_meas": ang_meas, "ang_cmd": ang_cmd})
        print(f"pair {a['idx']}->{b['idx']}: matches {len(good)} inliers {int(mask2.sum())} "
              f"angle_meas {ang_meas:.2f} angle_cmd {ang_cmd:.2f}")
    if not results:
        raise SystemExit("no essential-matrix pairs")
    # 由 R_rel = R_ct^T R_tool R_ct 求 R_ct（Procrustes 迭代）
    R_ct = np.eye(3)
    for _ in range(200):
        M = np.zeros((3, 3))
        for r in results:
            M += r["R_tool"].T @ R_ct @ r["R_rel"]
        U, S, Vt = np.linalg.svd(M)
        R_ct = U @ Vt
    resid = []
    for r in results:
        pred = R_ct.T @ r["R_tool"] @ R_ct
        resid.append(math.degrees(math.acos(np.clip((np.trace(pred.T @ r["R_rel"]) - 1) / 2, -1, 1))))
    print("R_tcp_cam=\n", np.round(R_ct, 4))
    print("optical axis in tool frame:", np.round(R_ct[:, 2], 3).tolist())
    print("camera x axis in tool frame:", np.round(R_ct[:, 0], 3).tolist())
    print("per-pair rotation residual deg:", [round(v, 2) for v in resid])
    (RUN / "camera_rotation.json").write_text(json.dumps(
        {"R_tcp_cam": R_ct.tolist(), "residuals_deg": resid,
         "pairs": [{"pair": r["pair"], "matches": r["matches"], "inliers": r["inliers"],
                    "ang_meas": r["ang_meas"], "ang_cmd": r["ang_cmd"]} for r in results]},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()