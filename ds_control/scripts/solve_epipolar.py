"""对极误差搜索相机安装姿态：对纯旋转帧对，用 SIFT 匹配 + 已知工具相对旋转，
最小化 Sampson 距离（本质矩阵由 E = [t]x R_ct^T R_tool R_ct 给出）。"""
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


def rot_axis_angle(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def sampson(E, n1, n2):
    x1 = np.hstack([n1, np.ones((len(n1), 1))])
    x2 = np.hstack([n2, np.ones((len(n2), 1))])
    Ex2 = (E @ x2.T).T
    Etx1 = (E.T @ x1.T).T
    num = np.sum(x1 * Ex2, axis=1) ** 2
    den = Ex2[:, 0] ** 2 + Ex2[:, 1] ** 2 + Etx1[:, 0] ** 2 + Etx1[:, 1] ** 2
    return num / np.maximum(den, 1e-15)


def main():
    I = intr()
    K = np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1]], float)
    Kinv = np.linalg.inv(K)
    frames = []
    for p in obs():
        stem = Path(p["camera"]["color_file"]).stem
        idx = int(stem.split("_")[1])
        bgr = cv2.imread(str(RUN / f"frame_{idx:06d}_color.png"), cv2.IMREAD_COLOR)
        frames.append({"idx": idx, "gray": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
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
        good = [m for m, n in (pr for pr in matches if len(pr) == 2) if m.distance < 0.8 * n.distance]
        if len(good) < 15:
            continue
        x1 = np.float64([ka[m.queryIdx].pt for m in good]).reshape(-1, 2)
        x2 = np.float64([kb[m.trainIdx].pt for m in good]).reshape(-1, 2)
        n1 = (Kinv @ np.hstack([x1, np.ones((len(x1), 1))]).T).T[:, :2]
        n2 = (Kinv @ np.hstack([x2, np.ones((len(x2), 1))]).T).T[:, :2]
        pairs.append({"pair": (a["idx"], b["idx"]), "n1": n1, "n2": n2, "R_tool": b["R"] @ a["R"].T,
                      "matches": len(x1)})
        print(f"pair {a['idx']}->{b['idx']}: {len(x1)} matches")
    if not pairs:
        raise SystemExit("no pairs")

    dirs = []
    for az in np.linspace(0, 2 * math.pi, 24, endpoint=False):
        for el in np.linspace(-1.4, 1.4, 15):
            dirs.append(np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]))
    dirs = np.array(dirs)

    def score(R_ct, u):
        tot, cnt = 0.0, 0
        for p in pairs:
            R_rel = R_ct.T @ p["R_tool"] @ R_ct
            t = u * 40.0
            E = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) @ R_rel
            r = sampson(E, p["n1"], p["n2"])
            tot += float(np.mean(np.minimum(r, 0.01))) * p["matches"]
            cnt += p["matches"]
        return tot / cnt

    def score_u(R_ct):
        return min((score(R_ct, u), i) for i, u in enumerate(dirs))

    best = None
    axes = []
    for az in np.linspace(0, 2 * math.pi, 12, endpoint=False):
        for el in np.linspace(-1.4, 1.4, 9):
            axes.append([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    for ax in axes:
        for deg in np.arange(-60, 61, 15):
            R_ct = rot_axis_angle(ax, math.radians(deg))
            s, ui = score_u(R_ct)
            if best is None or s < best[0]:
                best = (s, R_ct.copy(), dirs[ui].copy())
    print("stage1 best score %.3e" % best[0])
    print("stage1 R_ct=\n", np.round(best[1], 4))
    print("stage1 optical axis in tool:", np.round(best[1][:, 2], 3).tolist())

    # 局部细化：绕 R_ct 的旋转扰动 + 平移方向扰动
    R_ct, u = best[1], best[2]
    step_r, step_u = math.radians(8), 0.25
    cur = best[0]
    for it in range(60):
        improved = False
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            dR = rot_axis_angle(np.array(d, float), step_r)
            for Rcand in (dR @ R_ct, R_ct @ dR):
                s, ui = score_u(Rcand)
                if s < cur - 1e-12:
                    R_ct, cur, u = Rcand, s, dirs[ui].copy()
                    improved = True
        for _ in range(40):
            cand = u + step_u * np.random.default_rng(0).normal(size=3)
            cand = cand / np.linalg.norm(cand)
            s = score(R_ct, cand)
            if s < cur - 1e-12:
                u, cur, improved = cand, s, True
        if not improved:
            step_r *= 0.5
            step_u *= 0.5
            if step_r < math.radians(0.3):
                break
    print("final score %.4e" % cur)
    print("R_tcp_cam=\n", np.round(R_ct, 4))
    print("optical axis in tool frame:", np.round(R_ct[:, 2], 4).tolist())
    print("camera x axis in tool frame:", np.round(R_ct[:, 0], 4).tolist())
    print("baseline direction in tool frame:", np.round(u, 3).tolist())
    (RUN / "camera_rotation_epipolar.json").write_text(json.dumps(
        {"R_tcp_cam": R_ct.tolist(), "score": cur, "t_dir_tool": u.tolist(),
         "pairs": [[p["pair"][0], p["pair"][1], p["matches"]] for p in pairs]}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()