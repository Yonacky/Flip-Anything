"""解相机相对 TCP 的姿态与光心：对纯姿态变化的相邻帧，
用工具相对旋转 + 未知相机安装关系构造本质矩阵，以 Sampson 距离做全局搜索与细化。"""
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


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def sampson(E, n1, n2):
    one1 = np.ones((len(n1), 1))
    one2 = np.ones((len(n2), 1))
    x1 = np.hstack([n1, one1])
    x2 = np.hstack([n2, one2])
    Ex2 = (E @ x2.T).T
    Etx1 = (E.T @ x1.T).T
    num = np.sum(x1 * Ex2, axis=1) ** 2
    den = Ex2[:, 0] ** 2 + Ex2[:, 1] ** 2 + Etx1[:, 0] ** 2 + Etx1[:, 1] ** 2
    return num / np.maximum(den, 1e-15)


def build_pairs(frames, K):
    Kinv = np.linalg.inv(K)
    pairs = []
    for a, b in zip(frames, frames[1:]):
        if np.linalg.norm(np.array(b["pose"][:3]) - np.array(a["pose"][:3])) > 3.0:
            continue
        p0 = cv2.goodFeaturesToTrack(a["gray"], maxCorners=1000, qualityLevel=0.01, minDistance=8, blockSize=7)
        if p0 is None:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(a["gray"], b["gray"], p0, None, winSize=(21, 21), maxLevel=3)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(b["gray"], a["gray"], p1, None, winSize=(21, 21), maxLevel=3)
        fb = np.linalg.norm(back - p0, axis=2).reshape(-1)
        ok = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 2.0)
        x1 = p0.reshape(-1, 2)[ok]
        x2 = p1.reshape(-1, 2)[ok]
        print("  candidate pair", a["idx"], b["idx"], "features", len(p0), "kept", len(x1))
        if len(x1) < 60:
            continue
        n1 = (Kinv @ np.hstack([x1, np.ones((len(x1), 1))]).T).T[:, :2]
        n2 = (Kinv @ np.hstack([x2, np.ones((len(x2), 1))]).T).T[:, :2]
        R_tool = b["R"] @ a["R"].T
        pairs.append({"pair": (a["idx"], b["idx"]), "n": len(n1), "n1": n1, "n2": n2, "R_tool": R_tool,
                      "angle": math.degrees(math.acos(np.clip((np.trace(R_tool) - 1) / 2, -1, 1)))})
    return pairs


def cost(R_ct, u, pairs, scale_mm=40.0):
    total, cnt = 0.0, 0
    for p in pairs:
        R_rel = R_ct.T @ p["R_tool"] @ R_ct
        t = scale_mm * p["angle"] / 20.0 * u
        E = skew(t) @ R_rel
        r = sampson(E, p["n1"], p["n2"])
        total += float(np.median(r)) * p["n"]
        cnt += p["n"]
    return total / cnt


def best_u(R_ct, pairs, scale_mm=40.0):
    best = None
    for az in np.linspace(0, 2 * math.pi, 24, endpoint=False):
        for el in np.linspace(-1.3, 1.3, 13):
            u = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            c = cost(R_ct, u, pairs, scale_mm)
            if best is None or c < best[0]:
                best = (c, u.copy())
    return best[1], best[0]


def mat_to_euler(R):
    ry = math.asin(max(-1, min(1, -R[2][0])))
    rx = math.atan2(R[2][1], R[2][2])
    rz = math.atan2(R[1][0], R[0][0])
    return np.array([rx, ry, rz])


def euler_mat(v):
    rx, ry, rz = v
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def main():
    I = intr()
    K = np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1]], float)
    frames = []
    for p in obs():
        stem = Path(p["camera"]["color_file"]).stem
        idx = int(stem.split("_")[1])
        bgr = cv2.imread(str(RUN / f"frame_{idx:06d}_color.png"), cv2.IMREAD_COLOR)
        frames.append({"idx": idx, "gray": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                       "R": np.array(pose_matrix(p["pose"]))[:3, :3], "pose": p["pose"]})
    pairs = build_pairs(frames, K)
    print("pairs:", [(p["pair"], p["n"], round(p["angle"], 2)) for p in pairs])
    if not pairs:
        raise SystemExit("no pairs")

    best = None
    for rx in np.deg2rad(np.arange(-180, 180, 20)):
        for ry in np.deg2rad(np.arange(-90, 91, 20)):
            for rz in np.deg2rad(np.arange(-180, 180, 20)):
                R_ct = euler_mat(np.array([rx, ry, rz]))
                c = cost(R_ct, np.array([0.0, 0.0, 1.0]), pairs)
                if best is None or c < best[0]:
                    best = (c, R_ct.copy())
    print("stage1 best cost %.4e" % best[0])
    R_ct = best[1]

    step = np.deg2rad(10)
    cur = mat_to_euler(R_ct)
    for it in range(4):
        improved = True
        while improved:
            improved = False
            for d in itertools6(step):
                cand = cur + np.array(d)
                R_c = euler_mat(cand)
                u, c = best_u(R_c, pairs)
                if c < best[0] - 1e-15:
                    best = (c, R_c)
                    cur = cand
                    improved = True
        step /= 2
        print("iter %d cost %.5e rpy_deg %s" % (it, best[0], np.round(np.degrees(cur), 2).tolist()))
    R_ct = best[1]
    u, c = best_u(R_ct, pairs)
    print("R_tcp_cam=\n", np.round(R_ct, 4))
    print("optical axis in tool frame:", np.round(R_ct[:, 2], 3).tolist())
    print("image x axis in tool frame:", np.round(R_ct[:, 0], 3).tolist())
    print("translation direction (tool frame):", np.round(u, 3).tolist(), "cost", c)
    (RUN / "camera_rotation.json").write_text(json.dumps(
        {"R_tcp_cam": R_ct.tolist(), "t_dir_tool": u.tolist(), "sampson_cost": c,
         "pairs": [[p["pair"][0], p["pair"][1], p["n"]] for p in pairs]}, ensure_ascii=False, indent=2),
        encoding="utf-8")


def itertools6(step):
    for a in (-step, 0, step):
        for b in (-step, 0, step):
            for c in (-step, 0, step):
                if a or b or c:
                    yield (a, b, c)


if __name__ == "__main__":
    main()