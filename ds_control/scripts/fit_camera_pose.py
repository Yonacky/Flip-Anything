"""用深度 + 光流直接拟合相机相对 TCP 的安装姿态与光心偏移（自写高斯-牛顿，无需 scipy）。

对纯姿态变化的帧对 i->i+1：
  C_i = p_i + R_i t_c    (t_c 为光心在工具坐标系的偏移)
  X_cam(i) = R_ct^T (X_base - C_i)
  X_cam(i+1) = R_rel^T ( X_cam(i) - R_ct^T (C_{i+1} - C_i) )
  R_rel = R_ct^T R_tool_i+1,i R_ct
"""
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


def rodrigues(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    return R


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
    data = []
    for a, b in zip(frames, frames[1:]):
        if np.linalg.norm(b["pose"][:3] - a["pose"][:3]) > 3.0:
            continue
        p0 = cv2.goodFeaturesToTrack(a["gray"], maxCorners=3000, qualityLevel=0.005, minDistance=6, blockSize=7)
        if p0 is None:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(a["gray"], b["gray"], p0, None, winSize=(31, 31), maxLevel=4)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(b["gray"], a["gray"], p1, None, winSize=(31, 31), maxLevel=4)
        fb = np.linalg.norm(back - p0, axis=2).reshape(-1)
        ok = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 3.0)
        pts = p0.reshape(-1, 2)[ok]
        nxt = p1.reshape(-1, 2)[ok]
        depth = a["depth"]
        keep = []
        for (u, v), (u2, v2) in zip(pts, nxt):
            z = depth[int(round(v)), int(round(u))]
            if 1 < z < 3000:
                keep.append((u, v, z, u2, v2))
        if len(keep) < 25:
            print(f"pair {a['idx']}->{b['idx']}: {len(keep)} usable points (skipped)")
            continue
        keep = np.array(keep, float)
        R_tool = b["R"] @ a["R"].T
        data.append({"pair": (a["idx"], b["idx"]), "R_tool": R_tool, "obs": keep,
                     "R_i": a["R"], "R_j": b["R"], "p_i": a["pose"][:3], "p_j": b["pose"][:3],
                     "angle": math.degrees(math.acos(np.clip((np.trace(R_tool) - 1) / 2, -1, 1)))})
        print(f"pair {a['idx']}->{b['idx']}: {len(keep)} points, {data[-1]['angle']:.1f} deg")
    if not data:
        raise SystemExit("no usable pairs")

    def residuals(x):
        R_ct = rodrigues(x[0:3])
        t_c = x[3:6]
        out = []
        for d in data:
            R_rel = R_ct.T @ d["R_tool"] @ R_ct
            C_rel = d["R_j"] @ t_c - d["R_i"] @ t_c          # C_{i+1} - C_i（相对基座位姿平移部分）
            shift = R_ct.T @ C_rel
            o = d["obs"]
            u, v, z = o[:, 0], o[:, 1], o[:, 2]
            Xc = np.stack([(u - I["cx"]) * z / I["fx"], (v - I["cy"]) * z / I["fy"], z], axis=1)
            Xn = (Xc - shift) @ R_rel
            pr = Xn @ K.T
            uv = pr[:, :2] / pr[:, 2:3]
            out.append((uv - o[:, 3:5]).ravel())
        return np.concatenate(out)

    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 110.0])
    lam = 1e-3
    r = residuals(x)
    cost = float((r ** 2).mean())
    for it in range(80):
        J = np.zeros((len(r), 6))
        for k in range(6):
            dx = np.zeros(6)
            dx[k] = 1e-6
            J[:, k] = (residuals(x + dx) - r) / 1e-6
        A = J.T @ J + lam * np.eye(6)
        g = J.T @ r
        try:
            step = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError:
            break
        x_new = x + step
        r_new = residuals(x_new)
        if (r_new ** 2).mean() < cost:
            x, r, cost = x_new, r_new, float((r_new ** 2).mean())
            lam = max(lam * 0.5, 1e-9)
        else:
            lam *= 5
            if lam > 1e6:
                break
    rms = float(np.sqrt((r ** 2).mean()))
    R_ct = rodrigues(x[0:3])
    t_c = x[3:6]
    print("rms px:", round(rms, 4))
    print("R_tcp_cam=\n", np.round(R_ct, 4))
    print("t_cam_in_tcp_mm:", np.round(t_c, 2).tolist(), "|t|=", round(float(np.linalg.norm(t_c)), 2))
    print("optical axis (tool):", np.round(R_ct[:, 2], 3).tolist())
    # 残差按帧对统计
    pos = 0
    for d in data:
        n = len(d["obs"]) * 2
        seg = r[pos:pos + n]
        pos += n
        print(f"  pair {d['pair']} rms {np.sqrt((seg**2).mean()):.2f} px")
    (RUN / "camera_pose_fit.json").write_text(json.dumps(
        {"R_tcp_cam": R_ct.tolist(), "t_tcp_cam_mm": t_c.tolist(), "rms_px": rms,
         "pairs": [[d["pair"][0], d["pair"][1], len(d["obs"])] for d in data]},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()