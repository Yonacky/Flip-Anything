"""现场几何：用 RGB-D 深度 + 位姿，从水平桌面平面的多视角约束求解
相机光心相对 TCP 的固定偏移与桌面在基座系中的高度。"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flip_anything.geometry import pose_matrix  # noqa: E402


def load_events(run: Path):
    return [json.loads(l) for l in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def intrinsics(run: Path):
    for e in load_events(run):
        if e["kind"] == "camera_started":
            return e["payload"]["metadata"]["factory_camera_param"]["color_intrinsic"]
    raise RuntimeError("no camera_started")


def observations(run: Path):
    out, last = [], None
    for e in load_events(run):
        if e["kind"] == "robot_sample":
            last = e["payload"]
        elif e["kind"] == "observation" and e["payload"].get("camera"):
            p = dict(e["payload"])
            p["state"] = last or {}
            out.append(p)
    return out


def load_frame(run: Path, payload):
    import cv2
    stem = Path(payload["camera"]["color_file"]).stem
    idx = int(stem.split("_")[1])
    bgr = cv2.imread(str(run / f"frame_{idx:06d}_color.png"), cv2.IMREAD_COLOR)
    depth = np.load(run / f"frame_{idx:06d}_depth_raw.npy").astype(np.float64)
    return idx, bgr, depth


def cloud_from_depth(depth, intr, step=4):
    h, w = depth.shape
    gu, gv = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    z = depth[gv, gu]
    m = (z > 1) & (z < 6000)
    u = gu[m].astype(float)
    v = gv[m].astype(float)
    zz = z[m]
    pts = np.stack([(u - intr["cx"]) * zz / intr["fx"],
                    (v - intr["cy"]) * zz / intr["fy"], zz], axis=1)
    return pts, (u.astype(int), v.astype(int))


def fit_horizontal_plane(points, R_tool_base, tol=5.0, iters=1500, seed=5, min_nz=0.88):
    """RANSAC 只接受法向接近基座 Z 的平面（桌面）。"""
    rng = np.random.default_rng(seed)
    n = len(points)
    best = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        if ln < 1e-6:
            continue
        nrm = nrm / ln
        nb = R_tool_base @ nrm
        if abs(nb[2]) < min_nz:
            continue
        d = -nrm @ p0
        dist = np.abs(points @ nrm + d)
        inl = dist < tol
        cnt = int(inl.sum())
        if cnt < 400:
            continue
        if best is None or cnt > best[0]:
            best = (cnt, nrm, d)
    if best is None:
        return None
    cnt, nrm, d = best
    for _ in range(3):                     # 迭代精修
        dist = np.abs(points @ nrm + d)
        sel = dist < tol
        pts = points[sel]
        c = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - c)
        cand = vh[-1]
        if (R_tool_base @ cand)[2] * (R_tool_base @ nrm)[2] < 0:
            cand = -cand
        nrm = cand
        d = -nrm @ c
    dist = np.abs(points @ nrm + d)
    sel = dist < tol
    nb = R_tool_base @ nrm
    if nb[2] > 0:                          # 统一取基座 -Z 方向为正
        nrm, d, nb = -nrm, -d, -nb
    return {"normal_cam": nrm, "d": float(d), "normal_base": nb, "inliers": sel,
            "count": int(sel.sum()), "residual_rms_mm": float(np.sqrt((dist[sel] ** 2).mean())),
            "cam_to_plane_mm": float(abs(d))}


def green_segments(bgr):
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 50, 40), (95, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnt, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, cnt):
        if stats[i, cv2.CC_STAT_AREA] < 1500:
            continue
        out.append({"bbox_xywh": [int(v) for v in stats[i]], "centroid_px": [round(float(v), 1) for v in cents[i]],
                    "area_px": int(stats[i, cv2.CC_STAT_AREA])})
    out.sort(key=lambda r: -r["area_px"])
    return out


def main():
    run = Path(sys.argv[1]).resolve()
    intr = intrinsics(run)
    obs = observations(run)
    report = {"run": run.name, "intrinsics": intr, "frames": []}
    A, b, used = [], [], []
    for i, p in enumerate(obs):
        idx, bgr, depth = load_frame(run, p)
        pose = p["state"]["pose"]
        R = np.array(pose_matrix(pose))[:3, :3]
        pts, _ = cloud_from_depth(depth, intr)
        table = fit_horizontal_plane(pts, R)
        entry = {"i": i, "frame": idx, "pose": [round(v, 3) for v in pose],
                 "green_px": green_segments(bgr)[:3]}
        if table is not None:
            n_cam = np.array(table["normal_cam"])
            d_cam = table["d"]
            h_i = np.array(pose[:3]) @ (R @ n_cam)
            entry["table"] = {"count": table["count"],
                              "residual_rms_mm": round(table["residual_rms_mm"], 2),
                              "normal_base": [round(float(v), 4) for v in table["normal_base"]],
                              "cam_to_plane_mm": round(table["cam_to_plane_mm"], 1),
                              "height_in_base_along_normal": round(float(h_i), 2)}
            Rcn = R @ n_cam
            A.append(Rcn)
            b.append(-d_cam + h_i)
            used.append({"i": i, "frame": idx})
        report["frames"].append(entry)
    if len(A) >= 3:
        A = np.array(A)
        b = np.array(b)
        sol, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        report["solve"] = {"n_planes": len(A), "rank": int(rank),
                           "singular_values": [round(float(v), 3) for v in sv],
                           "t_cam_in_tcp_mm": [round(float(v), 2) for v in sol],
                           "norm_mm": round(float(np.linalg.norm(sol)), 2),
                           "residual_rms_mm": round(float(np.sqrt(((A @ sol - b) ** 2).mean())), 3),
                           "residuals_mm": [round(float(v), 2) for v in (A @ sol - b)],
                           "planes_used": used}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    (run / "geometry_report.json").write_text(text, encoding="utf-8")
    print(json.dumps(report.get("solve", {}), ensure_ascii=False, indent=2))
    for e in report["frames"]:
        t = e.get("table")
        print(e["frame"], e["pose"][3:], "table" if t else "no-table",
              (str(t["normal_base"]) + " rms=" + str(t["residual_rms_mm"]) + " cam2plane=" + str(t["cam_to_plane_mm"])) if t else "")


if __name__ == "__main__":
    main()