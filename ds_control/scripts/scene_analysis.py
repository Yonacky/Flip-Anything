"""现场几何分析：从 RGB-D + 位姿估计桌面、目标与相机相对 TCP 的关系。"""
import json
import sys
from pathlib import Path

import numpy as np


def load_events(run: Path):
    return [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def load_intrinsics(run: Path):
    for event in load_events(run):
        if event["kind"] == "camera_started":
            return event["payload"]["metadata"]["factory_camera_param"]["color_intrinsic"]
    raise RuntimeError("no camera_started event")


def observation_payloads(run: Path):
    """把每个 observation 与其之前的最后一个 robot_sample 配对。"""
    out = []
    last_sample = None
    for event in load_events(run):
        if event["kind"] == "robot_sample":
            last_sample = event["payload"]
        elif event["kind"] == "observation" and event["payload"].get("camera"):
            payload = dict(event["payload"])
            payload["state"] = last_sample or {}
            out.append(payload)
    return out


def load_frame(run: Path, index: int):
    import cv2
    bgr = cv2.imread(str(run / f"frame_{index:06d}_color.png"), cv2.IMREAD_COLOR)
    depth = np.load(run / f"frame_{index:06d}_depth_raw.npy").astype(np.float64)
    return bgr, depth


def backproject(depth, intr, step=4):
    h, w = depth.shape
    us = np.arange(0, w, step)
    vs = np.arange(0, h, step)
    gu, gv = np.meshgrid(us, vs)
    z = depth[gv, gu]
    valid = (z > 1) & (z < 4000)
    u = gu[valid].astype(np.float64)
    v = gv[valid].astype(np.float64)
    z = z[valid]
    x = (u - intr["cx"]) * z / intr["fx"]
    y = (v - intr["cy"]) * z / intr["fy"]
    return np.stack([x, y, z], axis=1), valid, (gu, gv)


def fit_plane(points, iters=600, tol=4.0, seed=7):
    rng = np.random.default_rng(seed)
    best = None
    n = len(points)
    if n < 50:
        return None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -normal @ p0
        dist = np.abs(points @ normal + d)
        inliers = dist < tol
        count = int(inliers.sum())
        if best is None or count > best[0]:
            best = (count, normal, d, inliers)
    if best is None:
        return None
    count, normal, d, inliers = best
    pts = points[inliers]
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    normal = vh[-1]
    if normal @ centroid + (-normal @ centroid) < 0:
        pass
    d = -normal @ centroid
    dist = np.abs(points @ normal + d)
    inliers = dist < tol
    residual = float(np.sqrt((dist[inliers] ** 2).mean()))
    return {"count": int(inliers.sum()), "normal": normal.tolist(), "d": float(d),
            "residual_rms_mm": residual, "inliers": inliers}


def segment_green(bgr):
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 60, 50), (95, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None, np.zeros_like(mask)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == best).astype(np.uint8) * 255
    return {"bbox_xywh": stats[best].tolist(), "area_px": int(stats[best, cv2.CC_STAT_AREA]),
            "centroid_px": centroids[best].tolist()}, component


def main():
    run = Path(sys.argv[1]).resolve()
    index = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    intr = load_intrinsics(run)
    payload = observation_payloads(run)[index]
    file_index = int(Path(payload["camera"]["color_file"]).stem.split("_")[1])
    bgr, depth = load_frame(run, file_index)
    points, valid, (gu, gv) = backproject(depth, intr)
    plane = fit_plane(points)
    green, component = segment_green(bgr)

    result = {"frame": file_index, "phase": payload.get("phase"),
              "pose_base_mm_deg": payload["state"]["pose"],
              "joint_deg": payload["state"]["joint_deg"],
              "intrinsics": intr,
              "depth_valid_fraction": float(valid.mean()),
              "depth_percentiles_mm": {str(p): float(np.percentile(depth[(depth > 1) & (depth < 4000)], p)) for p in (5, 25, 50, 75, 95)},
              "green_segment": green}
    if plane is not None:
        result["table_plane_cam"] = {"count": plane["count"], "normal_xyz": [round(v, 5) for v in plane["normal"]],
                                     "d": round(plane["d"], 3), "residual_rms_mm": round(plane["residual_rms_mm"], 3)}
        # 桌面内点到光心距离分布
        dist = np.abs(points @ np.array(plane["normal"]) + plane["d"])
        sel = dist < 4.0
        result["table_plane_cam"]["range_mm"] = [round(float(points[sel, 2].min()), 1),
                                                 round(float(points[sel, 2].max()), 1)]
        result["table_plane_cam"]["pixel_bbox_uv"] = [int(gu[valid][sel].min()), int(gv[valid][sel].min()),
                                                     int(gu[valid][sel].max()), int(gv[valid][sel].max())]
    if green is not None:
        sel = component[(gv * 1)[valid], (gu * 1)[valid]] > 0
        pts = points[sel]
        if len(pts) > 20:
            result["green_points_cam"] = {"n": int(len(pts)),
                                          "mean_xyz_mm": np.round(pts.mean(axis=0), 2).tolist(),
                                          "min_xyz_mm": np.round(pts.min(axis=0), 2).tolist(),
                                          "max_xyz_mm": np.round(pts.max(axis=0), 2).tolist()}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    (run / f"analysis_{file_index:06d}.json").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()