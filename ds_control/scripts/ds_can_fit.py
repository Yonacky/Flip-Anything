"""Read-only: fit the target can (vertical cylinder) to the current image silhouette.

Uses: camera intrinsics, the camera->TCP prior, the fitted table height in base,
and the standard 330 ml can size. Grid-searches the can axis (x, y) in base and
reports IoU against the green silhouette. No motion commands.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flip_anything.geometry import pose_matrix  # noqa: E402

T_TCP_CAM = np.array([-77.5, -10.0, -81.5])
R_TCP_CAM = np.array([[0.0492, 0.9599, 0.276],
                      [-0.9987, 0.0442, 0.0244],
                      [0.0113, -0.2768, 0.9609]])
CAN_R = 33.0
CAN_H = 115.0


def setup(run):
    folder = ROOT.parent / 'logs' / 'experiments' / 'ds_control' / run
    k = json.loads((folder / "intrinsics.json").read_text(encoding="utf-8-sig"))["payload"]["metadata"]["factory_camera_param"]["color_intrinsic"]
    return folder, k


def green_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (35, 60, 40), (95, 255, 255)) > 0
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)) > 0
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n < 2:
        return None
    i = max(range(1, n), key=lambda k: st[k, cv2.CC_STAT_AREA])
    return (lab == i).astype(np.uint8)


def main():
    run, index = sys.argv[1], int(sys.argv[2])
    pose = [float(v) for v in sys.argv[3:9]]
    table_h = float(sys.argv[9]) if len(sys.argv) > 9 else None
    folder, k = setup(run)
    fx, fy, cx, cy = k["fx"], k["fy"], k["cx"], k["cy"]
    img = cv2.imread(str(folder / ("frame_%06d_color.png" % index)))
    depth = np.load(folder / ("frame_%06d_depth_raw.npy" % index)).astype(np.float32)
    h, w = depth.shape
    mask = green_mask(img)

    if table_h is None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gm = cv2.inRange(hsv, (35, 60, 40), (95, 255, 255)) > 0
        step = 2
        vv, uu = np.mgrid[0:h:step, 0:w:step]
        zz = depth[::step, ::step]
        sel = (zz > 100) & (zz < 900) & (vv > 620) & (~gm[::step, ::step])
        pts = np.stack([(uu[sel] - cx) * zz[sel] / fx, (vv[sel] - cy) * zz[sel] / fy, zz[sel]], 1)
        # fit in camera frame, convert with prior
        T = np.array(pose_matrix(pose))
        R_cam = T[:3, :3] @ R_TCP_CAM
        p_cam = T[:3, 3] + T[:3, :3] @ T_TCP_CAM
        base = np.array([p_cam + R_cam @ p for p in pts])
        rng = np.random.default_rng(1)
        bestp = None
        for _ in range(400):
            q = base[rng.choice(len(base), 3, replace=False)]
            n = np.cross(q[1] - q[0], q[2] - q[0])
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n = n / nn
            if n[2] < 0:
                n, d = -n, -(-n @ q[0])
                d = -(n @ q[0])
            else:
                d = -(n @ q[0])
            cnt = int((np.abs(base @ n + d) < 5).sum())
            if bestp is None or cnt > bestp[0]:
                bestp = (cnt, n, d)
        cnt, n, d = bestp
        inl = np.abs(base @ n + d) < 5
        m = base[inl].mean(0)
        n = np.linalg.svd(base[inl] - m)[2][-1]
        if n[2] < 0:
            n = -n
        d = -(n @ m)
        table_h = float(-d / n[2])
        tilt = float(np.degrees(np.arccos(min(1, abs(n[2])))))
        table_info = {'inliers': cnt, 'res_p95': round(float(np.percentile(np.abs(base[inl] @ n + d), 95)), 2)}
    else:
        tilt = float("nan")

    T = np.array(pose_matrix(pose))
    R_cam = T[:3, :3] @ R_TCP_CAM
    p_cam0 = T[:3, 3] + T[:3, :3] @ T_TCP_CAM
    R_wc = R_cam.T

    def project(P):
        pc = R_wc @ (P - p_cam0)
        z = pc[2]
        return np.array([fx * pc[0] / z + cx, fy * pc[1] / z + cy])

    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    zs = np.linspace(0, CAN_H, 9)
    ang = np.array([[np.cos(t), np.sin(t)] for t in th])

    def predicted(cand):
        pts = []
        for z in zs:
            ring = np.array([[cand[0] + CAN_R * a[0], cand[1] + CAN_R * a[1], table_h + z] for a in ang])
            pts.extend([project(p) for p in ring])
        pts = np.array(pts, dtype=np.float32)
        hull = cv2.convexHull(pts.astype(np.int32))
        canvas = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(canvas, hull, 1)
        return canvas

    def iou(a, b):
        inter = int(np.logical_and(a, b).sum())
        union = int(np.logical_or(a, b).sum())
        return inter / union if union else 0.0

    # initial guess: coarse centroid ray at the median valid depth
    d = depth[mask > 0]
    d = d[d > 50]
    z0 = float(np.median(d)) if d.size else 170.0
    ys, xs = np.where(mask > 0)
    u0 = float(np.median(xs[xs > np.percentile(xs, 40)]))
    P0 = p_cam0 + R_cam @ np.array([(u0 - cx) * z0 / fx, (float(ys.min() + 0.35 * (ys.max() - ys.min())) - cy) * z0 / fy, z0])

    best = None
    for dx in np.arange(-45, 45.1, 2.5):
        for dy in np.arange(-45, 45.1, 2.5):
            cand = (P0[0] + dx, P0[1] + dy)
            score = iou(predicted(cand), mask)
            if best is None or score > best[0]:
                best = (score, cand)
    step = 1.0
    for _ in range(3):
        base = best[1]
        for dx in np.arange(-2.5, 2.6, step):
            for dy in np.arange(-2.5, 2.6, step):
                cand = (base[0] + dx, base[1] + dy)
                score = iou(predicted(cand), mask)
                if score > best[0]:
                    best = (score, cand)
        step /= 2.0
    print(json.dumps({"frame": index, "pose": pose, "table_h": round(table_h, 1),
                      "table_tilt_deg": None if np.isnan(tilt) else round(tilt, 2),
                      "can_axis_base": [round(best[1][0], 1), round(best[1][1], 1)],
                      "can_top_z": round(table_h + CAN_H, 1), "iou": round(best[0], 3),
                      "can_px_u_range": [int(xs.min()), int(xs.max())],
                      "can_px_v_min": int(ys.min())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
