"""Read-only: verify a camera->TCP prior against the CURRENT observation."""
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


def intr(run):
    d = json.loads((ROOT.parent / 'logs' / 'experiments' / 'ds_control' / run / "intrinsics.json").read_text(encoding="utf-8-sig"))
    return d["payload"]["metadata"]["factory_camera_param"]["color_intrinsic"]


def green_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (35, 60, 40), (95, 255, 255)) > 0
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)) > 0
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)) > 0
    n, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n < 2:
        return None
    i = max(range(1, n), key=lambda k: st[k, cv2.CC_STAT_AREA])
    return lab == i


def main():
    run, index = sys.argv[1], int(sys.argv[2])
    pose = [float(v) for v in sys.argv[3:9]]
    folder = ROOT.parent / 'logs' / 'experiments' / 'ds_control' / run
    k = intr(run)
    fx, fy, cx, cy = k["fx"], k["fy"], k["cx"], k["cy"]
    img = cv2.imread(str(folder / ("frame_%06d_color.png" % index)))
    depth = np.load(folder / ("frame_%06d_depth_raw.npy" % index)).astype(np.float32)
    h, w = depth.shape

    T = np.array(pose_matrix(pose))
    R_tool, p_tool = T[:3, :3], T[:3, 3]
    R_cam = R_tool @ R_TCP_CAM
    p_cam = p_tool + R_tool @ T_TCP_CAM

    def to_base(px, py, z):
        pc = np.array([(px - cx) * z / fx, (py - cy) * z / fy, z])
        return p_cam + R_cam @ pc, pc

    mask = green_mask(img)
    out = {"pose": pose, "camera_in_base": np.round(p_cam, 1).tolist()}
    if mask is not None:
        ys, xs = np.where(mask)
        d = depth[mask]
        ok = d > 80
        out["can_px"] = {"u_range": [int(xs.min()), int(xs.max())],
                         "v_range": [int(ys.min()), int(ys.max())],
                         "area": int(mask.sum()),
                         "depth_med": round(float(np.median(d[ok])), 1) if ok.any() else None}
        rows = []
        for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
            r = int(ys.min() + frac * (ys.max() - ys.min()))
            band = np.where(mask[r])[0]
            if band.size > 20:
                rows.append((r, float(np.median(band)), float(depth[r, int(np.median(band))])))
        out["axis_rows"] = [{"v": r, "u": round(u, 1), "depth": round(z, 1) if z > 0 else None}
                            for r, u, z in rows]
        step = 4
        vv, uu = np.mgrid[0:h:step, 0:w:step]
        zz = depth[::step, ::step]
        mm = mask[::step, ::step]
        sel = (zz > 60) & (zz < 1500) & (vv > 0.55 * h) & (~mm)
        pts = np.stack([(uu[sel] - cx) * zz[sel] / fx,
                        (vv[sel] - cy) * zz[sel] / fy, zz[sel]], axis=1)
        if len(pts) > 200:
            base_pts = np.array([p_cam + R_cam @ p for p in pts])
            m = base_pts.mean(axis=0)
            nvec = np.linalg.svd(base_pts - m)[2][-1]
            if nvec[2] < 0:
                nvec = -nvec
            d0 = -nvec @ m
            inl = np.abs(base_pts @ nvec + d0) < 6
            m2 = base_pts[inl].mean(axis=0)
            nvec = np.linalg.svd(base_pts[inl] - m2)[2][-1]
            if nvec[2] < 0:
                nvec = -nvec
            d2 = -nvec @ m2
            res = np.abs(base_pts[inl] @ nvec + d2)
            out["table_in_base"] = {
                "normal": np.round(nvec, 4).tolist(),
                "tilt_from_base_z_deg": round(float(np.degrees(np.arccos(min(1, abs(nvec[2]))))), 2),
                "plane_z_at_origin": round(float(-d2 / nvec[2]), 1),
                "inliers": int(inl.sum()), "res_p95": round(float(np.percentile(res, 95)), 2)}
        if rows:
            r, u, z = rows[len(rows) // 2]
            P, _ = to_base(u, r, z)
            out["can_mid_in_base"] = np.round(P, 1).tolist()
            out["can_z_minus_table"] = round(float(P[2] - out.get("table_in_base", {}).get("plane_z_at_origin", 0)), 1)
        r0 = int(ys.min())
        band = np.where(mask[r0 + 3])[0]
        if band.size:
            u = float(np.median(band))
            z = float(np.median(depth[r0 + 3, band]))
            P, _ = to_base(u, r0 + 3, z)
            out["rim_in_base"] = np.round(P, 1).tolist()
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
