"""Measure this round's can: table plane, axis, diameter, height above the table."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb


def fit_circle(pts):
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = math_sqrt(sol[2] + cx * cx + cy * cy)
    res = np.abs(np.hypot(x - cx, y - cy) - r)
    return cx, cy, r, res


def math_sqrt(v):
    return float(np.sqrt(max(v, 0.0)))


def red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = (cv2.inRange(hsv, (0, 70, 50), (14, 255, 255)) | cv2.inRange(hsv, (166, 70, 50), (180, 255, 255))) > 0
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n < 2:
        return None
    i = max(range(1, n), key=lambda q: st[q, cv2.CC_STAT_AREA])
    return lab == i


def main():
    run = tb.run_dir()
    item = tb.results_with_camera(run)[-1]
    k = tb.intrinsics(run)
    mount = tb.load_mount(tb.PRIOR_MOUNT)
    R_cam, cam = tb.camera_pose(item['pose'], mount)
    img = tb.read_color(item['color'])
    depth = tb.load_depth(item['depth'])
    h, w = depth.shape
    out = {'frame': item['color'].name, 'pose': item['pose'],
           'camera_base': np.round(cam, 1).tolist()}

    # table plane from the near table area (bottom third, both sides of the can)
    table_mask = np.zeros((h, w), bool)
    table_mask[int(h * 0.55):, :] = True
    plane = tb.fit_plane(depth, k, lo=150.0, hi=900.0, mask=table_mask)
    base_plane = tb.plane_in_base(plane, R_cam, cam)
    z_table = base_plane['z_at_origin']
    out['table'] = {'z_base_mm': round(z_table, 1), 'tilt_deg': round(base_plane['tilt_deg'], 2),
                    'inliers': plane['inliers'], 'res_p95': round(plane['res_p95'], 1)}

    m = red_mask(img)
    ys, xs = np.where(m)
    out['mask_px'] = int(m.sum())
    out['bbox_uv'] = [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())]
    top, bot = int(ys.min()), int(ys.max())

    body = m.copy()
    body[:top + 18] = False          # drop the lid band
    body[bot - 2:] = False
    vv, uu = np.where(body & (depth > 0))
    pts = tb.backproject(depth, k, R_cam, cam, uu, vv)
    zc = pts[:, 2]
    sel = (zc > z_table + 12) & (zc < z_table + 150)
    print('body px', int(body.sum()), 'valid', int(sel.sum()),
          'z p5/p50/p95', np.round(np.percentile(zc[sel], [5, 50, 95]), 1) if sel.sum() else None)
    if sel.sum() < 200:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return
    P = pts[sel]
    cx_, cy_, r_, res = fit_circle(P[:, :2])
    keep = res < max(3.0, float(np.percentile(res, 60)))
    cx_, cy_, r_, res = fit_circle(P[keep])
    out['can'] = {'axis_base_xy': [round(float(cx_), 1), round(float(cy_), 1)],
                  'radius_mm': round(float(r_), 1), 'diameter_mm': round(float(2 * r_), 1),
                  'n_used': int(keep.sum()), 'res_med_mm': round(float(np.median(res)), 2),
                  'res_p95_mm': round(float(np.percentile(res, 95)), 2)}

    # top rim height: topmost mask rows, back-projected at their own depth
    rows = np.arange(top, min(top + 12, h))
    tv, tu = np.where(m[rows, :] & (depth[rows, :] > 0)) if rows.size else (np.array([]), np.array([]))
    if tv.size:
        tv = tv + rows[0]
        Ptop = tb.backproject(depth, k, R_cam, cam, tu, tv)
        out['top'] = {'z_med_mm': round(float(np.median(Ptop[:, 2])), 1),
                      'height_above_table_mm': round(float(np.median(Ptop[:, 2]) - z_table), 1),
                      'n': int(tv.size),
                      'axis_xy_at_top': [round(float(np.median(Ptop[:, 0])), 1),
                                         round(float(np.median(Ptop[:, 1])), 1)]}

    # base contact: lowest mask rows should sit on the table plane
    brow = np.arange(max(bot - 8, 0), bot + 1)
    bv, bu = np.where(m[brow, :] & (depth[brow, :] > 0))
    if bv.size:
        bv = bv + brow[0]
        Pbase = tb.backproject(depth, k, R_cam, cam, bu, bv)
        out['base'] = {'z_med_mm': round(float(np.median(Pbase[:, 2])), 1),
                       'above_table_mm': round(float(np.median(Pbase[:, 2]) - z_table), 1),
                       'n': int(bv.size)}
    out['width_px'] = int(xs.max() - xs.min())
    out['depth_on_mask_med_mm'] = round(float(np.median(depth[m & (depth > 0)])), 1) if (m & (depth > 0)).any() else None
    print(json.dumps(out, ensure_ascii=False, indent=1))
    (run / 'can_measure.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
