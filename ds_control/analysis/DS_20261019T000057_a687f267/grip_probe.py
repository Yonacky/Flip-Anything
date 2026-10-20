"""Measure where the finger pads actually meet the can, by intersecting pad pixels
with the known can cylinder. Answers "how much of the can is really gripped"."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb


def ray_cylinder(origin, R_cam, u, v, k, axis_xy, r):
    d_cam = np.stack([(u - k['cx']) / k['fx'], (v - k['cy']) / k['fy'], np.ones_like(u)], 1)
    d = d_cam @ R_cam.T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ox, oy, oz = origin[0] - axis_xy[0], origin[1] - axis_xy[1], origin[2]
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    a = dx * dx + dy * dy
    b = 2 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - r * r
    disc = b * b - 4 * a * c
    hit = disc > 0
    t = np.full(u.shape, np.nan)
    sq = np.sqrt(np.maximum(disc, 0))
    t0 = (-b - sq) / (2 * a)
    t1 = (-b + sq) / (2 * a)
    t = np.where(t0 > 0, t0, t1)
    ok = hit & (t > 0)
    P = np.full((u.size, 3), np.nan)
    P[ok] = origin + d[ok] * t[ok, None]
    return P, ok


def main():
    run = tb.run_dir()
    frames = tb.results_with_camera(run)
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    item = frames[idx]
    k = tb.intrinsics(run)
    mount = tb.load_mount(tb.PRIOR_MOUNT)
    R_cam, cam = tb.camera_pose(item['pose'], mount)
    img = tb.read_color(item['color'])
    m = json.loads((run / 'can_measure.json').read_text())
    axis = m['can']['axis_base_xy']
    r = m['can']['radius_mm']
    z_table = m['table']['z_base_mm']

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dark = (hsv[:, :, 2] < 70).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    h, w = dark.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    pads = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 2500:
            continue
        ys, xs = np.where(lab == i)
        pads.append({'label': int(i), 'area': int(st[i, cv2.CC_STAT_AREA]),
                     'bbox_uv': [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())],
                     'touches_frame_bottom': bool(ys.max() >= h - 3)})
    out = {'frame': item['color'].name, 'pose': item['pose'],
           'camera_base': np.round(cam, 1).tolist(), 'can_axis': axis, 'can_r': r,
           'z_table': z_table, 'pads': pads}
    # per pad: where its pixels land on the can cylinder
    detail = []
    for p in pads:
        ys, xs = np.where(lab == p['label'])
        P, ok = ray_cylinder(cam, R_cam, xs.astype(float), ys.astype(float), k, axis, r)
        if ok.sum() < 50:
            continue
        Q = P[ok]
        detail.append({'label': p['label'], 'n_on_can': int(ok.sum()),
                       'z_above_table_p5_p50_p95': [round(float(np.percentile(Q[:, 2] - z_table, q)), 1)
                                                    for q in (5, 50, 95)]})
    out['pads_on_can'] = detail
    print(json.dumps(out, ensure_ascii=False, indent=1))
    (run / 'grip_probe.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
