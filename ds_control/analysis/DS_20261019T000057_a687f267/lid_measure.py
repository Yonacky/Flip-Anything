"""Estimate the can top from the horizontal lid disc (points strictly inside the radius)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb


def main():
    run = tb.run_dir()
    item = tb.results_with_camera(run)[-1]
    k = tb.intrinsics(run)
    mount = tb.load_mount(tb.PRIOR_MOUNT)
    R_cam, cam = tb.camera_pose(item['pose'], mount)
    img = tb.read_color(item['color'])
    depth = tb.load_depth(item['depth'])
    m = json.loads((run / 'can_measure.json').read_text())
    cx, cy = m['can']['axis_base_xy']
    r = m['can']['radius_mm']
    z_table = m['table']['z_base_mm']

    us, vs = np.meshgrid(np.arange(570, 750), np.arange(170, 320))
    us = us.ravel().astype(float)
    vs = vs.ravel().astype(float)
    d = depth[vs.astype(int), us.astype(int)].astype(float)
    ok = d > 0
    pc = np.stack([(us[ok] - k['cx']) * d[ok] / k['fx'],
                   (vs[ok] - k['cy']) * d[ok] / k['fy'], d[ok]], 1)
    P = pc @ R_cam.T + cam
    rad = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
    inner = rad < r - 6.0
    zs = P[inner, 2]
    out = {'n_inner': int(inner.sum())}
    if inner.sum() > 30:
        out['inner_z_p'] = {q: round(float(np.percentile(zs, q)), 1) for q in (5, 25, 50, 75, 95)}
        out['inner_rad_p'] = {q: round(float(np.percentile(rad[inner], q)), 1) for q in (50, 90, 99)}
        # horizontal-plane test: the disc must be flat in z
        out['inner_z_iqr'] = round(float(np.percentile(zs, 75) - np.percentile(zs, 25)), 1)

    # bright metal lid ellipse: low saturation, high value, inside the can bbox
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    metal = ((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 110)).astype(np.uint8)
    metal[:150, :] = 0
    metal[330:, :] = 0
    metal[:, :560] = 0
    metal[:, 760:] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(metal, 8)
    cand = None
    if n > 1:
        i = max(range(1, n), key=lambda q: st[q, cv2.CC_STAT_AREA])
        mask = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        if len(c) >= 5:
            (ex, ey), (MA, ma), ang = cv2.fitEllipse(c)
            cand = {'center_uv': [round(ex, 1), round(ey, 1)],
                    'axes_px': [round(MA, 1), round(ma, 1)], 'angle_deg': round(ang, 1),
                    'contour_px': int(len(c)),
                    'semi_ratio': round(float(min(MA, ma) / max(MA, ma)), 3)}
    out['lid_ellipse'] = cand
    print(json.dumps(out, ensure_ascii=False, indent=1))
    (run / 'lid_measure.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
