"""Fit the vertical can as a circle on the horizontal plane from its own depth points."""
import json, sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(r'G:\Flip Anything\ds_control')
sys.path.insert(0, str(ROOT / 'src'))
from flip_anything.geometry import pose_matrix  # noqa: E402


def fit_circle(pts):
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = np.sqrt(sol[2] + cx * cx + cy * cy)
    res = np.abs(np.hypot(x - cx, y - cy) - r)
    return cx, cy, r, res


def main():
    run = ROOT.parent / 'logs' / 'experiments' / 'ds_control' / sys.argv[1]
    idx = int(sys.argv[2])
    pose = [float(v) for v in sys.argv[3:9]]
    mount = json.loads((run / 'mount.json').read_text())
    R_ct = np.array(mount['R_tcp_cam']); t_ct = np.array(mount['t_tcp_cam_mm'])
    h_tab = float(mount['table_height_mm'])
    k = json.loads((run / 'intrinsics.json').read_text(encoding='utf-8-sig'))['payload']['metadata']['factory_camera_param']['color_intrinsic']
    fx, fy, cx, cy = k['fx'], k['fy'], k['cx'], k['cy']
    img = cv2.imread(str(run / ('frame_%06d_color.png' % idx)))
    dep = np.load(run / ('frame_%06d_depth_raw.npy' % idx)).astype(np.float32)
    h, w = dep.shape
    T = np.array(pose_matrix(pose))
    R_tool, tcp = T[:3, :3], T[:3, 3]
    R_cam = R_tool @ R_ct
    cam = tcp + R_tool @ t_ct

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 60, 45), (95, 255, 255)) > 0
    green = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(green.astype(np.uint8), 8)
    i = max(range(1, n), key=lambda q: st[q, cv2.CC_STAT_AREA])
    gm = (lab == i)
    ys, xs = np.where(gm)
    top = int(ys.min())
    body = gm.copy()
    body[:top + 6] = False          # drop the lid band
    vv, uu = np.where(body & (dep > 0))
    z = dep[vv, uu]
    P = np.stack([(uu - cx) * z / fx, (vv - cy) * z / fy, z], 1)
    base = (R_cam @ P.T).T + cam
    zc = base[:, 2]
    sel = (zc > h_tab + 15) & (zc < h_tab + 130)
    pts = base[sel][:, :2]
    print('body points', len(pts), 'z range', round(float(np.percentile(zc[sel], 5)), 1),
          round(float(np.percentile(zc[sel], 95)), 1))
    if len(pts) < 200:
        print('too few points'); return
    cx_, cy_, r_, res = fit_circle(pts)
    keep = res < max(3.0, np.percentile(res, 70))
    cx_, cy_, r_, res = fit_circle(pts[keep])
    print(json.dumps({'axis_base': [round(float(cx_), 1), round(float(cy_), 1)],
                      'radius_mm': round(float(r_), 1), 'diameter_mm': round(float(2 * r_), 1),
                      'n_used': int(keep.sum()), 'res_med_mm': round(float(np.median(res)), 2),
                      'res_p95_mm': round(float(np.percentile(res, 95)), 2),
                      'table_h': round(h_tab, 1)}, ensure_ascii=False))

    # lid height: metal pixels above the body, nearest quarter
    lidv, lidu = np.where((dep > 0) & (np.arange(h)[:, None] < top + 4) &
                          (np.arange(h)[:, None] > top - 60) &
                          (np.arange(w)[None, :] > xs.min() - 15) &
                          (np.arange(w)[None, :] < xs.max() + 15))
    if lidv.size:
        zl = dep[lidv, lidu]
        Pl = (R_cam @ np.stack([(lidu - cx) * zl / fx, (lidv - cy) * zl / fy, zl], 1).T).T + cam
        print('lid points', lidv.size, 'z p10/p50/p90 =',
              np.round(np.percentile(Pl[:, 2], [10, 50, 90]), 1),
              ' height above table p50 =', round(float(np.percentile(Pl[:, 2], 50) - h_tab), 1))


if __name__ == '__main__':
    main()

