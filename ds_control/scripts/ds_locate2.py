"""Localize the target can in base coordinates using the freshly solved mount."""
import json, sys, math
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(r'G:\Flip Anything\ds_control')
sys.path.insert(0, str(ROOT / 'src'))
from flip_anything.geometry import pose_matrix  # noqa: E402


def main():
    run = ROOT.parent / 'logs' / 'experiments' / 'ds_control' / sys.argv[1]
    idx = int(sys.argv[2])
    pose = [float(v) for v in sys.argv[3:9]]
    r_can = float(sys.argv[9]) if len(sys.argv) > 9 else 33.0
    mount = json.loads((run / 'mount.json').read_text())
    R_ct = np.array(mount['R_tcp_cam']); t_ct = np.array(mount['t_tcp_cam_mm'])
    h_table = float(mount['table_height_mm'])
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
    gm = lab == i
    ys, xs = np.where(gm)
    top = int(ys.min()); bot = int(ys.max())
    out = {'pose': pose, 'camera_base': np.round(cam, 1).tolist(), 'table_h': h_table,
           'can_px': {'u': [int(xs.min()), int(xs.max())], 'v': [top, bot]}}
    d = dep[gm]
    dv = d[d > 0]
    out['depth_on_can'] = {'valid_frac': round(float(dv.size / gm.sum()), 3),
                           'med': round(float(np.median(dv)), 1) if dv.size else None}

    # depth inside the can bbox including the metal lid
    box = dep[max(0, top - 40):min(h, bot + 1), max(0, xs.min() - 10):min(w, xs.max() + 10)]
    bv = box[box > 0]
    out['depth_in_box'] = {'valid_frac': round(float(bv.size / box.size), 3),
                           'med': round(float(np.median(bv)), 1) if bv.size else None}

    def ray(u, v):
        d = R_cam @ np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        return d / np.linalg.norm(d)

    # silhouette tangency (no depth): axis from the two edge rays per row
    sols = []
    for v in range(top + 10, min(bot, h - 6), 6):
        row = np.where(gm[v])[0]
        if row.size < 40:
            continue
        h0 = ray(row.min(), v)[:2]; h1 = ray(row.max(), v)[:2]
        h0 /= np.linalg.norm(h0); h1 /= np.linalg.norm(h1)
        A = np.array([[-h0[1], h0[0]], [-h1[1], h1[0]]])
        hm = h0 + h1; hm /= np.linalg.norm(hm)
        for sgn in (1.0, -1.0):
            s = np.array([sgn * r_can, -sgn * r_can])
            try:
                C = np.linalg.solve(A, s + A @ cam[:2])
            except np.linalg.LinAlgError:
                continue
            if float((C - cam[:2]) @ hm) > 50.0:
                sols.append((v, C))
                break
    if sols:
        arr = np.array([s[1] for s in sols])
        med = np.median(arr, 0)
        out['axis_tangent_r%d' % int(r_can)] = [round(float(med[0]), 1), round(float(med[1]), 1)]
        out['axis_std'] = [round(float(arr[:, 0].std()), 2), round(float(arr[:, 1].std()), 2)]

    # top rim: topmost row centroid, back-projected at the box depth
    u_top = float(np.median(xs[ys == ys.min()]))
    if bv.size:
        z = float(np.median(bv))
        P = cam + R_cam @ np.array([(u_top - cx) * z / fx, (top - cy) * z / fy, z])
        out['rim_point_base'] = np.round(P, 1).tolist()
        out['can_height_above_table'] = round(float(P[2] - h_table), 1)
        out['implied_diameter_mm'] = round(float(2 * r_can), 1)
        width_px = float(xs.max() - xs.min())
        out['width_px'] = width_px
        # range from the camera to the can axis using the lid shrink factor
        out['range_est_mm'] = round(float(width_px and (2 * r_can * fx / width_px)), 1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

