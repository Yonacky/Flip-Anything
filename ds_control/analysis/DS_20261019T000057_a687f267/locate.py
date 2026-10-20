"""Rough localization of the red can from one recorded frame (read-only)."""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import tb


def red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 90, 60), (12, 255, 255))
    m2 = cv2.inRange(hsv, (168, 90, 60), (180, 255, 255))
    m = (m1 | m2) > 0
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n < 2:
        return None
    i = max(range(1, n), key=lambda q: st[q, cv2.CC_STAT_AREA])
    return lab == i


def main():
    run = tb.run_dir()
    frames = tb.results_with_camera(run)
    item = frames[-1] if len(sys.argv) < 2 else frames[int(sys.argv[1])]
    k = tb.intrinsics(run)
    mount = tb.load_mount(tb.PRIOR_MOUNT)
    R_cam, cam = tb.camera_pose(item['pose'], mount)
    img = tb.read_color(item['color'])
    depth = tb.load_depth(item['depth'])
    m = red_mask(img)
    ys, xs = np.where(m)
    vv, uu = np.where(m & (depth > 0))
    out = {'frame': item['color'].name, 'pose': item['pose'],
           'camera_base': np.round(cam, 1).tolist(),
           'mask_px': int(m.sum()), 'bbox_uv': [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())],
           'valid_depth_frac_on_mask': round(float(vv.size / max(1, m.sum())), 3)}
    if vv.size > 50:
        pts = tb.backproject(depth, k, R_cam, cam, uu, vv)
        out['depth_med_mm'] = round(float(np.median(depth[vv, uu])), 1)
        out['base_xy_med'] = [round(float(np.median(pts[:, 0])), 1), round(float(np.median(pts[:, 1])), 1)]
        out['base_z_range'] = [round(float(np.percentile(pts[:, 2], 5)), 1),
                              round(float(np.percentile(pts[:, 2], 95)), 1)]
        out['pts_n'] = int(vv.size)
    # full-resolution silhouette tangency without depth (needs an assumed radius; report only)
    out['width_px'] = int(xs.max() - xs.min()) if xs.size else None
    out['height_px_visible'] = int(ys.max() - ys.min()) if ys.size else None
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
