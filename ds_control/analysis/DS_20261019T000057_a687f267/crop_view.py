"""Crop + annotated depth checks for the can top, saved next to this run."""
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
    u0, u1, v0, v1 = (int(v) for v in sys.argv[1:5])
    crop = img[v0:v1, u0:u1]
    scale = max(1, int(min(6, 1100 // max(1, crop.shape[1]))))
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    path = run / ('crop_%s_%d_%d.png' % (item['color'].stem, u0, v0))
    cv2.imwrite(str(path), big)

    # vertical profile of base-frame depth through the can axis
    kx = json.loads((run / 'can_measure.json').read_text())['can']['axis_base_xy']
    T = np.array(tb.pose_matrix(item['pose']))
    R_tool, tcp = T[:3, :3], T[:3, 3]
    p_cam = (R_cam.T @ (np.array([kx[0], kx[1], 0.0]) - cam))
    # project the axis column: solve for pixel of a point above the axis
    rows = []
    for z in (0, 20, 40, 60, 80, 100, 120, 140):
        P = np.array([kx[0], kx[1], z], dtype=float)
        pc = R_cam.T @ (P - cam)
        u = k['fx'] * pc[0] / pc[2] + k['cx']
        v = k['fy'] * pc[1] / pc[2] + k['cy']
        rows.append({'z_base': z, 'u': round(float(u), 1), 'v': round(float(v), 1),
                     'depth_px': int(depth[int(np.clip(v, 0, depth.shape[0] - 1)),
                                           int(np.clip(u, 0, depth.shape[1] - 1))])})
    print(json.dumps({'crop': str(path), 'scale': scale, 'axis_projection': rows},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
