"""Solve the can top height from the projected top-rim circle.

The rim is a horizontal circle of the measured radius around the measured axis, so
its image ellipse is fully determined by z_top. Both the lowest (near) and highest
(far) projected rim points are matched against what is visible in the frame.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb


def project(P, k, R_cam, cam):
    pc = R_cam.T @ (np.asarray(P, dtype=float) - cam)
    return (k['fx'] * pc[0] / pc[2] + k['cx'], k['fy'] * pc[1] / pc[2] + k['cy'])


def main():
    run = tb.run_dir()
    item = tb.results_with_camera(run)[-1]
    k = tb.intrinsics(run)
    mount = tb.load_mount(tb.PRIOR_MOUNT)
    R_cam, cam = tb.camera_pose(item['pose'], mount)
    m = json.loads((run / 'can_measure.json').read_text())
    cx, cy = m['can']['axis_base_xy']
    r = m['can']['radius_mm']
    v_far = float(sys.argv[1]) if len(sys.argv) > 1 else 182.0
    v_near = float(sys.argv[2]) if len(sys.argv) > 2 else 260.0
    u_left = float(sys.argv[3]) if len(sys.argv) > 3 else 582.0
    u_right = float(sys.argv[4]) if len(sys.argv) > 4 else 723.0
    ang = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    rows = []
    for z in np.arange(90.0, 190.0, 0.5):
        ring = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang), np.full_like(ang, z)], 1)
        uv = np.array([project(p, k, R_cam, cam) for p in ring])
        rows.append({'z': float(z), 'v_min': float(uv[:, 1].min()), 'v_max': float(uv[:, 1].max()),
                     'u_min': float(uv[:, 0].min()), 'u_max': float(uv[:, 0].max())})
    arr = rows
    best_v = min(arr, key=lambda q: abs(q['v_min'] - v_far) + abs(q['v_max'] - v_near))
    best_u = min(arr, key=lambda q: abs(q['u_min'] - u_left) + abs(q['u_max'] - u_right))
    z_table = m['table']['z_base_mm']
    out = {'observed': {'v_far': v_far, 'v_near': v_near, 'u_left': u_left, 'u_right': u_right},
           'z_top_from_rim_v_mm': best_v['z'], 'rim_v_residual_px': [round(best_v['v_min'] - v_far, 1),
                                                                    round(best_v['v_max'] - v_near, 1)],
           'z_top_from_rim_u_mm': best_u['z'],
           'rim_u_at_that_z': [round(best_v['u_min'], 1), round(best_v['u_max'], 1)],
           'table_z_mm': z_table,
           'height_above_table_mm': round(best_v['z'] - z_table, 1),
           'can_height_mm': round(best_v['z'] - z_table, 1),
           'axis_base_xy': [cx, cy], 'diameter_mm': m['can']['diameter_mm']}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    (run / 'rim_solve.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
