"""Check depth coverage of the can lid and estimate the top surface height."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb


def main():
    run = tb.run_dir()
    item = tb.results_with_camera(run)[-1]
    k = tb.intrinsics(run)
    mount = tb.load_mount(tb.PRIOR_MOUNT)
    R_cam, cam = tb.camera_pose(item['pose'], mount)
    depth = tb.load_depth(item['depth'])
    m = json.loads((run / 'can_measure.json').read_text())
    cx, cy = m['can']['axis_base_xy']
    r = m['can']['radius_mm']
    z_table = m['table']['z_base_mm']

    # pixels whose ray passes within r of the axis, above the table: candidates for the lid
    us, vs = np.meshgrid(np.arange(560, 760), np.arange(150, 320))
    us = us.ravel().astype(float)
    vs = vs.ravel().astype(float)
    d = depth[vs.astype(int), us.astype(int)].astype(float)
    ok = d > 0
    pc = np.stack([(us[ok] - k['cx']) * d[ok] / k['fx'],
                   (vs[ok] - k['cy']) * d[ok] / k['fy'], d[ok]], 1)
    P = pc @ R_cam.T + cam
    out = {'n_valid': int(ok.sum()), 'n_box': int(us.size)}
    # points near the can axis tube and high above the table
    rad = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
    tube = rad < r + 3
    out['n_tube'] = int(tube.sum())
    if tube.sum():
        zs = P[tube, 2]
        out['tube_z_p50_p90_max'] = [round(float(np.percentile(zs, 50)), 1),
                                     round(float(np.percentile(zs, 90)), 1),
                                     round(float(zs.max()), 1)]
        lid = zs > z_table + 90
        out['n_lid'] = int(lid.sum())
        if lid.sum() > 20:
            out['lid_z_p10_p50_p90'] = [round(float(np.percentile(zs[lid], q)), 1) for q in (10, 50, 90)]
            out['lid_height_above_table_p50'] = round(float(np.median(zs[lid]) - z_table), 1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
