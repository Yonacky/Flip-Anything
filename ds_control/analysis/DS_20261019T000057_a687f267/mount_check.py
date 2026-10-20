"""Check the historical camera->TCP mount against THIS round's own observations.

For every recorded frame it fits the table plane from depth, converts it to base
coordinates with the prior mount, then reports:
  * plane tilt in base (a horizontal table must come out ~0 deg),
  * implied table height (must be the same constant across poses),
  * residual between the fitted camera-plane distance and the distance predicted
    from the mount plus the measured table height.
A mount that is wrong in rotation shows up as tilt; wrong in translation as a
height that drifts with pose.
"""
import json
import sys

import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import tb


def main():
    run = tb.run_dir()
    k = tb.intrinsics(run)
    prior = tb.load_mount(tb.PRIOR_MOUNT)
    print('prior mount: t_tcp_cam =', np.round(prior['t_tcp_cam_mm'], 2),
          ' table_z =', round(prior['table_height_mm'], 2))
    print('prior R_tcp_cam:\n', np.round(prior['R_tcp_cam'], 4))
    rows = []
    for item in tb.results_with_camera(run):
        depth = tb.load_depth(item['depth'])
        plane = tb.fit_plane(depth, k, lo=150.0, hi=800.0)
        if plane is None:
            print(item['color'].name, 'no plane')
            continue
        R_cam, cam = tb.camera_pose(item['pose'], prior)
        base = tb.plane_in_base(plane, R_cam, cam)
        rows.append({'frame': item['color'].name, 'pose': np.round(item['pose'], 1).tolist(),
                     'camera_base': np.round(cam, 1).tolist(),
                     'plane_inliers': plane['inliers'], 'res_p95': round(plane['res_p95'], 1),
                     'camera_plane_dist': round(plane['dist'], 1),
                     'tilt_deg': round(base['tilt_deg'], 2),
                     'table_z': round(base['z_at_origin'], 1),
                     'camera_height': round(float(cam[2] - base['z_at_origin']), 1)})
        print(json.dumps(rows[-1], ensure_ascii=False))
    if rows:
        print('table_z spread mm:', round(float(np.ptp([r['table_z'] for r in rows])), 1),
              ' tilt max deg:', round(max(r['tilt_deg'] for r in rows), 2))
    (tb.RUN / 'mount_check.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
