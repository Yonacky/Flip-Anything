"""Solve camera->TCP from table-plane observations across diverse tool poses.



Only camera + arm kinematics + the table (a natural horizontal plane) are used.

Outputs mount.json with R_tcp_cam, t_tcp_cam and diagnostics.

"""

import json, math, sys

from pathlib import Path



import cv2

import numpy as np



ROOT = Path(r'G:\Flip Anything\ds_control')

sys.path.insert(0, str(ROOT / 'src'))

from flip_anything.geometry import pose_matrix  # noqa: E402



MAN = ROOT.parent / 'logs' / 'experiments' / 'ds_control' / 'DS_20261018T203231_b2ac5b99' / 'calib_manifest.json'





def intrinsics(run):

    d = json.loads((Path(run) / 'intrinsics.json').read_text(encoding='utf-8-sig'))

    return d['payload']['metadata']['factory_camera_param']['color_intrinsic']





def fit_plane(depth, k, lo=150.0, hi=800.0, iters=800, thresh=4.0, seed=7):

    h, w = depth.shape

    step = 3

    vv, uu = np.mgrid[0:h:step, 0:w:step]

    zz = depth[::step, ::step]

    sel = (zz > lo) & (zz < hi)

    if sel.sum() < 500:

        return None

    pts = np.stack([(uu[sel] - k['cx']) * zz[sel] / k['fx'],

                    (vv[sel] - k['cy']) * zz[sel] / k['fy'], zz[sel]], 1)

    rng = np.random.default_rng(seed)

    sub = pts[rng.choice(len(pts), min(20000, len(pts)), replace=False)]

    best = None

    for _ in range(iters):

        a, b, c = sub[rng.choice(len(sub), 3, replace=False)]

        n = np.cross(b - a, c - a)

        L = np.linalg.norm(n)

        if L < 1e-9:

            continue

        n = n / L

        d = -n @ a

        cnt = int((np.abs(sub @ n + d) < thresh).sum())

        if best is None or cnt > best[0]:

            best = (cnt, n, d)

    cnt, n, d = best

    inl = np.abs(pts @ n + d) < thresh

    if inl.sum() < 300:

        return None

    m = pts[inl].mean(0)

    n = np.linalg.svd(pts[inl] - m)[2][-1]

    d = -n @ m

    if n[2] > 0:                      # normal must point back toward the camera

        n, d = -n, -d

    res = np.abs(pts[inl] @ n + d)

    return {'n': n, 'd': float(d), 'dist': float(abs(d)), 'inliers': int(inl.sum()),

            'res_p95': float(np.percentile(res, 95))}





def main():

    entries = json.loads(MAN.read_text())

    samples = []

    for e in entries:

        run = Path(e['run'])

        if not e.get('frame'):

            print('skip (no frame):', e['name'])

            continue

        idx = int(e['frame'].split('_')[1])

        depth = np.load(run / ('frame_%06d_depth_raw.npy' % idx)).astype(np.float32)

        k = intrinsics(run)

        pl = fit_plane(depth, k)

        if pl is None:

            print('skip (no plane):', e['name'])

            continue

        T = np.array(pose_matrix(e['pose']))

        if pl['inliers'] < 5000 or not (140.0 <= pl['dist'] <= 420.0):

            print('skip (implausible plane):', e['name'], pl['inliers'], round(pl['dist'], 1))

            continue

        samples.append({'name': e['name'], 'R': T[:3, :3], 'tcp': T[:3, 3], 'plane': pl})

        print(f"{e['name']:16s} plane inliers={pl['inliers']:6d} dist={pl['dist']:7.1f} "

              f"n={np.round(pl['n'],3)} res95={pl['res_p95']:.1f}")

    if len(samples) < 4:

        raise SystemExit('not enough plane samples')



    # 1) rotation: R_ct maps camera normal -> tool-frame "up"

    up = np.array([0.0, 0.0, 1.0])

    M = np.zeros((3, 3))

    for s in samples:

        u = s['R'].T @ up           # base up expressed in the tool frame

        M += np.outer(u, s['plane']['n'])

    U, _, Vt = np.linalg.svd(M)

    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])

    R_ct = U @ D @ Vt

    print('R_tcp_cam =\n', np.round(R_ct, 4))



    # 2) translation + table height:  u_i . c - h = -(dist_i + tcp_z_i)

    A = np.zeros((len(samples), 4))

    b = np.zeros(len(samples))

    for i, s in enumerate(samples):

        u = s['R'].T @ up

        A[i, :3] = u

        A[i, 3] = -1.0

        b[i] = s['plane']['dist'] - s['tcp'][2]

    keep = np.ones(len(A), bool)

    for _ in range(6):

        sol, *_ = np.linalg.lstsq(A[keep], b[keep], rcond=None)

        r_all = A @ sol - b

        new = np.abs(r_all) < max(12.0, 3.0 * np.std(r_all[keep]))

        if new.sum() < 4 or (new == keep).all():

            keep = new if new.sum() >= 4 else keep

            break

        keep = new

    A, b = A[keep], b[keep]

    samples = [x for x, kp in zip(samples, keep) if kp]

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)

    c = sol[:3]

    h = float(sol[3])

    pred = A @ sol

    resid = pred - b

    print(f't_tcp_cam = {np.round(c,2)} mm   table height h = {h:.1f} mm')

    print(f'height residual: mean {resid.mean():.2f} mm  std {resid.std():.2f} mm  max {np.abs(resid).max():.2f} mm')

    # per-sample check

    print('per-sample height check:')

    for s, r in zip(samples, resid):

        print(f"  {s['name']:16s} residual {r:+.2f} mm")

    # angular consistency of the plane normal under R_ct

    ang_err = []

    for s in samples:

        u = s['R'].T @ up

        n_pred = R_ct.T @ u

        ang_err.append(math.degrees(math.acos(max(-1, min(1, float(n_pred @ s['plane']['n']))))))

    print(f'normal angular residual: mean {np.mean(ang_err):.2f} deg  max {np.max(ang_err):.2f} deg')



    out = {'R_tcp_cam': R_ct.tolist(), 't_tcp_cam_mm': c.tolist(), 'table_height_mm': h,

           'n_samples': len(samples), 'height_resid_std_mm': float(resid.std()),

           'normal_resid_max_deg': float(np.max(ang_err))}

    dst = ROOT.parent / 'logs' / 'experiments' / 'ds_control' / 'DS_20261018T203231_b2ac5b99' / 'mount.json'

    dst.write_text(json.dumps(out, indent=1))

    print('written', dst)





if __name__ == '__main__':

    main()

