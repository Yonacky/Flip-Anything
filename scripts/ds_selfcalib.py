"""自标定求解（抗差版）：机械臂位姿读数 + RGB-D 静止特征 -> 相机<->TCP<->基座。"""
import json, math, sys
from pathlib import Path
import cv2
import numpy as np

K = dict(fx=691.4324951171875, fy=691.7568359375, cx=643.6455078125, cy=358.8835144042969)


def pose_matrix(pose):
    x, y, z, roll, pitch, yaw = pose
    r, p, q = map(math.radians, (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(q), math.sin(q)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, x],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, y],
                     [-sp, cp*sr, cp*cr, z], [0, 0, 0, 1]])


def fit_plane(points, thresh=2.0, iters=1200, seed=3):
    rng = np.random.default_rng(seed)
    S = points[rng.choice(len(points), min(40000, len(points)), replace=False)]
    best = None
    for _ in range(iters):
        a, b, c = S[rng.choice(len(S), 3, replace=False)]
        n = np.cross(b - a, c - a); L = np.linalg.norm(n)
        if L < 1e-9:
            continue
        n = n / L; d = -n @ a
        cnt = int((np.abs(S @ n + d) < thresh).sum())
        if best is None or cnt > best[0]:
            best = (cnt, n, d)
    return best


def irls(A, b, sigma, iters=8):
    w = np.ones(A.shape[0])
    for _ in range(iters):
        W = np.sqrt(w)[:, None]
        sol, *_ = np.linalg.lstsq(A * W, b * np.sqrt(w), rcond=None)
        r = A @ sol - b
        w = 1.0 / (1.0 + (r / sigma) ** 2)
    return sol, r, w


def main():
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    gray, depths = [], []
    for f in manifest:
        base = Path(f['run']) / f['frame']
        gray.append(cv2.cvtColor(cv2.imread(str(base) + '_color.png'), cv2.COLOR_BGR2GRAY))
        depths.append(np.load(str(base) + '_depth_raw.npy').astype(np.float32))
    nfr = len(manifest)
    print(nfr, 'frames')

    def seed(fi):
        d = depths[fi]
        mask = np.zeros(d.shape, np.uint8)
        mask[(d > 200) & (d < 900)] = 255
        mask[650:, :] = 0
        pts = cv2.goodFeaturesToTrack(gray[fi], maxCorners=600, qualityLevel=0.005,
                                      minDistance=7, blockSize=7, mask=mask)
        return [] if pts is None else [(float(u), float(v)) for u, v in pts.reshape(-1, 2)]

    lk = dict(winSize=(41, 41), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.005))
    all_tracks = [[(0, u, v)] for u, v in seed(0)]
    active = list(range(len(all_tracks)))
    pts = np.array([[t[0][1], t[0][2]] for t in all_tracks], np.float32)
    print('frame 0: seeded', len(all_tracks))
    for i in range(1, nfr):
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(gray[i-1], gray[i], pts.reshape(-1, 1, 2), None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray[i], gray[i-1], nxt, None, **lk)
        fb = np.linalg.norm(back.reshape(-1, 2) - pts, axis=1)
        keep = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.5)
        new_active, new_pts = [], []
        for idx, good in enumerate(keep):
            if good:
                ti = active[idx]
                all_tracks[ti].append((i, float(nxt[idx, 0, 0]), float(nxt[idx, 0, 1])))
                new_active.append(ti); new_pts.append(nxt[idx, 0])
        active, pts = new_active, (np.array(new_pts, np.float32) if new_pts else np.zeros((0, 2), np.float32))
        added = 0
        if len(pts) < 250:
            occ = pts
            for u, v in seed(i):
                if len(occ) and np.min(np.linalg.norm(occ - [u, v], axis=1)) < 7:
                    continue
                all_tracks.append([(i, u, v)]); active.append(len(all_tracks) - 1); added += 1
                occ = np.vstack([occ, [[u, v]]]) if len(occ) else np.array([[u, v]], np.float32)
            pts = np.array([[all_tracks[ti][-1][1], all_tracks[ti][-1][2]] for ti in active], np.float32)
        print(f'  frame {i}: alive {len(new_active) - added}, new {added}, tracks total {len(all_tracks)}')

    def depth3d(fi, u, v):
        d = depths[fi]; h, w = d.shape
        ui, vi = int(round(u)), int(round(v))
        if not (6 <= ui < w-6 and 6 <= vi < h-6):
            return None
        patch = d[vi-3:vi+4, ui-3:ui+4]
        ok = patch[(patch > 150) & (patch < 1500)]
        if ok.size < 25 or ok.std() > 2.5:
            return None
        z = float(np.median(ok))
        return np.array([(u-K['cx'])*z/K['fx'], (v-K['cy'])*z/K['fy'], z])

    obs = {}
    for t, trk in enumerate(all_tracks):
        entry = {}
        for fi, u, v in trk:
            P = depth3d(fi, u, v)
            if P is not None:
                entry[fi] = P
        if len(entry) >= 2:
            obs[t] = entry
    print('features with >=2 valid 3D obs:', len(obs))

    Rb = [pose_matrix(f['pose'])[:3, :3] for f in manifest]
    pb = [np.array(f['pose'][:3]) for f in manifest]

    def same_orientation(a, b):
        cos = (np.trace(a.T @ b) - 1.0) / 2.0
        return math.degrees(math.acos(max(-1.0, min(1.0, cos)))) < 0.02

    rows, rhs = [], []
    deltas = []
    for fi in range(nfr):
        for fj in range(fi+1, nfr):
            if not same_orientation(Rb[fi], Rb[fj]):
                continue
            d = pb[fj] - pb[fi]
            common = [e for e in obs.values() if fi in e and fj in e]
            if not common or float(np.linalg.norm(d)) <= 1.0:
                continue
            mag = np.linalg.norm([e[fj] - e[fi] for e in common], axis=1)
            deltas.append(d)
            print(f'  pair {fi}->{fj} |d|={np.linalg.norm(d):6.1f} n={len(common):4d} '
                  f'median|dp|={np.median(mag):7.2f} p10={np.percentile(mag,10):6.2f} p90={np.percentile(mag,90):6.2f}')
            for e in common:
                dp = e[fj] - e[fi]
                for m in range(3):
                    row = np.zeros(9); row[m*3:(m+1)*3] = d
                    rows.append(row); rhs.append(dp[m])
    if deltas:
        print('  平移方向秩:', np.linalg.matrix_rank(np.array(deltas), tol=1e-6), '/', len(deltas))
    A, b = np.array(rows), np.array(rhs)
    Mv, res1, w1 = irls(A, b, sigma=2.0)
    Mv = Mv.reshape(3, 3)
    print(f'阶段1: {A.shape[0]} 方程, 有效权重>0.5 的比例 {float((w1>0.5).mean()):.3f}, '
          f'加权 rms {float(np.sqrt(np.average(res1**2, weights=w1))):.3f} mm')
    U, S, Vt = np.linalg.svd(-Mv)
    Rcb = U @ Vt
    if np.linalg.det(Rcb) < 0:
        U[:, -1] *= -1; Rcb = U @ Vt
    print('  svd:', np.round(S, 4))
    print('R_cam_base:\n', np.round(Rcb, 4))
    Rc = Rb[0].T @ Rcb.T
    print('R_tcp_cam:\n', np.round(Rc, 4))

    feats = sorted(obs.items(), key=lambda kv: -len(kv[1]))[:300]
    nf = len(feats)
    rows2, rhs2 = [], []
    for k, (t, entry) in enumerate(feats):
        for fi, P in entry.items():
            blk = np.zeros((3, 3*nf + 3)); blk[:, 3*k:3*k+3] = Rb[fi].T; blk[:, 3*nf:3*nf+3] = -np.eye(3)
            rows2.append(blk); rhs2.append(Rc @ P + Rb[fi].T @ pb[fi])
    A2, b2 = np.vstack(rows2), np.concatenate(rhs2)
    sol, res2, w2 = irls(A2, b2, sigma=2.0)
    tc = sol[3*nf:3*nf+3]
    print(f'阶段2: 特征 {nf}, 方程 {A2.shape[0]}, 有效权重>0.5 比例 {float((w2>0.5).mean()):.3f}')
    print('t_c (camera origin in tcp frame) mm:', np.round(tc, 2), '|t_c|', round(float(np.linalg.norm(tc)), 2))
    print(f'  加权 rms {float(np.sqrt(np.average(res2**2, weights=w2))):.3f} mm')

    T = np.eye(4); T[:3, :3] = Rc; T[:3, 3] = tc
    Tinv = np.linalg.inv(T)
    errs, perr = [], []
    for k, (t, entry) in enumerate(feats):
        Pk = np.append(sol[3*k:3*k+3], 1.0)
        for fi, P in entry.items():
            Pc = (Tinv @ np.linalg.inv(pose_matrix(manifest[fi]['pose'])) @ Pk)[:3]
            if Pc[2] <= 1:
                continue
            uv = np.array([Pc[0]*K['fx']/Pc[2] + K['cx'], Pc[1]*K['fy']/Pc[2] + K['cy']])
            errs.append(np.linalg.norm(uv - [P[0]*K['fx']/P[2] + K['cx'], P[1]*K['fy']/P[2] + K['cy']]))
            perr.append(np.linalg.norm(Pc - P))
    print(f'重投影 px: rms {np.sqrt(np.mean(np.square(errs))):.3f} 中位 {np.median(errs):.3f}  '
          f'| 三维点误差 mm: rms {np.sqrt(np.mean(np.square(perr))):.2f} 中位 {np.median(perr):.2f}')

    fi = nfr - 1
    d = depths[fi]; h, w = d.shape
    vv, uu = np.mgrid[0:h, 0:w]
    sel = (d > 150) & (d < 900) & (vv > 380)
    Z = d[sel]; X = (uu[sel]-K['cx'])*Z/K['fx']; Y = (vv[sel]-K['cy'])*Z/K['fy']
    cnt, n, dd = fit_plane(np.stack([X, Y, Z], 1))
    if n[2] > 0:
        n, dd = -n, -dd
    n_base = Rb[fi] @ Rc @ n
    Tb = pose_matrix(manifest[fi]['pose'])
    h_table = float(n_base @ (Tb[:3, 3] + Tb[:3, :3] @ tc) + dd)
    tilt = math.degrees(math.acos(min(1.0, abs(float(n_base[2])))))
    print(f'独立校验: 桌面平面内点 {cnt}, 相机距桌面 {abs(dd):.1f} mm')
    print('  基座中的法向:', np.round(n_base, 4), '-> 偏离竖直', round(tilt, 2), 'deg')
    print('  隐含桌面高度(基座系):', round(h_table, 1), 'mm')

    out = {'frames': manifest, 'R_cam_base': Rcb.tolist(), 'R_tcp_cam': Rc.tolist(),
           't_tcp_cam_mm': tc.tolist(), 'n_features': nf,
           'reprojection_rms_px': float(np.sqrt(np.mean(np.square(errs)))),
           'point_rms_mm': float(np.sqrt(np.mean(np.square(perr)))),
           'table_normal_base': n_base.tolist(), 'table_tilt_deg': tilt,
           'table_height_mm': h_table}
    Path(sys.argv[2]).write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('written', sys.argv[2])


main()
