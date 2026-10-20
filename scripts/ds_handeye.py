"""手眼标定（最终版）：
 A. 桌面平面偏移量 -> 直接线性解出 t_c 与桌面高度、深度尺度（不依赖特征匹配与偏航角）
 B. 动态掩膜后的光度比对 -> 解出偏航角（R_cb 第三列已由桌面法向给出）
 C. 独立校验：罐心在基座系应静止、罐轴应竖直
"""
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


def fit_plane(points, thresh=2.0, iters=1500, seed=3):
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


def bilinear(img, u, v):
    x0 = np.floor(u).astype(np.int32); y0 = np.floor(v).astype(np.int32)
    wx = (u - x0).astype(np.float32); wy = (v - y0).astype(np.float32)
    return (img[y0, x0]*(1-wx)*(1-wy) + img[y0, x0+1]*wx*(1-wy)
            + img[y0+1, x0]*(1-wx)*wy + img[y0+1, x0+1]*wx*wy)


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-6 or nb < 1e-6 else float(a @ b / (na * nb))


def main():
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    n = len(manifest)
    gray, depth, Rb, pb, ncam, dcam = [], [], [], [], [], []
    for f in manifest:
        b = Path(f['run']) / f['frame']
        gray.append(cv2.cvtColor(cv2.imread(str(b)+'_color.png'), cv2.COLOR_BGR2GRAY).astype(np.float32))
        depth.append(np.load(str(b)+'_depth_raw.npy').astype(np.float32))
        T = pose_matrix(f['pose']); Rb.append(T[:3, :3]); pb.append(T[:3, 3])
    h, w = gray[0].shape
    vv, uu = np.mgrid[0:h, 0:w]

    print('=== 桌面平面拟合 ===')
    for i in range(n):
        d = depth[i]
        sel = (d > 150) & (d < 1500) & (vv > 400)
        Z = d[sel]; X = (uu[sel]-K['cx'])*Z/K['fx']; Y = (vv[sel]-K['cy'])*Z/K['fy']
        cnt, nn, dd = fit_plane(np.stack([X, Y, Z], 1))
        if nn[1] > 0:
            nn, dd = -nn, -dd
        ncam.append(nn); dcam.append(dd)
        print(f'  {i:2d} {manifest[i]["name"][:16]:16s} 内点 {cnt:6d} n={np.round(nn,4)} d={dd:8.2f}')

    # A. 线性解 t_c / h / 深度尺度
    rows, rhs = [], []
    for i in range(n):
        u_i = Rb[i].T @ np.array([0.0, 0.0, 1.0])
        rows.append(np.concatenate([u_i, [-1.0, -dcam[i]]]))
        rhs.append(-pb[i][2])
    A, b = np.array(rows), np.array(rhs)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    tc_plane, h_table, s_plane = sol[:3], sol[3], sol[4]
    r = A @ sol - b
    print('=== A: 由桌面平面偏移解 t_c ===')
    print(f'  t_c = {np.round(tc_plane,2)} mm  |t_c| = {np.linalg.norm(tc_plane):.1f}')
    print(f'  桌面高度(基座系) h = {h_table:.1f} mm   深度尺度 s = {s_plane:.4f}')
    print(f'  残差 rms {np.sqrt(np.mean(r**2)):.2f} mm  最大 {np.abs(r).max():.2f} mm')

    # B. 动态掩膜 + 光度法求偏航
    same = [(i, j) for i in range(n) for j in range(i+1, n)
            if np.linalg.norm(pb[i]-pb[j]) < 1.5 and np.linalg.norm(Rb[i]-Rb[j]) < 0.01]
    dyn = np.zeros((h, w), bool)
    for i, j in same:
        dyn |= np.abs(gray[i]-gray[j]) > 22
        print(f'  同位姿对 {i}-{j} 用于动态掩膜, 动态像素 {int(dyn.sum())}')
    dyn = cv2.dilate(dyn.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)

    def samples(i):
        d = depth[i]
        sel = (d > 150) & (d < 1000) & (vv < 640) & (~dyn)
        sel &= (np.arange(h)[:, None] % 4 == 0) & (np.arange(w)[None, :] % 4 == 0)
        vi, ui = np.nonzero(sel)
        Z = d[vi, ui]
        return np.stack([(ui-K['cx'])*Z/K['fx'], (vi-K['cy'])*Z/K['fy'], Z], 1).astype(np.float32), gray[i][vi, ui]

    S = [samples(i) for i in range(n)]
    print('  每帧静态采样点:', [len(s[0]) for s in S])

    def score(i, j, R_rel, t_rel):
        P, a = S[i]
        Q = (R_rel @ P.T).T + t_rel[None, :]
        z = Q[:, 2]
        ok = z > 60
        u = K['fx']*Q[:, 0]/z + K['cx']; v = K['fy']*Q[:, 1]/z + K['cy']
        ok &= (u > 2) & (u < w-3) & (v > 2) & (v < h-3)
        return -1.0 if ok.sum() < 200 else ncc(a[ok], bilinear(gray[j], u[ok], v[ok]))

    n_up = np.mean([ncam[i] for i in range(n) if np.linalg.norm(Rb[i]-Rb[2]) < 0.01], axis=0)
    n_up /= np.linalg.norm(n_up)
    a_vec = np.cross(n_up, [0.0, 0.0, 1.0])
    if np.linalg.norm(a_vec) < 1e-6:
        a_vec = np.cross(n_up, [1.0, 0.0, 0.0])
    a_vec /= np.linalg.norm(a_vec); b_vec = np.cross(n_up, a_vec)

    def Rcb_of(psi):
        c1 = math.cos(psi)*a_vec + math.sin(psi)*b_vec
        return np.stack([c1, np.cross(n_up, c1), n_up], 1)

    pairs_t = [(i, j) for i in range(n) for j in range(i+1, n)
               if np.linalg.norm(Rb[i]-Rb[j]) < 0.01 and 1 < np.linalg.norm(pb[j]-pb[i]) < 200]
    best = None
    for psi in np.arange(0, 2*math.pi, math.radians(2)):
        Rcb = Rcb_of(psi)
        for sc in (0.93, 0.96, 0.99, 1.02):
            v = sum(score(i, j, np.eye(3), Rcb @ (sc*(pb[i]-pb[j]))) for i, j in pairs_t)
            if best is None or v > best[0]:
                best = (v, float(psi), sc)
    for psi in np.arange(best[1]-math.radians(4), best[1]+math.radians(4), math.radians(0.1)):
        Rcb = Rcb_of(psi)
        for sc in np.arange(0.90, 1.05, 0.002):
            v = sum(score(i, j, np.eye(3), Rcb @ (sc*(pb[i]-pb[j]))) for i, j in pairs_t)
            if v > best[0]:
                best = (v, float(psi), float(sc))
    Rcb = Rcb_of(best[1]); s_photo = best[2]
    Rc = Rb[0].T @ Rcb.T
    print('=== B: 光度法求偏航 ===')
    print(f'  psi = {math.degrees(best[1]) % 360:.2f} deg   光度深度尺度 = {s_photo:.4f}  '
          f'平均NCC = {best[0]/len(pairs_t):.4f}')
    for i, j in pairs_t:
        print(f'    {i}->{j} NCC {score(i, j, np.eye(3), Rcb @ (s_photo*(pb[i]-pb[j]))):.4f}')
    print('  R_cam_base:\n', np.round(Rcb, 4))
    print('  R_tcp_cam:\n', np.round(Rc, 4))

    # C. 校验：罐心在基座系是否静止
    print('=== C: 独立校验（罐心在基座系应静止）===')
    can_centers = []
    for i in range(n):
        d = depth[i]
        hsv = cv2.cvtColor(cv2.imread(str(Path(manifest[i]['run'])/manifest[i]['frame'])+'_color.png'), cv2.COLOR_BGR2HSV)
        g = cv2.inRange(hsv, (35, 60, 50), (95, 255, 255))
        g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cnt, lab, st, ct = cv2.connectedComponentsWithStats(g)
        if cnt < 2:
            continue
        k = 1 + int(np.argmax(st[1:, 4]))
        if st[k, 4] < 4000:
            continue
        m = (lab == k) & (d > 0)
        if m.sum() < 500:
            continue
        Z = d[m]; U = uu[m]; V = vv[m]
        P = np.stack([(U-K['cx'])*Z/K['fx'], (V-K['cy'])*Z/K['fy'], Z], 1)
        Pc = P.mean(0)
        base = pb[i] + Rb[i] @ (Rc @ Pc + tc_plane)
        can_centers.append((i, base))
    if len(can_centers) > 2:
        arr = np.array([c[1] for c in can_centers])
        print('  罐心(基座系)均值', np.round(arr.mean(0), 1), ' 各帧散布 std', np.round(arr.std(0), 2))
        for i, c in can_centers:
            print(f'    frame {i:2d} {np.round(c,1)}')

    out = {'R_cam_base': Rcb.tolist(), 'R_tcp_cam': Rc.tolist(),
           't_tcp_cam_mm': tc_plane.tolist(),
           't_tcp_cam_photometric_mm': None,
           'table_height_mm': float(h_table), 'depth_scale_plane': float(s_plane),
           'depth_scale_photometric': float(s_photo),
           'yaw_deg': float(math.degrees(best[1]) % 360),
           'plane_fit_residual_rms_mm': float(np.sqrt(np.mean(r**2))),
           'n_cam_per_frame': [x.tolist() for x in ncam],
           'd_cam_per_frame': [float(x) for x in dcam],
           'mean_ncc_yaw': float(best[0]/len(pairs_t))}
    Path(sys.argv[2]).write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('written', sys.argv[2])


main()
