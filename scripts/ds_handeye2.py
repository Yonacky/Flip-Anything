"""手眼标定 v2：以预测平面做目标化提取，充分利用大姿态多样性打破退化。
   n_cam = R_c^T (R_b^T ẑ)   与 t_c 无关，可由 R_c 直接预测
   d_cam = (p_b,z + u·t_c - h)/s
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


def ransac_plane(points, thresh=2.0, iters=1500, seed=3):
    rng = np.random.default_rng(seed)
    S = points[rng.choice(len(points), min(30000, len(points)), replace=False)]
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


def svd_plane(P):
    c = P.mean(0)
    C = np.cov((P - c).T)
    w, v = np.linalg.eigh(C)
    n = v[:, 0]
    return n, -float(n @ c), float(math.sqrt(max(w[0], 0.0)))


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
    gray, depth, Rb, pb = [], [], [], []
    for f in manifest:
        b = Path(f['run']) / f['frame']
        gray.append(cv2.cvtColor(cv2.imread(str(b)+'_color.png'), cv2.COLOR_BGR2GRAY).astype(np.float32))
        depth.append(np.load(str(b)+'_depth_raw.npy').astype(np.float32))
        T = pose_matrix(f['pose']); Rb.append(T[:3, :3]); pb.append(T[:3, 3])
    h_img, w_img = gray[0].shape
    vv, uu = np.mgrid[0:h_img, 0:w_img]
    print(n, 'frames')

    def cloud(i, dmax=1500):
        d = depth[i]
        sel = (d > 150) & (d < dmax)
        Z = d[sel]; X = (uu[sel]-K['cx'])*Z/K['fx']; Y = (vv[sel]-K['cy'])*Z/K['fy']
        return np.stack([X, Y, Z], 1)

    # 参考姿态帧的 RANSAC 平面 -> 重力方向在相机系中的表达
    ref_R = Rb[2]
    same = [i for i in range(n) if math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rb[i].T @ ref_R)-1)/2)))) < 0.02]
    print('同参考姿态帧:', same)
    n_up = []
    for i in same:
        P = cloud(i, 1200)
        sel = P[:, 1] > 0
        cnt, nn, dd = ransac_plane(P[sel] if sel.sum() > 5000 else P)
        if nn[1] > 0:
            nn, dd = -nn, -dd
        n_up.append(nn)
    n_up = np.mean(n_up, 0); n_up /= np.linalg.norm(n_up)
    print('平均重力方向(相机系):', np.round(n_up, 4))

    # 光度法求偏航
    dyn = np.zeros((h_img, w_img), bool)
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pb[i]-pb[j]) < 1.5 and np.linalg.norm(Rb[i]-Rb[j]) < 0.01:
                dyn |= np.abs(gray[i]-gray[j]) > 22
    dyn = cv2.dilate(dyn.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)

    def samples(i):
        d = depth[i]
        sel = (d > 150) & (d < 1000) & (~dyn)
        sel &= (np.arange(h_img)[:, None] % 4 == 0) & (np.arange(w_img)[None, :] % 4 == 0)
        vi, ui = np.nonzero(sel)
        Z = d[vi, ui]
        return np.stack([(ui-K['cx'])*Z/K['fx'], (vi-K['cy'])*Z/K['fy'], Z], 1).astype(np.float32), gray[i][vi, ui]

    S = [samples(i) for i in range(n)]

    def score(i, j, t_rel):
        P, a = S[i]
        Q = P + t_rel[None, :]; z = Q[:, 2]; ok = z > 60
        u = K['fx']*Q[:, 0]/z + K['cx']; v = K['fy']*Q[:, 1]/z + K['cy']
        ok &= (u > 2) & (u < w_img-3) & (v > 2) & (v < h_img-3)
        return -1.0 if ok.sum() < 200 else ncc(a[ok], bilinear(gray[j], u[ok], v[ok]))

    av = np.cross(n_up, [0.0, 0.0, 1.0])
    if np.linalg.norm(av) < 1e-6:
        av = np.cross(n_up, [1.0, 0.0, 0.0])
    av /= np.linalg.norm(av); bv = np.cross(n_up, av)

    def Rcb_of(psi):
        c1 = math.cos(psi)*av + math.sin(psi)*bv
        return np.stack([c1, np.cross(n_up, c1), n_up], 1)

    def same_rot(i, j):
        return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rb[i].T @ Rb[j])-1)/2)))) < 0.02

    pairs_t = [(i, j) for i in range(n) for j in range(i+1, n)
               if same_rot(i, j) and 1 < np.linalg.norm(pb[j]-pb[i]) < 200]
    print('纯平移对:', pairs_t)
    best = None
    for psi in np.arange(0, 2*math.pi, math.radians(2)):
        Rcb = Rcb_of(psi)
        for sc in (0.93, 0.96, 0.99, 1.02):
            v = sum(score(i, j, Rcb @ (sc*(pb[i]-pb[j]))) for i, j in pairs_t)
            if best is None or v > best[0]:
                best = (v, float(psi), sc)
    for psi in np.arange(best[1]-math.radians(4), best[1]+math.radians(4), math.radians(0.1)):
        Rcb = Rcb_of(psi)
        for sc in np.arange(0.90, 1.05, 0.002):
            v = sum(score(i, j, Rcb @ (sc*(pb[i]-pb[j]))) for i, j in pairs_t)
            if v > best[0]:
                best = (v, float(psi), float(sc))
    Rcb = Rcb_of(best[1]); s_photo = best[2]
    Rc = Rb[0].T @ Rcb.T
    print(f'偏航 {math.degrees(best[1])%360:.2f} deg  光度尺度 {s_photo:.4f}  平均NCC {best[0]/len(pairs_t):.4f}')
    print('R_tcp_cam:\n', np.round(Rc, 4))

    # 初始化 (t_c, h, s)：用参考姿态帧的 RANSAC 平面
    planes = [None]*n
    for i in same:
        P = cloud(i, 1200)
        cnt, nn, dd = ransac_plane(P)
        if nn[1] > 0:
            nn, dd = -nn, -dd
        planes[i] = (nn, dd, cnt)
    tc = np.array([-77.5, -10.0, -81.5])   # 光度法初值(相机在TCP上方, 隐含桌面高度≈0)
    h_table, sc = 0.0, 0.985
    print(f'初始: t_c={np.round(tc,2)} h={h_table:.1f} s={sc:.4f}')

    for it in range(4):
        pred = []
        for i in range(n):
            n_pred = Rc.T @ (Rb[i].T @ np.array([0.0, 0.0, 1.0]))
            u_i = Rb[i].T @ np.array([0.0, 0.0, 1.0])
            d_pred = (pb[i][2] + u_i @ tc - h_table) / sc
            P = cloud(i, 1500)[::2]
            dist = P @ n_pred + d_pred
            sel = np.abs(dist) < 15.0
            if sel.sum() < 3000:
                planes[i] = None
                continue
            nn2, dd2, rms = svd_plane(P[sel])
            planes[i] = (nn2, dd2, int(sel.sum()), rms)
        rows, rhs = [], []
        for i in range(n):
            if planes[i] is None:
                continue
            u_i = Rb[i].T @ np.array([0.0, 0.0, 1.0])
            rows.append(np.concatenate([u_i, [-1.0, -planes[i][1]]]))
            rhs.append(-pb[i][2])
        sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
        tc, h_table, sc = sol[:3], sol[3], sol[4]
        r = np.array(rows) @ sol - np.array(rhs)
        inc = [i for i in range(n) if planes[i] is not None]
        print(f'  迭代 {it}: 采用 {len(inc)}/{n} 帧  t_c={np.round(tc,2)} |t|={np.linalg.norm(tc):.1f} '
              f'h={h_table:.1f} s={sc:.4f}  残差rms={np.sqrt(np.mean(r**2)):.2f}mm')
        if it == 3:
            for i in range(n):
                if planes[i] is not None:
                    print(f'    {i:2d} {manifest[i]["name"][:16]:16s} 内点{planes[i][2]:7d} rms={planes[i][3]:.2f}'
                          f' n={np.round(planes[i][0],3)} d={planes[i][1]:8.2f}')

    n_base = np.mean([Rb[i] @ Rc @ planes[i][0] for i in range(n) if planes[i] is not None], axis=0)
    n_base /= np.linalg.norm(n_base)
    print('校验: 桌面法向在基座系', np.round(n_base, 4),
          ' 偏离竖直 %.2f deg' % math.degrees(math.acos(min(1.0, abs(n_base[2])))))
    out = {'R_tcp_cam': Rc.tolist(), 't_tcp_cam_mm': tc.tolist(), 'table_height_mm': float(h_table),
           'depth_scale': float(sc), 'table_normal_base': n_base.tolist(),
           'yaw_deg': float(math.degrees(best[1]) % 360), 'frames_used': int(sum(1 for p in planes if p)),
           'R_cam_base': Rcb.tolist(), 'depth_scale_photometric': float(s_photo),
           'mean_ncc_yaw': float(best[0]/len(pairs_t))}
    Path(sys.argv[2]).write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('written', sys.argv[2])


main()
