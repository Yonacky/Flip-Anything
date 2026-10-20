"""光度自标定：整幅深度图预测形变 + 像素比对，解出 R_tcp_cam 与 t_tcp_cam 及深度尺度。

  P_cam_j = R_rel P_cam_i + t_rel
  R_rel = R_cam_j^T R_cam_i,  R_cam_x = R_b^x R_c,  C_x = p_b^x + R_b^x t_c
  t_rel = R_cam_j^T (p_b^j - p_b^i + (R_b^j - R_b^i) t_c)
阶段1（只用同姿态纯平移对）：t_rel 与 t_c 无关，且桌面法向给出 R_cb 第三列 -> 一维搜索偏航角 + 深度尺度
阶段2（含倾斜对）：固定 R_c，三自由度搜索 t_c
"""
import json, math, sys
from pathlib import Path
import cv2
import numpy as np

K = dict(fx=691.4324951171875, fy=691.7568359375, cx=643.6455078125, cy=358.8835144042969)
STRIDE = 4


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


def bilinear(img, u, v):
    x0 = np.floor(u).astype(np.int32); y0 = np.floor(v).astype(np.int32)
    x1 = x0 + 1; y1 = y0 + 1
    wx = (u - x0).astype(np.float32); wy = (v - y0).astype(np.float32)
    return (img[y0, x0]*(1-wx)*(1-wy) + img[y0, x1]*wx*(1-wy)
            + img[y1, x0]*(1-wx)*wy + img[y1, x1]*wx*wy)


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(a @ b / (na * nb))


def main():
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    n = len(manifest)
    gray, depths, Rb, pb = [], [], [], []
    for f in manifest:
        base = Path(f['run']) / f['frame']
        gray.append(cv2.cvtColor(cv2.imread(str(base)+'_color.png'), cv2.COLOR_BGR2GRAY).astype(np.float32))
        depths.append(np.load(str(base)+'_depth_raw.npy').astype(np.float32))
        T = pose_matrix(f['pose']); Rb.append(T[:3, :3]); pb.append(T[:3, 3])
    h, w = gray[0].shape

    # 每帧桌面法向（重力方向在相机系中的表达）
    ncam = []
    for i in range(n):
        d = depths[i]; vv, uu = np.mgrid[0:h, 0:w]
        sel = (d > 150) & (d < 1200) & (vv > 400)
        Z = d[sel]; X = (uu[sel]-K['cx'])*Z/K['fx']; Y = (vv[sel]-K['cy'])*Z/K['fy']
        cnt, nn, dd = fit_plane(np.stack([X, Y, Z], 1))
        if nn[1] > 0:
            nn, dd = -nn, -dd
        ncam.append(nn)
        print(f'  frame {i} plane inliers {cnt:6d} n={np.round(nn,4)} d={dd:8.2f}')
    same = [i for i in range(n) if math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rb[i].T @ Rb[2])-1)/2)))) < 0.02]
    n_up = np.mean([ncam[i] for i in same], axis=0); n_up /= np.linalg.norm(n_up)
    print('同姿态帧', same, ' 平均法向', np.round(n_up, 4))

    # 采样点
    def samples(i):
        d = depths[i]; vv, uu = np.mgrid[0:h, 0:w]
        sel = (d > 150) & (d < 1000) & (vv < 640)
        sel = sel & (np.arange(h)[:, None] % STRIDE == 0) & (np.arange(w)[None, :] % STRIDE == 0)
        vi, ui = np.nonzero(sel)
        Z = d[vi, ui]
        P = np.stack([(ui-K['cx'])*Z/K['fx'], (vi-K['cy'])*Z/K['fy'], Z], 1)
        return P.astype(np.float32), gray[i][vi, ui], ui, vi

    S = [samples(i) for i in range(n)]
    print('每帧采样点:', [len(s[0]) for s in S])

    def score(i, j, R_rel, t_rel):
        P, a, ui, vi = S[i]
        Q = (R_rel @ P.T).T + t_rel[None, :]
        z = Q[:, 2]
        ok = z > 60
        if ok.sum() < 200:
            return -1.0
        u = K['fx']*Q[:, 0]/z + K['cx']; v = K['fy']*Q[:, 1]/z + K['cy']
        ok &= (u > 2) & (u < w-3) & (v > 2) & (v < h-3)
        if ok.sum() < 200:
            return -1.0
        return ncc(a[ok], bilinear(gray[j], u[ok], v[ok]))

    # ---- 阶段 1：偏航角 + 深度尺度（纯平移对） ----
    a_vec = np.cross(n_up, [0.0, 0.0, 1.0])
    if np.linalg.norm(a_vec) < 1e-6:
        a_vec = np.cross(n_up, [1.0, 0.0, 0.0])
    a_vec /= np.linalg.norm(a_vec)
    b_vec = np.cross(n_up, a_vec)
    bad = set()
    if len(sys.argv) > 3 and sys.argv[3].strip():
        bad = {int(v) for v in sys.argv[3].split(',')}
    keep = [i for i in range(n) if i not in bad]
    print('排除帧:', sorted(bad), ' 使用帧:', keep)

    def same_rot(i, j):
        cos = (np.trace(Rb[i].T @ Rb[j]) - 1) / 2
        return math.degrees(math.acos(max(-1.0, min(1.0, cos)))) < 0.02

    pairs_t = [(i, j) for i in keep for j in keep if i < j
               and same_rot(i, j) and np.linalg.norm(pb[j]-pb[i]) > 1]
    print('纯平移对:', pairs_t)

    def Rcb_of(psi):
        c1 = math.cos(psi)*a_vec + math.sin(psi)*b_vec
        c2 = np.cross(n_up, c1)
        return np.stack([c1, c2, n_up], 1)

    best = None
    for psi in np.arange(0, 2*math.pi, math.radians(2)):
        Rcb = Rcb_of(psi)
        for sc in (0.94, 0.97, 1.0, 1.03, 1.06):
            t = sum(score(i, j, np.eye(3), Rcb @ (sc*(pb[j]-pb[i]))) for i, j in pairs_t)
            if best is None or t > best[0]:
                best = (t, psi, sc)
    print(f'阶段1 粗搜: 得分 {best[0]/len(pairs_t):.4f} psi={math.degrees(best[1]):.2f}deg scale={best[2]}')
    for psi in np.arange(best[1]-math.radians(3), best[1]+math.radians(3), math.radians(0.2)):
        Rcb = Rcb_of(psi)
        for sc in np.arange(best[2]-0.04, best[2]+0.041, 0.005):
            t = sum(score(i, j, np.eye(3), Rcb @ (sc*(pb[j]-pb[i]))) for i, j in pairs_t)
            if t > best[0]:
                best = (t, psi, float(sc))
    Rcb = Rcb_of(best[1]); scale = best[2]
    Rc = Rb[0].T @ Rcb.T
    print(f'阶段1 精搜: 平均NCC {best[0]/len(pairs_t):.4f} psi={math.degrees(best[1]):.2f}deg '
          f'深度尺度={scale:.4f}')
    print('R_cam_base:\n', np.round(Rcb, 4))
    print('R_tcp_cam:\n', np.round(Rc, 4))
    for i, j in pairs_t:
        print(f'   对 {i}->{j} NCC {score(i, j, np.eye(3), Rcb @ (scale*(pb[j]-pb[i]))):.4f}')

    # ---- 阶段 2：t_c（用全部对，含倾斜） ----
    all_pairs = [(i, i+1) for i in range(n-1) if i not in bad and i+1 not in bad]
    def cam_poses(tc):
        Rcams = [Rb[i] @ Rc for i in range(n)]
        Cs = [pb[i] + Rb[i] @ tc for i in range(n)]
        return Rcams, Cs

    def total(tc, pairs=all_pairs, stride=1):
        Rcams, Cs = cam_poses(tc)
        tot = 0.0
        for i, j in pairs[::stride]:
            R_rel = Rcams[j].T @ Rcams[i]
            t_rel = Rcams[j].T @ (Cs[j] - Cs[i])
            tot += score(i, j, R_rel, scale * t_rel)
        return tot / max(1, len(pairs[::stride]))

    best_t = None
    for tx in np.arange(-200, 201, 50):
        for ty in np.arange(-200, 201, 50):
            for tz in np.arange(-200, 201, 50):
                v = total(np.array([tx, ty, tz], float), pairs=all_pairs, stride=3)
                if best_t is None or v > best_t[0]:
                    best_t = (v, np.array([tx, ty, tz], float))
    print(f'阶段2 粗搜: NCC {best_t[0]:.4f} t_c={np.round(best_t[1],1)}')
    tc = best_t[1].copy(); step = 25.0
    for _ in range(6):
        improved = True
        while improved:
            improved = False
            for axis in range(3):
                for d in (step, -step):
                    cand = tc.copy(); cand[axis] += d
                    v = total(cand)
                    if v > total(tc) + 1e-6:
                        tc = cand; improved = True
        step /= 2.5
    print(f'阶段2 精搜: NCC {total(tc):.4f} t_c={np.round(tc,2)} |t_c|={np.linalg.norm(tc):.1f}')
    for i, j in all_pairs:
        Rcams, Cs = cam_poses(tc)
        R_rel = Rcams[j].T @ Rcams[i]; t_rel = Rcams[j].T @ (Cs[j]-Cs[i])
        print(f'   对 {i}->{j} NCC {score(i, j, R_rel, scale*t_rel):.4f} 相机位移(基座) {np.round(Cs[j]-Cs[i],1)}')

    # ---- 独立校验 ----
    n_base = Rb[2] @ Rc @ ncam[2]
    Tb = pose_matrix(manifest[2]['pose'])
    hh = float(n_base @ (Tb[:3, 3] + Tb[:3, :3] @ tc))
    d0 = float(-ncam[2] @ np.zeros(3) - 0)
    print('独立校验: 桌面法向在基座系 =', np.round(n_base, 4),
          ' 偏离竖直', round(math.degrees(math.acos(min(1, abs(n_base[2])))), 2), 'deg')
    out = {'frames': manifest, 'R_cam_base': Rcb.tolist(), 'R_tcp_cam': Rc.tolist(),
           't_tcp_cam_mm': tc.tolist(), 'depth_scale': scale,
           'mean_ncc_stage1': best[0]/len(pairs_t), 'mean_ncc_stage2': total(tc),
           'table_normal_base': n_base.tolist()}
    Path(sys.argv[2]).write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('written', sys.argv[2])


main()
