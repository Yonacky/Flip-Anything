"""用光度法解 t_c：滚转位姿带来巨大视差，可牢牢定住相机光心相对 TCP 的位置。"""
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


def ransac_plane(points, thresh=2.0, iters=1200, seed=3):
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


def bilinear(img, u, v):
    x0 = np.floor(u).astype(np.int32); y0 = np.floor(v).astype(np.int32)
    wx = (u - x0).astype(np.float32); wy = (v - y0).astype(np.float32)
    return (img[y0, x0]*(1-wx)*(1-wy) + img[y0, x0+1]*wx*(1-wy)
            + img[y0+1, x0]*(1-wx)*wy + img[y0+1, x0+1]*wx*wy)


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-6 or nb < 1e-6 else float(a @ b / (na * nb))


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

# 重力方向（相机系）用参考姿态帧的平面拟合
ref = Rb[2]
same = [i for i in range(n) if math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rb[i].T @ ref)-1)/2)))) < 0.02]
nups = []
for i in same:
    d = depth[i]
    sel = (d > 150) & (d < 1200) & (vv > 400)
    Z = d[sel]; X = (uu[sel]-K['cx'])*Z/K['fx']; Y = (vv[sel]-K['cy'])*Z/K['fy']
    cnt, nn, dd = ransac_plane(np.stack([X, Y, Z], 1))
    if nn[1] > 0:
        nn = -nn
    nups.append(nn)
n_up = np.mean(nups, 0); n_up /= np.linalg.norm(n_up)
print('重力方向(相机系):', np.round(n_up, 4), ' 用帧', same)

dyn = np.zeros((h_img, w_img), bool)
for i in range(n):
    for j in range(i+1, n):
        if np.linalg.norm(pb[i]-pb[j]) < 1.5 and np.linalg.norm(Rb[i]-Rb[j]) < 0.01:
            dyn |= np.abs(gray[i]-gray[j]) > 22
dyn = cv2.dilate(dyn.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
print('动态像素比例 %.3f' % float(dyn.mean()))


def samples(i, stride=4):
    d = depth[i]
    sel = (d > 150) & (d < 1200) & (~dyn)
    sel &= (np.arange(h_img)[:, None] % stride == 0) & (np.arange(w_img)[None, :] % stride == 0)
    vi, ui = np.nonzero(sel)
    Z = d[vi, ui]
    return np.stack([(ui-K['cx'])*Z/K['fx'], (vi-K['cy'])*Z/K['fy'], Z], 1).astype(np.float32)


mask0 = (depth[0] > 150) & (depth[0] < 1200) & (~dyn) & \
        (np.arange(h_img)[:, None] % 4 == 0) & (np.arange(w_img)[None, :] % 4 == 0)
vi0, ui0 = np.nonzero(mask0)
Z0 = depth[0][vi0, ui0]
S = np.stack([(ui0-K['cx'])*Z0/K['fx'], (vi0-K['cy'])*Z0/K['fy'], Z0], 1).astype(np.float32)
print('固定采样点', len(S))

# 偏航搜索（固定重力方向）
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
pairs_roll = [(i, j) for i in range(n) for j in range(i+1, n) if 20 < np.linalg.norm(Rb[i]-Rb[j])*180/math.pi]
print('平移对', pairs_t)
print('大角度旋转对数量', len(pairs_roll))


def score_pair(i, j, R_rel, t_rel):
    P = S
    Q = (R_rel @ P.T).T + t_rel[None, :]
    z = Q[:, 2]; ok = z > 60
    u = K['fx']*Q[:, 0]/z + K['cx']; v = K['fy']*Q[:, 1]/z + K['cy']
    ok &= (u > 2) & (u < w_img-3) & (v > 2) & (v < h_img-3)
    if ok.sum() < 300:
        return -1.0
    return ncc(gray[i][(np.arange(h_img)[:, None] % 4 == 0) & (np.arange(w_img)[None, :] % 4 == 0)][ok], bilinear(gray[j], u[ok], v[ok]))


binmask = (np.arange(h_img)[:, None] % 4 == 0) & (np.arange(w_img)[None, :] % 4 == 0)


def score_pair2(i, j, R_rel, t_rel):
    Q = (R_rel @ S.T).T + t_rel[None, :]
    z = Q[:, 2]; ok = z > 60
    u = K['fx']*Q[:, 0]/z + K['cx']; v = K['fy']*Q[:, 1]/z + K['cy']
    ok &= (u > 2) & (u < w_img-3) & (v > 2) & (v < h_img-3)
    if ok.sum() < 300:
        return -1.0
    return ncc(gray[i][vi0, ui0][ok], bilinear(gray[j], u[ok], v[ok]))


# 1) 偏航
best = None
for psi in np.arange(0, 2*math.pi, math.radians(2)):
    Rcb = Rcb_of(psi)
    for sc in (0.95, 0.98, 1.0, 1.02):
        v = sum(score_pair2(i, j, np.eye(3), Rcb @ (sc*(pb[i]-pb[j]))) for i, j in pairs_t)
        if best is None or v > best[0]:
            best = (v, float(psi), sc)
for psi in np.arange(best[1]-math.radians(3), best[1]+math.radians(3), math.radians(0.1)):
    Rcb = Rcb_of(psi)
    for sc in np.arange(0.90, 1.05, 0.002):
        v = sum(score_pair2(i, j, np.eye(3), Rcb @ (sc*(pb[i]-pb[j]))) for i, j in pairs_t)
        if v > best[0]:
            best = (v, float(psi), float(sc))
Rcb = Rcb_of(best[1]); s_ph = best[2]
Rc = Rb[0].T @ Rcb.T
print(f'偏航 {math.degrees(best[1])%360:.2f} deg  尺度 {s_ph:.4f}  平均NCC {best[0]/len(pairs_t):.4f}')
print('R_tcp_cam:\n', np.round(Rc, 4))

# 2) t_c 三维搜索（用旋转对 + 平移对）
use_pairs = pairs_t + pairs_roll


def total_ncc(tc, pairs):
    Rcams = [Rb[i] @ Rc for i in range(n)]
    Cs = [pb[i] + Rb[i] @ tc for i in range(n)]
    vals = []
    for i, j in pairs:
        R_rel = Rcams[j].T @ Rcams[i]
        t_rel = Rcams[j].T @ (Cs[i] - Cs[j])
        vals.append(score_pair2(i, j, R_rel, s_ph*t_rel))
    return float(np.mean(vals)), vals


print('--- t_c 粗搜 ---')
bestt = None
for tx in np.arange(-250, 251, 100):
    for ty in np.arange(-150, 151, 100):
        for tz in np.arange(-250, 251, 100):
            v, _ = total_ncc(np.array([tx, ty, tz], float), pairs_roll[::6])
            if bestt is None or v > bestt[0]:
                bestt = (v, np.array([tx, ty, tz], float))
print('粗搜最佳', np.round(bestt[1], 1), 'NCC', round(bestt[0], 4))
tc = bestt[1].copy()
for step in (40.0, 20.0, 10.0, 4.0, 1.5):
    improved = True
    while improved:
        improved = False
        base_val, _ = total_ncc(tc, use_pairs)
        for axis in range(3):
            for dd in (step, -step):
                cand = tc.copy(); cand[axis] += dd
                v, _ = total_ncc(cand, use_pairs)
                if v > base_val + 1e-5:
                    tc = cand; improved = True; break
            if improved:
                break
v_all, vals = total_ncc(tc, use_pairs)
print('--- 最终 ---')
print('t_c =', np.round(tc, 2), ' |t_c| =', round(float(np.linalg.norm(tc)), 1), ' 平均NCC =', round(v_all, 4))
for (i, j), v in zip(use_pairs, vals):
    print(f'   {i:2d}->{j:2d} NCC {v:6.3f}')
out = {'R_cam_base': Rcb.tolist(), 'R_tcp_cam': Rc.tolist(), 't_tcp_cam_mm': tc.tolist(),
       'yaw_deg': float(math.degrees(best[1]) % 360), 'depth_scale': float(s_ph),
       'mean_ncc': v_all, 'n_pairs': len(use_pairs)}
Path(sys.argv[2]).write_text(json.dumps(out, indent=2), encoding='utf-8')
print('written', sys.argv[2])
