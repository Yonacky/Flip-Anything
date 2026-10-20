"""诊断：纯平移下特征位移是否恒定；桌面平面偏移是否给出重力方向。"""
import json, math, sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, 'scripts')
K = dict(fx=691.4324951171875, fy=691.7568359375, cx=643.6455078125, cy=358.8835144042969)
manifest = json.loads(Path('G:/Flip Anything/logs/experiments/legacy/ds_calib_manifest.json').read_text(encoding='utf-8'))


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


gray, depths = [], []
for f in manifest:
    base = Path(f['run']) / f['frame']
    gray.append(cv2.cvtColor(cv2.imread(str(base)+'_color.png'), cv2.COLOR_BGR2GRAY))
    depths.append(np.load(str(base)+'_depth_raw.npy').astype(np.float32))

print('=== 桌面平面：每帧拟合 ===')
planes = []
for i, f in enumerate(manifest):
    d = depths[i]; h, w = d.shape
    vv, uu = np.mgrid[0:h, 0:w]
    sel = (d > 150) & (d < 1200) & (vv > 400)
    Z = d[sel]; X = (uu[sel]-K['cx'])*Z/K['fx']; Y = (vv[sel]-K['cy'])*Z/K['fy']
    cnt, n, dd = fit_plane(np.stack([X, Y, Z], 1))
    if n[1] > 0:
        n, dd = -n, -dd
    planes.append((n, dd))
    print(f'  {i} {f["name"]:16s} inliers {cnt:6d}  n={np.round(n,4)}  d={dd:8.2f}  (|n.z|={abs(n[2]):.3f})')

print('=== 用平面偏移解重力方向 n_base ===')
Rb = [pose_matrix(f['pose'])[:3, :3] for f in manifest]
pb = [np.array(f['pose'][:3]) for f in manifest]
rows, rhs = [], []
for i in range(len(manifest)):
    for j in range(i+1, len(manifest)):
        if not np.allclose(Rb[i], Rb[j], atol=1e-6):
            continue
        dd = pb[j] - pb[i]
        if np.linalg.norm(dd) < 1:
            continue
        rows.append(dd); rhs.append(-(planes[j][1] - planes[i][1]))
A, b = np.array(rows), np.array(rhs)
n_base, *_ = np.linalg.lstsq(A, b, rcond=None)
print('  n_base 估计:', np.round(n_base, 4), ' |n|=', round(float(np.linalg.norm(n_base)), 4),
      ' 与[0,0,1]夹角', round(math.degrees(math.acos(min(1, abs(n_base[2]/max(1e-9,np.linalg.norm(n_base))))))), 'deg')
print('  预测 vs 实测平面偏移变化:')
for r, y in zip(rows, rhs):
    print(f'    d={np.round(r,1)} predicted {float(r@n_base):8.2f}  measured {y:8.2f}')

print('=== 特征位移恒定性（纯平移应给出同一常向量）===')
lk = dict(winSize=(41, 41), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.005))


def depth3d(i, u, v):
    d = depths[i]; h, w = d.shape
    ui, vi = int(round(u)), int(round(v))
    if not (6 <= ui < w-6 and 6 <= vi < h-6):
        return None
    patch = d[vi-3:vi+4, ui-3:ui+4]
    ok = patch[(patch > 150) & (patch < 1500)]
    if ok.size < 25 or ok.std() > 2.5:
        return None
    z = float(np.median(ok))
    return np.array([(u-K['cx'])*z/K['fx'], (v-K['cy'])*z/K['fy'], z])


def seed(i):
    d = depths[i]
    m = np.zeros(d.shape, np.uint8); m[(d > 200) & (d < 900)] = 255; m[650:, :] = 0
    p = cv2.goodFeaturesToTrack(gray[i], maxCorners=500, qualityLevel=0.005, minDistance=7, blockSize=7, mask=m)
    return [] if p is None else p.reshape(-1, 2)


for (i, j) in [(0,1),(1,2),(3,4),(2,4),(0,4)]:
    p0 = seed(i)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(gray[i], gray[j], p0.reshape(-1,1,2), None, **lk)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray[j], gray[i], nxt, None, **lk)
    fb = np.linalg.norm(back.reshape(-1,2) - p0, axis=1)
    good = (st.ravel()==1)&(st2.ravel()==1)&(fb<0.5)
    dps = []
    for idx in np.nonzero(good)[0]:
        Pi = depth3d(i, *p0[idx]); Pj = depth3d(j, *nxt[idx,0])
        if Pi is not None and Pj is not None:
            dps.append(Pj - Pi)
    dps = np.array(dps)
    mags = np.linalg.norm(dps, axis=1)
    print(f'  {i}->{j} 匹配 {len(dps)}  d={np.round(pb[j]-pb[i],1)}  期望|dp|={np.linalg.norm(pb[j]-pb[i]):.1f}')
    print(f'      mean dp={np.round(dps.mean(0),2)}  各轴std={np.round(dps.std(0),2)}   |dp| 中位={np.median(mags):.2f} p10={np.percentile(mags,10):.2f} p90={np.percentile(mags,90):.2f}')
    print(f'      |dp| 与深度相关性: ', np.round(np.corrcoef(mags, dps[:,2]+np.array([depth3d(i,*p) for p in p0[np.nonzero(good)[0]]])[:,2]/2)[0,1], 3))
