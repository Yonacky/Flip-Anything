"""用光度法 t_c 检查罐轴在基座系的位置是否合理（若 h≈0，罐轴可见段应在 z≈+30..+100）。"""
import json, math
from pathlib import Path
import cv2
import numpy as np
K = dict(fx=691.4324951171875, fy=691.7568359375, cx=643.6455078125, cy=358.8835144042969)
cal = json.loads(Path('G:/Flip Anything/logs/experiments/legacy/ds_tc.json').read_text(encoding='utf-8'))
Rc = np.array(cal['R_tcp_cam']); tc = np.array(cal['t_tcp_cam_mm']); S = cal['depth_scale']
man = json.loads(Path('G:/Flip Anything/logs/experiments/legacy/ds_calib_manifest.json').read_text(encoding='utf-8'))


def pm(pose):
    x, y, z, r, p, q = pose
    r, p, q = map(math.radians, (r, p, q))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(q), math.sin(q)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, x],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, y],
                     [-sp, cp*sr, cp*cr, z], [0, 0, 0, 1]])


def can_axis(i):
    b = Path(man[i]['run']) / man[i]['frame']
    img = cv2.imread(str(b) + '_color.png')
    d = np.load(str(b) + '_depth_raw.npy').astype(np.float32) * S
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, (35, 55, 45), (95, 255, 255))
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, ct = cv2.connectedComponentsWithStats(g)
    k = 1 + int(np.argmax(st[1:, 4]))
    m = lab == k
    rows = np.nonzero(m.sum(1) > 10)[0]
    if len(rows) < 20:
        return None
    r0, r1 = rows.min(), rows.max()
    mids, ws = [], []
    for r in range(r0 + int(0.25*(r1-r0)), r1 - int(0.10*(r1-r0))):
        cols = np.nonzero(m[r])[0]
        if len(cols) < 8:
            continue
        uL, uR = cols.min(), cols.max()
        def dep(u):
            pt = d[r-2:r+3, max(0, u-2):u+3]
            ok = pt[(pt > 100) & (pt < 2000)]
            return float(np.median(ok)) if ok.size >= 6 else None
        zL, zR = dep(uL+2), dep(uR-2)
        if zL is None or zR is None:
            continue
        PL = np.array([(uL-K['cx'])*zL/K['fx'], (r-K['cy'])*zL/K['fy'], zL])
        PR = np.array([(uR-K['cx'])*zR/K['fx'], (r-K['cy'])*zR/K['fy'], zR])
        mids.append((PL+PR)/2); ws.append((uR-uL)*zL/K['fx'])
    if len(mids) < 15:
        return None
    T = pm(man[i]['pose'])
    W = np.array([T[:3, 3] + T[:3, :3] @ (Rc @ p + tc) for p in np.array(mids)])
    return W.mean(0), W.std(0), float(np.median(ws))


res = {}
for i in range(len(man)):
    r = can_axis(i)
    if r is None:
        continue
    res[i] = r
    print(f'frame {i:2d} {man[i]["name"][:16]:16s} 罐轴(基座)={np.round(r[0],1)} 帧内散布={np.round(r[1],1)} 直径≈{r[2]:.1f}')
if len(res) > 2:
    A = np.array([v[0] for v in res.values()])
    print('跨帧一致性: 均值', np.round(A.mean(0),1), ' std', np.round(A.std(0),2))
