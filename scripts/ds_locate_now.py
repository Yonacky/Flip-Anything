"""用新模型定位罐子（基座系），并给出侧抓所需的预抓取位姿。"""
import json, math, sys
from pathlib import Path
import cv2
import numpy as np
K = dict(fx=691.4324951171875, fy=691.7568359375, cx=643.6455078125, cy=358.8835144042969)
cal = json.loads(Path('G:/Flip Anything/logs/experiments/legacy/ds_tc.json').read_text(encoding='utf-8'))
Rc = np.array(cal['R_tcp_cam']); tc = np.array(cal['t_tcp_cam_mm']); S = cal['depth_scale']
frame = sys.argv[1]; run = Path(sys.argv[2]); pose = json.loads(sys.argv[3])


def pm(p):
    x, y, z, r, pp, q = p
    r, pp, q = map(math.radians, (r, pp, q))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(pp), math.sin(pp), math.cos(q), math.sin(q)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, x],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, y],
                     [-sp, cp*sr, cp*cr, z], [0, 0, 0, 1]])


b = run / frame
img = cv2.imread(str(b) + '_color.png')
d = np.load(str(b) + '_depth_raw.npy').astype(np.float32) * S
h, w = d.shape
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
g = cv2.inRange(hsv, (35, 55, 45), (95, 255, 255))
g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
n, lab, st, ct = cv2.connectedComponentsWithStats(g)
k = 1 + int(np.argmax(st[1:, 4]))
m = lab == k
rows = np.nonzero(m.sum(1) > 10)[0]
print('绿色区域 bbox', st[k, :4], '行范围', rows.min(), rows.max())
T = pm(pose)
mids, ws, zs = [], [], []
for r in range(rows.min() + int(0.2*(rows.max()-rows.min())), rows.max() - int(0.08*(rows.max()-rows.min()))):
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
    mids.append((PL+PR)/2); ws.append((uR-uL)*zL/K['fx']); zs.append((zL+zR)/2)
if len(mids) < 10:
    print('罐体定位失败: 中点太少', len(mids)); sys.exit(1)
W = np.array([T[:3, 3] + T[:3, :3] @ (Rc @ p + tc) for p in np.array(mids)])
axis = W.mean(0)
print(f'罐轴(基座系) = {np.round(axis,1)}  帧内散布 {np.round(W.std(0),1)}  直径≈{np.median(ws):.1f} mm  深度中位 {np.median(zs):.0f} mm')
print(f'罐轴竖直性检查: x范围 {W[:,0].min():.0f}..{W[:,0].max():.0f}  y范围 {W[:,1].min():.0f}..{W[:,1].max():.0f}  z范围 {W[:,2].min():.0f}..{W[:,2].max():.0f}')
# 相机位置
print('相机位置(基座系) =', np.round(T[:3, 3] + T[:3, :3] @ tc, 1))
# 预抓取位姿：工具Z水平指向罐轴，抓取高度 h_g
R = T[:3, :3]
ax_dir = R[:, 2]
for h_g in (40.0, 50.0, 60.0):
    target_grasp = np.array([axis[0], axis[1], h_g])
    for stand in (150.0, 180.0):
        tcp = target_grasp - stand*ax_dir
        print(f'  抓取高度 {h_g:.0f}mm 站距 {stand:.0f}mm -> TCP {np.round(tcp,1)}')
