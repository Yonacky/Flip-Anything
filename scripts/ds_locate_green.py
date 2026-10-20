"""定位 run06 frame_000001 中的绿色物体在基座系的位置，并检查罐子原位置。"""
import json, math
from pathlib import Path
import cv2
import numpy as np
K = dict(fx=691.4324951171875, fy=691.7568359375, cx=643.6455078125, cy=358.8835144042969)
cal = json.loads(Path('G:/Flip Anything/logs/experiments/legacy/ds_handeye.json').read_text(encoding='utf-8'))
Rc = np.array(cal['R_tcp_cam']); tc = np.array(cal['t_tcp_cam_mm']); S = cal['depth_scale_plane']


def pose_matrix(pose):
    x, y, z, roll, pitch, yaw = pose
    r, p, q = map(math.radians, (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(q), math.sin(q)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, x],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, y],
                     [-sp, cp*sr, cp*cr, z], [0, 0, 0, 1]])


b = Path(r'G:/Flip Anything/logs/experiments/legacy/ds_flip_20261014_06\frame_000001')
pose = [618.45697, -13.350404, 66.861519, 179.932468, -3.002299, -0.00699]
img = cv2.imread(str(b) + '_color.png')
d = np.load(str(b) + '_depth_raw.npy').astype(np.float32) * S
h, w = d.shape
vv, uu = np.mgrid[0:h, 0:w]
T = pose_matrix(pose)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
g = cv2.inRange(hsv, (35, 55, 45), (95, 255, 255))
g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
n, lab, st, ct = cv2.connectedComponentsWithStats(g)
print('绿色连通域:')
for k in range(1, min(n, 6)):
    x, y, ww, hh, area = st[k]
    m = (lab == k) & (d > 0)
    if m.sum() < 200:
        print(f'  #{k} area={area} bbox=({x},{y},{ww},{hh}) 有效深度像素={int(m.sum())}')
        continue
    Z = d[m]; U = uu[m]; V = vv[m]
    P = np.stack([(U-K['cx'])*Z/K['fx'], (V-K['cy'])*Z/K['fy'], Z], 1)
    base = (T[:3, 3][None, :] + (T[:3, :3] @ (Rc @ P.T + tc[:, None])).T)
    print(f'  #{k} area={area} bbox=({x},{y},{ww},{hh})  基座系中心={np.round(base.mean(0),1)} '
          f'x范围[{base[:,0].min():.0f},{base[:,0].max():.0f}] y范围[{base[:,1].min():.0f},{base[:,1].max():.0f}] '
          f'z范围[{base[:,2].min():.0f},{base[:,2].max():.0f}] 深度中位={np.median(Z):.0f}')
# 罐子原位置附近有没有东西
P_all = None
sel = d > 0
Z = d[sel]; U = uu[sel]; V = vv[sel]
P = np.stack([(U-K['cx'])*Z/K['fx'], (V-K['cy'])*Z/K['fy'], Z], 1)
base = (T[:3, 3][None, :] + (T[:3, :3] @ (Rc @ P.T + tc[:, None])).T)
near = (np.abs(base[:, 0]-596.9) < 60) & (np.abs(base[:, 1]+12.8) < 60)
print('罐子原位置 (596.9,-12.8) ±60mm 范围内的观测点数:', int(near.sum()))
if near.sum() > 100:
    print('  该范围内 z 范围', np.round([base[near, 2].min(), base[near, 2].max()], 1), ' 中位', round(float(np.median(base[near, 2])), 1))
print('相机位置(基座系):', np.round(T[:3, 3] + T[:3, :3] @ tc, 1))
