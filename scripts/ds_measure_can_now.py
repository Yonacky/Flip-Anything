"""测量当前帧中罐子在相机系里的真实位置，判定手指/抓取点相对罐子的关系。"""
import json
from pathlib import Path
import cv2
import numpy as np
K = dict(fx=691.4324951171875, fy=691.7568359375, cx=643.6455078125, cy=358.8835144042969)
cal = json.loads(Path('G:/Flip Anything/logs/experiments/legacy/ds_tc.json').read_text(encoding='utf-8'))
Rc = np.array(cal['R_tcp_cam']); tc = np.array(cal['t_tcp_cam_mm']); S = cal['depth_scale']
run = Path(r'G:/Flip Anything/logs/experiments/legacy/ds_flip_20261014_07')
b = run/'frame_000003'
img = cv2.imread(str(b)+'_color.png')
d = np.load(str(b)+'_depth_raw.npy').astype(np.float32)*S
h, w = d.shape
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
g = cv2.inRange(hsv, (35, 55, 45), (95, 255, 255))
g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
n, lab, st, ct = cv2.connectedComponentsWithStats(g)
k = 1 + int(np.argmax(st[1:, 4]))
m = (lab == k) & (d > 0)
print('罐体像素', int(m.sum()), ' bbox', st[k, :4])
Z = d[m]; vv, uu = np.mgrid[0:h, 0:w]
U = uu[m]; V = vv[m]
print('深度: 中位 %.0f mm  p5 %.0f  p95 %.0f' % (np.median(Z), np.percentile(Z, 5), np.percentile(Z, 95)))
P = np.stack([(U-K['cx'])*Z/K['fx'], (V-K['cy'])*Z/K['fy'], Z], 1)
Pc = P.mean(0)
print('罐体在相机系质心 %s  到相机距离 %.0f mm' % (np.round(Pc, 1), np.linalg.norm(Pc)))
print('  相机系下: 横向 %.1f  竖直 %.1f  前后 %.1f' % (Pc[0], Pc[1], Pc[2]))
# 两个假设下，罐轴应在相机系中的哪里
pose = [537.8, -34.0, 35.0, 180, 85, 180]


def pm(p):
    x, y, z, r, pp, q = p
    r, pp, q = map(np.radians, (r, pp, q))
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(pp), np.sin(pp), np.cos(q), np.sin(q)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, x],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, y],
                     [-sp, cp*sr, cp*cr, z], [0, 0, 0, 1]])


T = pm(pose)
Re = np.array(cal['R_tcp_cam'])
def cam_of_tool(P_tool):
    return Re.T @ (np.array(P_tool) - tc)
for label, P_tool in (('假设A: 罐轴在TCP处', [0, 0, 0]),
                      ('假设B: 罐轴在TCP前174mm', [0, 0, 174]),
                      ('手指尖(约在TCP)', [0, 0, 0])):
    Pc_h = cam_of_tool(P_tool)
    print(f'{label}: 相机系 {np.round(Pc_h,1)}  距离 {np.linalg.norm(Pc_h):.0f} mm')
# 手指标称位置估计（两指中点在 TCP）
print()
print('结论: 若实测罐心距离≈112mm 则罐轴与 TCP 会合(假设A)；若更远则罐子在更前面')
