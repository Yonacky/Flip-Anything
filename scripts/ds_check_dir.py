"""检验 R_cb 的作用方向：罐心相机系位移是否等于 -R_cb*δ。"""
import json, math
from pathlib import Path
import cv2
import numpy as np
K = dict(fx=691.4324951171875, fy=691.7568359375, cx=643.6455078125, cy=358.8835144042969)
manifest = json.loads(Path('G:/Flip Anything/logs/experiments/legacy/ds_calib_manifest.json').read_text(encoding='utf-8'))
cal = json.loads(Path('G:/Flip Anything/logs/experiments/legacy/ds_handeye.json').read_text(encoding='utf-8'))
Rcb = np.array(cal['R_cam_base'])


def pose_matrix(pose):
    x, y, z, roll, pitch, yaw = pose
    r, p, q = map(math.radians, (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(q), math.sin(q)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, x],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, y],
                     [-sp, cp*sr, cp*cr, z], [0, 0, 0, 1]])


def can_centroid(i):
    b = Path(manifest[i]['run']) / manifest[i]['frame']
    img = cv2.imread(str(b)+'_color.png')
    d = np.load(str(b)+'_depth_raw.npy').astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, (35, 60, 50), (95, 255, 255))
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, ct = cv2.connectedComponentsWithStats(g)
    k = 1 + int(np.argmax(st[1:, 4]))
    if st[k, 4] < 4000:
        return None, None, None
    h, w = d.shape; vv, uu = np.mgrid[0:h, 0:w]
    m = (lab == k) & (d > 0)
    # 只取圆柱侧面中部，避开轮廓边缘
    rows = np.nonzero(m.any(1))[0]
    r0, r1 = rows.min()+int(0.3*(rows.max()-rows.min())), rows.min()+int(0.7*(rows.max()-rows.min()))
    band = m.copy(); band[:r0] = False; band[r1:] = False
    Z = d[band]; U = uu[band]; V = vv[band]
    P = np.stack([(U-K['cx'])*Z/K['fx'], (V-K['cy'])*Z/K['fy'], Z], 1)
    return P.mean(0), st[k, 0], st[k, :4]


print('R_cb =\n', np.round(Rcb, 4))
print('R_cb^T =\n', np.round(Rcb.T, 4))
prev = None
for i in [1, 2, 3, 4, 10, 11]:
    Pc, x, box = can_centroid(i)
    if Pc is None:
        print(i, 'no can'); continue
    pb = pose_matrix(manifest[i]['pose'])[:3, 3]
    print(f'frame {i:2d} {manifest[i]["name"][:16]:16s} 罐心(相机系) {np.round(Pc,1)}  bbox={box}')
    if prev is not None:
        d = pb - prev[0]
        if np.linalg.norm(d) > 1:
            dp = Pc - prev[1]
            print(f'    相机平移(基座) {np.round(d,1)}  实测 dp_cam {np.round(dp,2)}')
            print(f'    -R_cb*d = {np.round(-Rcb@d,2)}   -R_cb^T*d = {np.round(-Rcb.T@d,2)}')
    prev = (pb, Pc)
