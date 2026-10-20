"""罐体轴向定位（轮廓左右边缘中点）+ 端到端交叉验证。"""
import json, math
from pathlib import Path
import cv2
import numpy as np
K = dict(fx=691.4324951171875, fy=691.7568359375, cx=643.6455078125, cy=358.8835144042969)
cal = json.loads(Path('G:/Flip Anything/logs/experiments/legacy/ds_handeye.json').read_text(encoding='utf-8'))
Rc = np.array(cal['R_tcp_cam']); tc = np.array(cal['t_tcp_cam_mm'])
S = cal['depth_scale_plane']


def pose_matrix(pose):
    x, y, z, roll, pitch, yaw = pose
    r, p, q = map(math.radians, (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(q), math.sin(q)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, x],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, y],
                     [-sp, cp*sr, cp*cr, z], [0, 0, 0, 1]])


def to_base(P_cam, pose):
    T = pose_matrix(pose)
    return T[:3, 3] + T[:3, :3] @ (Rc @ P_cam + tc)


def to_cam(P_base, pose):
    T = pose_matrix(pose)
    return Rc.T @ (T[:3, :3].T @ (P_base - T[:3, 3]) - tc)


def project(P_cam):
    return np.array([K['fx']*P_cam[0]/P_cam[2] + K['cx'], K['fy']*P_cam[1]/P_cam[2] + K['cy']])


def can_axis(i, manifest, verbose=False):
    b = Path(manifest[i]['run']) / manifest[i]['frame']
    img = cv2.imread(str(b)+'_color.png')
    d = np.load(str(b)+'_depth_raw.npy').astype(np.float32) * S     # 深度尺度校正
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
    mids, widths = [], []
    for r in range(r0 + int(0.25*(r1-r0)), r1 - int(0.10*(r1-r0))):
        cols = np.nonzero(m[r])[0]
        if len(cols) < 8:
            continue
        uL, uR = cols.min(), cols.max()
        def depth_at(u):
            patch = d[r-2:r+3, max(0, u-2):u+3]
            ok = patch[(patch > 150) & (patch < 1500)]
            return float(np.median(ok)) if ok.size >= 6 else None
        zL, zR = depth_at(uL+2), depth_at(uR-2)
        if zL is None or zR is None:
            continue
        PL = np.array([(uL-K['cx'])*zL/K['fx'], (r-K['cy'])*zL/K['fy'], zL])
        PR = np.array([(uR-K['cx'])*zR/K['fx'], (r-K['cy'])*zR/K['fy'], zR])
        mids.append((PL+PR)/2); widths.append((uR-uL)*zL/K['fx'])
    if len(mids) < 15:
        return None
    mids = np.array(mids)
    world = np.array([to_base(p, manifest[i]['pose']) for p in mids])
    return {'axis_base': world.mean(0), 'spread': world.std(0), 'diameter_mm': float(np.median(widths)),
            'row_range': (int(r0), int(r1)), 'n_mid': len(mids)}


if __name__ == '__main__':
    manifest = json.loads(Path('G:/Flip Anything/logs/experiments/legacy/ds_calib_manifest.json').read_text(encoding='utf-8'))
    idx = [1, 2, 3, 4, 10, 11]
    res = {}
    for i in idx:
        r = can_axis(i, manifest)
        if r is None:
            print(f'frame {i}: 定位失败'); continue
        res[i] = r
        print(f'frame {i:2d} {manifest[i]["name"][:16]:16s} 轴线(基座) {np.round(r["axis_base"],1)} '
              f'帧内散布 {np.round(r["spread"],1)} 直径估计 {r["diameter_mm"]:.1f} mm 行 {r["row_range"]}')
    if len(res) > 1:
        arr = np.array([res[i]['axis_base'] for i in res])
        print('跨帧轴线一致性: 均值', np.round(arr.mean(0),1), ' std', np.round(arr.std(0),2))
        print('直径估计中位 %.1f mm' % np.median([res[i]['diameter_mm'] for i in res]))
    # 端到端：用帧 A 的轴线预测帧 B 的图像位置
    print('=== 端到端交叉验证（预测 vs 实测图像位置）===')
    for a in res:
        Pa = res[a]['axis_base']
        for b in res:
            if a == b:
                continue
            Pb = res[b]['axis_base']
            # 用 a 的轴点预测 b 帧图像位置：取 a 的轴线上与 b 帧等高的点
            Pc = to_cam(Pa, manifest[b]['pose'])
            if Pc[2] <= 50:
                continue
            uv = project(Pc)
            # 实测：b 帧轴线点的图像位置
            bb = Path(manifest[b]['run']) / manifest[b]['frame']
            d = np.load(str(bb)+'_depth_raw.npy').astype(np.float32) * S
            img = cv2.imread(str(bb)+'_color.png')
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            g = cv2.inRange(hsv, (35, 55, 45), (95, 255, 255))
            g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            n, lab, st, ct = cv2.connectedComponentsWithStats(g)
            k = 1 + int(np.argmax(st[1:, 4]))
            m = lab == k
            rows = np.nonzero(m.sum(1) > 10)[0]
            rmid = int((rows.min()+rows.max())/2)
            cols = np.nonzero(m[rmid])[0]
            obs_u = (cols.min()+cols.max())/2
            print(f'  {a}->{b}: 预测 u={uv[0]:7.1f} v={uv[1]:7.1f} | 实测中线 u={obs_u:7.1f} | du={uv[0]-obs_u:+6.1f}px')
