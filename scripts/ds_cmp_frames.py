"""比较同一物理位姿但不同会话的两帧：图像与深度是否一致。"""
from pathlib import Path
import cv2
import numpy as np

pairs = [('G:/Flip Anything/logs/experiments/legacy/ds_flip_20261014_04/frame_000001', 'G:/Flip Anything/logs/experiments/legacy/ds_flip_20261014_05/frame_000001', '同位姿 430,18,176.7 (04 vs 05)'),
         ('G:/Flip Anything/logs/experiments/legacy/ds_flip_20261014_02/frame_000002', 'G:/Flip Anything/logs/experiments/legacy/ds_flip_20261014_04/frame_000000', '同物理位姿 380,18,176.7 (02 vs 04)')]
for a, b, label in pairs:
    ia = cv2.cvtColor(cv2.imread(a + '_color.png'), cv2.COLOR_BGR2GRAY).astype(np.float32)
    ib = cv2.cvtColor(cv2.imread(b + '_color.png'), cv2.COLOR_BGR2GRAY).astype(np.float32)
    da = np.load(a + '_depth_raw.npy').astype(np.float32)
    db = np.load(b + '_depth_raw.npy').astype(np.float32)
    print(label)
    print(f'  灰度 mean {ia.mean():.1f} / {ib.mean():.1f}   std {ia.std():.1f} / {ib.std():.1f}')
    va = da[(da > 100) & (da < 1500)]; vb = db[(db > 100) & (db < 1500)]
    print(f'  深度 有效率 {float(((da>0)&(da<3000)).mean()):.3f}/{float(((db>0)&(db<3000)).mean()):.3f} '
          f' 中位 {np.median(va):.1f}/{np.median(vb):.1f}  p10 {np.percentile(va,10):.1f}/{np.percentile(vb,10):.1f}'
          f'  p90 {np.percentile(va,90):.1f}/{np.percentile(vb,90):.1f}')
    m = (da > 100) & (da < 1500) & (db > 100) & (db < 1500)
    diff = db[m] - da[m]
    print(f'  同像素深度差: 中位 {np.median(diff):+.2f} mm  rms {np.sqrt(np.mean(diff**2)):.2f} mm  比例 {np.median(db[m]/da[m]):.4f}')
    v = (ia > 10) & (ib > 10)
    x = ia[v] - ia[v].mean(); y = ib[v] - ib[v].mean()
    print(f'  同像素灰度 NCC {float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))):.4f}')
    d = np.clip(np.abs(ib - ia)[60:600, :], 0, 80).astype(np.uint8)
    cv2.imwrite('G:/Flip Anything/logs/experiments/legacy/_cmp_' + Path(a).parent.name + '.png', cv2.applyColorMap(d * 3, cv2.COLORMAP_JET))
