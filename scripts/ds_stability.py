"""两次静止观测对比：罐子是否仍在同一位置（倒立稳定）。"""
import json
from pathlib import Path
import cv2
import numpy as np
run = Path(r'G:/Flip Anything/logs/experiments/legacy/ds_flip_20261014_07')
res = []
for f in ('frame_000067', 'frame_000068'):
    p = run/f
    if not (p.with_name(f + '_color.png')).exists():
        print('缺', f); continue
    img = cv2.imread(str(p) + '_color.png')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, (35, 55, 45), (95, 255, 255))
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, ct = cv2.connectedComponentsWithStats(g)
    k = 1 + int(np.argmax(st[1:, 4]))
    res.append((f, st[k, :4], ct[k]))
    print(f, 'bbox', st[k, :4], 'centroid', np.round(ct[k], 1), 'area', st[k, 4])
if len(res) == 2:
    d = np.linalg.norm(res[0][2] - res[1][2])
    print('两次观测罐体质心像素差 = %.2f px' % d)
    print('判定:', '保持倒立且位置未变 ✓' if d < 5 else '位置有变化 ✗')
