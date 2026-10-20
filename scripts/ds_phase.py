"""相位相关与模板匹配：直接测量两帧之间的真实图像位移。"""
import json
from pathlib import Path
import cv2
import numpy as np

manifest = json.loads(Path('G:/Flip Anything/logs/experiments/legacy/ds_calib_manifest.json').read_text(encoding='utf-8'))
for (i, j) in [(1, 2), (0, 1), (2, 4), (3, 4)]:
    bi = Path(manifest[i]['run']) / manifest[i]['frame']
    bj = Path(manifest[j]['run']) / manifest[j]['frame']
    gi = cv2.cvtColor(cv2.imread(str(bi)+'_color.png'), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gj = cv2.cvtColor(cv2.imread(str(bj)+'_color.png'), cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow((1280, 480), cv2.CV_32F)
    a = gi[60:540, :] * win
    b = gj[60:540, :] * win
    (dx, dy), resp = cv2.phaseCorrelate(a, b)
    print(f'{i}->{j} 位姿差={np.round(np.array(manifest[j]["pose"][:3])-np.array(manifest[i]["pose"][:3]),1)} '
          f'相位相关位移 dx={dx:7.2f} dy={dy:7.2f} 置信={resp:.3f}')
    # 用罐子区域做模板匹配（罐子只在 0/1/2 帧可见）
    if i in (0, 1, 2) and j in (0, 1, 2):
        tmpl = gi[300:600, 560:760]
        if tmpl.size and gj.shape == gi.shape:
            res = cv2.matchTemplate(gj, tmpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            print(f'     罐子模板匹配: 位移 dx={loc[0]-560:6d} dy={loc[1]-300:6d} 相关={mx:.3f}')
