"""把 1->2 平移对的特征位移可视化：绿=位移量在 40..60mm，红=异常。"""
import json, math
from pathlib import Path
import cv2
import numpy as np
K = dict(fx=691.4324951171875, fy=691.7568359375, cx=643.6455078125, cy=358.8835144042969)
manifest = json.loads(Path('G:/Flip Anything/logs/experiments/legacy/ds_calib_manifest.json').read_text(encoding='utf-8'))
i, j = 1, 2
base_i = Path(manifest[i]['run']) / manifest[i]['frame']
base_j = Path(manifest[j]['run']) / manifest[j]['frame']
gi = cv2.cvtColor(cv2.imread(str(base_i)+'_color.png'), cv2.COLOR_BGR2GRAY)
gj = cv2.cvtColor(cv2.imread(str(base_j)+'_color.png'), cv2.COLOR_BGR2GRAY)
di = np.load(str(base_i)+'_depth_raw.npy').astype(np.float32)
dj = np.load(str(base_j)+'_depth_raw.npy').astype(np.float32)


def depth3d(d, u, v):
    h, w = d.shape
    ui, vi = int(round(u)), int(round(v))
    if not (6 <= ui < w-6 and 6 <= vi < h-6):
        return None
    patch = d[vi-3:vi+4, ui-3:ui+4]
    ok = patch[(patch > 150) & (patch < 1500)]
    if ok.size < 25 or ok.std() > 2.5:
        return None
    z = float(np.median(ok))
    return np.array([(u-K['cx'])*z/K['fx'], (v-K['cy'])*z/K['fy'], z])


m = np.zeros(di.shape, np.uint8); m[(di > 200) & (di < 900)] = 255; m[650:, :] = 0
p0 = cv2.goodFeaturesToTrack(gi, maxCorners=500, qualityLevel=0.005, minDistance=7, blockSize=7, mask=m).reshape(-1, 2)
lk = dict(winSize=(41, 41), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.005))
nxt, st, _ = cv2.calcOpticalFlowPyrLK(gi, gj, p0.reshape(-1,1,2), None, **lk)
back, st2, _ = cv2.calcOpticalFlowPyrLK(gj, gi, nxt, None, **lk)
fb = np.linalg.norm(back.reshape(-1,2)-p0, axis=1)
good = (st.ravel()==1)&(st2.ravel()==1)&(fb<0.5)
vis = cv2.cvtColor(gi, cv2.COLOR_GRAY2BGR)
rows = []
for idx in np.nonzero(good)[0]:
    Pi = depth3d(di, *p0[idx]); Pj = depth3d(dj, *nxt[idx,0])
    if Pi is None or Pj is None:
        continue
    dp = Pj - Pi; mag = float(np.linalg.norm(dp))
    rows.append((mag, p0[idx][0], p0[idx][1], float(Pi[2]), float(Pj[2]), nxt[idx,0,0]-p0[idx][0], nxt[idx,0,1]-p0[idx][1]))
    color = (0, 255, 0) if 40 <= mag <= 60 else (0, 0, 255)
    cv2.circle(vis, (int(p0[idx][0]), int(p0[idx][1])), 4, color, 1)
mags = np.array([r[0] for r in rows])
print('匹配数', len(rows), ' |dp| 中位', round(float(np.median(mags)),2), ' 在40-60区间的比例', round(float(((mags>=40)&(mags<=60)).mean()),3))
pix = np.array([[r[5], r[6]] for r in rows]); zs = np.array([[r[3], r[4]] for r in rows])
print('图像位移 中位', np.round(np.median(pix, axis=0),1), ' p90', np.round(np.percentile(np.abs(pix),90,axis=0),1))
ok = (mags>=40)&(mags<=60)
if ok.sum() > 5:
    print('位移正确点的像素位置分布: u', np.round(np.percentile([rows[k][1] for k in np.nonzero(ok)[0]],[10,50,90]),0),
          ' v', np.round(np.percentile([rows[k][2] for k in np.nonzero(ok)[0]],[10,50,90]),0),
          ' 深度', np.round(np.percentile([rows[k][3] for k in np.nonzero(ok)[0]],[10,50,90]),0))
print('位移偏小点的像素位置分布: u', np.round(np.percentile([rows[k][1] for k in np.nonzero(~ok)[0]],[10,50,90]),0),
      ' v', np.round(np.percentile([rows[k][2] for k in np.nonzero(~ok)[0]],[10,50,90]),0),
      ' 深度', np.round(np.percentile([rows[k][3] for k in np.nonzero(~ok)[0]],[10,50,90]),0))
print('图像位移 vs 位移量 相关: ', np.round(np.corrcoef(mags, np.linalg.norm(pix,axis=1))[0,1],3))
cv2.imwrite('G:/Flip Anything/logs/experiments/legacy/_pair_1_2_overlay.png', vis)
cv2.imwrite('G:/Flip Anything/logs/experiments/legacy/_pair_1_2_frame_j.png', cv2.cvtColor(gj, cv2.COLOR_GRAY2BGR))
print('saved G:/Flip Anything/logs/experiments/legacy/_pair_1_2_overlay.png')
