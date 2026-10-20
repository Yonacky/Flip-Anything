"""开合夹爪时哪些像素在运动 -> 定位手指并求开合方向。"""
from pathlib import Path
import cv2
import numpy as np
run = Path('G:/Flip Anything/logs/experiments/legacy/ds_flip_20261014_05')
a = cv2.imread(str(run/'frame_000004_color.png'))   # 张开
b = cv2.imread(str(run/'frame_000003_color.png'))   # 闭合
c = cv2.imread(str(run/'frame_000005_color.png'))   # 中间
ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
gc = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
d = np.abs(ga - gb)
print('张开 vs 闭合: 差分 mean %.2f max %.0f   >25 的像素 %d' % (d.mean(), d.max(), int((d>25).sum())))
# 运动区域的包围盒与重心
m = d > 25
m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3,3),np.uint8)).astype(bool)
if m.sum() > 50:
    ys, xs = np.nonzero(m)
    print('运动像素 bbox u[%d..%d] v[%d..%d] 重心 (%.1f, %.1f) 数量 %d' % (xs.min(), xs.max(), ys.min(), ys.max(), xs.mean(), ys.mean(), m.sum()))
    # 分块统计
    for v0 in range(0, 720, 120):
        print('   v %3d-%3d: %d 个运动像素' % (v0, v0+119, int(m[v0:v0+120].sum())))
# 用模板匹配估计运动位移（取运动最集中的小块）
best = None
for v0 in range(400, 720, 40):
    for u0 in range(200, 1100, 60):
        t = ga[v0:v0+40, u0:u0+40]
        if t.std() < 6:
            continue
        res = cv2.matchTemplate(gb, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if best is None or mx > best[0]:
            best = (mx, u0, v0, loc[0]-u0, loc[1]-v0)
print('最强局部位移: 相关=%.3f 位置 u=%d v=%d  位移 (du=%d, dv=%d)' % best)
cv2.imwrite('G:/Flip Anything/logs/experiments/legacy/_grip_motion.png', cv2.applyColorMap(np.clip(d*4,0,255).astype(np.uint8), cv2.COLORMAP_JET))
print('saved G:/Flip Anything/logs/experiments/legacy/_grip_motion.png')
