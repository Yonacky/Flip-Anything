"""Read-only: fit the target can's top rim ellipse and derive camera-to-can distance."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FX = 691.4324951171875
FY = 691.7568359375
CX = 643.6455078125
CY = 358.8835144042969


def main():
    run = sys.argv[1]
    index = int(sys.argv[2])
    folder = ROOT.parent / 'logs' / 'experiments' / 'ds_control' / run
    img = cv2.imread(str(folder / ("frame_%06d_color.png" % index)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
    h, w = gray.shape
    # 上半幅包含罐口的区域
    contours, _ = cv2.findContours(edges[0:int(h * 0.65)], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cands = []
    for c in contours:
        if len(c) < 90:
            continue
        (ex, ey), (MA, ma), ang = cv2.fitEllipse(c)
        if ma <= 0:
            continue
        ratio = MA / ma
        area = cv2.contourArea(c)
        if not (0.45 < ratio < 1.0):
            continue
        if not (120 < MA < 1000):
            continue
        cands.append((area, (ex, ey), (MA, ma), ang, len(c)))
    cands.sort(key=lambda t: -t[0])
    print("frame", index, "ellipse candidates (biggest first):")
    for area, (ex, ey), (MA, ma), ang, n in cands[:6]:
        # 用相机内参由椭圆短轴推距离：直径为 D 的圆，短轴像素 = f*D/z
        for D in (66.0, 58.0, 53.0):
            z_short = FY * D / ma
            z_long = FX * D / ma
            print("   c=(%6.1f,%6.1f) axes=(%5.1f,%5.1f) ang=%5.1f area=%7.0f | D=%.0f -> z~%.0f mm"
                  % (ex, ey, MA, ma, ang, area, D, z_short))
        # 罐口中心相对光心的角度
        print("      offset from optical axis: x=%+.0f px (%.1f deg), y=%+.0f px (%.1f deg)"
              % (ex - CX, np.degrees(np.arctan2(ex - CX, FX)), ey - CY, np.degrees(np.arctan2(ey - CY, FY))))
    cv2.imwrite(str(folder / ("edges_%06d.png" % index)), edges)
    depth = np.load(folder / ("frame_%06d_depth_raw.npy" % index)).astype(np.float32)
    # 光心附近与罐体区域的有效深度
    for (px, py) in ((int(CX), int(CY)), (620, 400), (660, 500), (600, 300)):
        z = depth[py, px]
        print("   depth at (%d,%d) = %s" % (px, py, "invalid" if z <= 0 else "%.0f mm" % z))


if __name__ == "__main__":
    main()
