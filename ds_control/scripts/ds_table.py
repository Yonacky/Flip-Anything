"""Read-only: find and fit the horizontal support surface visible under the target.

No motion commands.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def intr(run):
    p = (ROOT.parent / 'logs' / 'experiments' / 'ds_control' / run / "intrinsics.json")
    if not p.exists():
        txt = (ROOT.parent / 'logs' / 'experiments' / 'ds_control' / run / "_intrinsics.txt").read_text(encoding="utf-8-sig")
        p.write_text(txt, encoding="utf8")
    d = json.loads(p.read_text(encoding="utf-8-sig"))
    return d["payload"]["metadata"]["factory_camera_param"]["color_intrinsic"]


def main():
    run = sys.argv[1]
    index = int(sys.argv[2])
    folder = ROOT.parent / 'logs' / 'experiments' / 'ds_control' / run
    k = intr(run)
    fx, fy, cx, cy = k["fx"], k["fy"], k["cx"], k["cy"]
    depth = np.load(folder / ("frame_%06d_depth_raw.npy" % index)).astype(np.float32)
    img = cv2.imread(str(folder / ("frame_%06d_color.png" % index)))
    h, w = depth.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 60, 50), (90, 255, 255)) > 0
    green = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8)) > 0
    # 目标罐的水平范围
    cols = np.where(green.any(axis=0))[0]
    gx0, gx1 = (int(cols[0]), int(cols[-1])) if cols.size else (0, w - 1)

    step = 3
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    zz = depth[::step, ::step]
    gm = green[::step, ::step]
    # 只用“非目标”的远处下半幅点拟合桌面；排除罐体本身与其两侧
    sel = (zz > 120) & (zz < 2500) & (ys > 400) & (~gm)
    pts = np.stack([(xs[sel] - cx) * zz[sel] / fx, (ys[sel] - cy) * zz[sel] / fy, zz[sel]], axis=1)
    print("frame", index, "candidate non-target points in lower half:", pts.shape[0])
    rng = np.random.default_rng(0)
    if pts.shape[0] > 300:
        best = None
        for _ in range(300):
            pick = pts[rng.choice(len(pts), 3, replace=False)]
            n = np.cross(pick[1] - pick[0], pick[2] - pick[0])
            nn = np.linalg.norm(n)
            if nn < 1e-6:
                continue
            n = n / nn
            d = -n @ pick[0]
            cnt = int((np.abs(pts @ n + d) < 8.0).sum())
            if best is None or cnt > best[0]:
                best = (cnt, n, d)
        cnt, n, d = best
        inl = np.abs(pts @ n + d) < 8.0
        m = pts[inl].mean(axis=0)
        v = np.linalg.svd(pts[inl] - m)[2][-1]
        d2 = -v @ m
        if v[2] < 0:
            v, d2 = -v, -d2
        res = np.abs(pts[inl] @ v + d2)
        print("plane inliers %d normal %s d=%.1f res_mean=%.2f" % (inl.sum(), np.round(v, 4), d2, res.mean()))
        print("-> distance from camera to that plane along its normal: %.0f mm" % abs(d2))
        print("-> plane normal angle to camera optical axis: %.1f deg"
              % np.degrees(np.arccos(min(1.0, abs(v[2])))))
        # 该平面内可见点的深度范围
        dd = pts[inl][:, 2]
        print("   inlier depth range %.0f..%.0f mm" % (dd.min(), dd.max()))
    # 罐体两侧的背景深度（左右各一列）
    for x in (max(0, gx0 - 60), min(w - 1, gx1 + 60)):
        col = depth[:, x]
        v2 = col[col > 0]
        if v2.size:
            print("column %d depth med %.0f valid %.2f" % (x, np.median(v2), v2.size / h))
    cv2.imwrite(str(folder / ("mask_%06d.png" % index)),
                np.hstack([cv2.cvtColor(green.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR), img]))


if __name__ == "__main__":
    main()
