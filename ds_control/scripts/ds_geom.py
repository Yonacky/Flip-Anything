"""Read-only: back-project one observation and report geometry. No motion commands."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def intrinsics(run):
    txt = (ROOT.parent / 'logs' / 'experiments' / 'ds_control' / run / "_intrinsics.txt").read_text(encoding="utf-8-sig")
    return json.loads(txt)["payload"]["metadata"]["factory_camera_param"]["color_intrinsic"]


def main():
    run = sys.argv[1]
    index = int(sys.argv[2])
    folder = ROOT.parent / 'logs' / 'experiments' / 'ds_control' / run
    intr = intrinsics(run)
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    depth = np.load(folder / ("frame_%06d_depth_raw.npy" % index)).astype(np.float32)
    h, w = depth.shape

    def back(px, py, z):
        return np.array([(px - cx) * z / fx, (py - cy) * z / fy, z])

    step = 4
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    zz = depth[::step, ::step]
    sel = (zz > 150) & (zz < 1200) & (ys > 380)
    pts = np.stack([(xs[sel] - cx) * zz[sel] / fx,
                    (ys[sel] - cy) * zz[sel] / fy, zz[sel]], axis=1)
    print("plane candidate points:", pts.shape)
    rng = np.random.default_rng(0)
    best = None
    sub = pts[rng.choice(len(pts), size=min(6000, len(pts)), replace=False)]
    for it in range(200):
        pick = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(pick[1] - pick[0], pick[2] - pick[0])
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n = n / norm
        d = -n @ pick[0]
        dist = np.abs(sub @ n + d)
        cnt = int((dist < 6.0).sum())
        if best is None or cnt > best[0]:
            best = (cnt, n, d)
    cnt, n, d = best
    inl = np.abs(pts @ n + d) < 6.0
    m = pts[inl].mean(axis=0)
    v = np.linalg.svd(pts[inl] - m)[2][-1]
    d2 = -v @ m
    res = np.abs(pts[inl] @ v + d2)
    if v[2] < 0:
        v, d2 = -v, -d2
    print("plane: inliers=%d/%d normal=%s d=%.1f res_mean=%.2f res_p95=%.2f"
          % (inl.sum(), len(pts), np.round(v, 4), d2, res.mean(), np.percentile(res, 95)))
    print("plane distance along optical axis (mm): %.1f" % d2)

    print("block scan: (px,py) depth_mm euclid_mm cam_xyz")
    for r in range(4):
        for c in range(4):
            py = int((r + 0.5) * h / 4)
            px = int((c + 0.5) * w / 4)
            z = float(depth[py, px])
            if z <= 0:
                print("  (%4d,%4d) invalid" % (px, py))
                continue
            p = back(px, py, z)
            print("  (%4d,%4d) %7.0f %7.0f %s" % (px, py, z, np.linalg.norm(p), np.round(p, 1)))

    print("centre column scan")
    for py in range(20, h, 40):
        z = float(depth[py, int(cx)])
        if z > 0:
            p = back(cx, py, z)
            print("  row=%3d depth=%7.0f euclid=%7.0f cam=%s" % (py, z, np.linalg.norm(p), np.round(p, 1)))
    np.save(folder / ("plane_%06d.npy" % index), np.array([*v, d2]))


if __name__ == "__main__":
    main()
