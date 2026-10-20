"""Read-only analysis of one saved observation: depth colouring plus near-structure stats.

Sends no motion commands.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = sys.argv[1]
    index = int(sys.argv[2])
    folder = ROOT.parent / 'logs' / 'experiments' / 'ds_control' / run
    color = cv2.imread(str(folder / ("frame_%06d_color.png" % index)))
    depth = np.load(folder / ("frame_%06d_depth_raw.npy" % index)).astype(np.float32)
    h, w = depth.shape
    usable = depth > 100
    print("shape=%s usable=%.1f%%" % (depth.shape, usable.mean() * 100))
    v = depth[usable]
    print("range mm p1=%.0f p5=%.0f p25=%.0f med=%.0f p75=%.0f p95=%.0f"
          % tuple(np.percentile(v, q) for q in (1, 5, 25, 50, 75, 95)))
    print("median depth grid rows top->bottom:")
    for r in range(6):
        row = []
        for c in range(8):
            block = depth[r * h // 6:(r + 1) * h // 6, c * w // 8:(c + 1) * w // 8]
            b = block[block > 100]
            row.append("%6.0f" % np.median(b) if b.size > 50 else "    --")
        print("  " + " ".join(row))
    near, far = 150.0, 3000.0
    norm = np.clip((np.where(usable, depth, far) - near) / (far - near), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat[~usable] = (0, 0, 0)
    out = folder / ("frame_%06d_depthvis.png" % index)
    cv2.imwrite(str(out), np.hstack([color, heat]))
    print("wrote %s" % out)
    for limit in (500, 700, 1000):
        mask = (usable & (depth < limit)).astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        keep = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 800:
                continue
            sel = labels == i
            dv = depth[sel]
            keep.append((stats[i, cv2.CC_STAT_AREA], cents[i], stats[i], np.median(dv),
                         np.percentile(dv, 10)))
        keep.sort(key=lambda t: -t[0])
        print("components closer than %dmm: %d" % (limit, len(keep)))
        for area, cen, st, dm, dp in keep[:6]:
            print("  area=%6d cen=(%6.1f,%6.1f) bbox=(%d,%d,%dx%d) dmed=%.0f dp10=%.0f"
                  % (area, cen[0], cen[1], st[0], st[1], st[2], st[3], dm, dp))


if __name__ == "__main__":
    main()
