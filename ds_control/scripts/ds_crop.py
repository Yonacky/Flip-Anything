"""Read-only crop utility: upscale a region of a saved frame. No motion commands."""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
run = sys.argv[1]
index = int(sys.argv[2])
x0, y0, x1, y1 = map(int, sys.argv[3:7])
scale = float(sys.argv[7])
img = cv2.imread(str(ROOT.parent / 'logs' / 'experiments' / 'ds_control' / run / ("frame_%06d_color.png" % index)))
crop = img[y0:y1, x0:x1]
crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
out = ROOT.parent / 'logs' / 'experiments' / 'ds_control' / run / ("crop_%06d_%d_%d.png" % (index, x0, y0))
cv2.imwrite(str(out), crop)
print(out, crop.shape)
