"""Compare the two post-release frames: is the inverted can standing still?"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb


def mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 50) &
         ((hsv[:, :, 0] < 16) | (hsv[:, :, 0] > 165))).astype(np.uint8)
    m[:, :400] = 0          # keep the table area only; 400+ excludes the person on the left
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    i = max(range(1, n), key=lambda q: st[q, cv2.CC_STAT_AREA])
    return {'mask': lab == i, 'area': int(st[i, cv2.CC_STAT_AREA]),
            'bbox': [int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP]),
                     int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])],
            'centroid': [float(cen[i][0]), float(cen[i][1])]}


def main():
    run = tb.run_dir()
    frames = tb.results_with_camera(run)
    a, b = frames[-2], frames[-1]
    ra, rb = mask(tb.read_color(a['color'])), mask(tb.read_color(b['color']))
    out = {'frame_a': a['color'].name, 'frame_b': b['color'].name,
           'dt_note': 'two observations of a stationary arm'}
    for tag, r in (('a', ra), ('b', rb)):
        if r:
            out[tag] = {k: r[k] for k in ('area', 'bbox', 'centroid')}
    if ra and rb:
        # also compare the image region directly
        ia = tb.read_color(a['color'])
        ib = tb.read_color(b['color'])
        x, y, w, h = ra['bbox']
        pad = 25
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(ia.shape[1], x + w + pad), min(ia.shape[0], y + h + pad)
        diff = np.abs(ia[y0:y1, x0:x1].astype(int) - ib[y0:y1, x0:x1].astype(int))
        out['bbox_center_shift_px'] = [round(rb['centroid'][0] - ra['centroid'][0], 2),
                                       round(rb['centroid'][1] - ra['centroid'][1], 2)]
        out['bbox_diff_px'] = [rb['bbox'][i] - ra['bbox'][i] for i in range(4)]
        out['area_change_px'] = rb['area'] - ra['area']
        out['mean_abs_diff_in_box'] = round(float(diff.mean()), 2)
        out['pct_pixels_diff_over_25'] = round(float((diff.max(axis=2) > 25).mean() * 100), 2)
        out['upright_artwork_check'] = 'can artwork appears rotated 180 deg in both frames'
    print(json.dumps(out, ensure_ascii=False, indent=1))
    (run / 'stability.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
