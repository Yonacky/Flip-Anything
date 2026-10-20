"""Read-only: can axis from silhouette chords (no sparse depth, no can height)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flip_anything.geometry import pose_matrix  # noqa: E402

T_TCP_CAM = np.array([-77.5, -10.0, -81.5])
R_TCP_CAM = np.array([[0.0492, 0.9599, 0.276],
                      [-0.9987, 0.0442, 0.0244],
                      [0.0113, -0.2768, 0.9609]])


def main():
    run, index = sys.argv[1], int(sys.argv[2])
    pose = [float(v) for v in sys.argv[3:9]]
    r_can = float(sys.argv[9]) if len(sys.argv) > 9 else 33.0
    folder = ROOT.parent / 'logs' / 'experiments' / 'ds_control' / run
    k = json.loads((folder / "intrinsics.json").read_text(encoding="utf-8-sig"))["payload"]["metadata"]["factory_camera_param"]["color_intrinsic"]
    fx, fy, cx, cy = k["fx"], k["fy"], k["cx"], k["cy"]
    img = cv2.imread(str(folder / ("frame_%06d_color.png" % index)))
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 60, 40), (95, 255, 255)) > 0
    green = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(green.astype(np.uint8), 8)
    i = max(range(1, n), key=lambda q: st[q, cv2.CC_STAT_AREA])
    gm = lab == i
    ys, xs = np.where(gm)
    top, bot = int(ys.min()), int(ys.max())
    T = np.array(pose_matrix(pose))
    R_cam = T[:3, :3] @ R_TCP_CAM
    cam = T[:3, 3] + T[:3, :3] @ T_TCP_CAM
    R_wc = R_cam.T

    est = []
    for v in range(top + 8, min(bot, h - 6), 6):
        row = np.where(gm[v])[0]
        if row.size < 40:
            continue
        u0, u1 = float(np.percentile(row, 2)), float(np.percentile(row, 98))
        chord_px = u1 - u0
        if chord_px < 60:
            continue
        z = 2 * r_can * fx / chord_px
        P = []
        for u in (u0, u1):
            pc = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])
            P.append(cam + R_cam @ pc)
        mid = (P[0] + P[1]) / 2
        horiz = np.array([mid[0] - cam[0], mid[1] - cam[1]])
        d = np.linalg.norm(horiz)
        sin_a = min(0.99, r_can / max(d, 1.0))
        axis = mid + np.array([horiz[0] / d, horiz[1] / d, 0]) * (r_can * np.sqrt(1 - sin_a ** 2))
        est.append((v, chord_px, z, axis))
    arr = np.array([e[3] for e in est])
    med = np.median(arr, axis=0)
    print(json.dumps({"frame": index,
                      "rows_used": len(est), "v_range": [top, bot],
                      "axis_median": [round(float(med[0]), 1), round(float(med[1]), 1)],
                      "axis_spread_mm": [round(float(arr[:, 0].std()), 2), round(float(arr[:, 1].std()), 2)],
                      "axis_vs_row": [[int(e[0]), round(float(e[3][0]), 1), round(float(e[3][1]), 1), round(float(e[2]), 1)] for e in est[::3]]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
