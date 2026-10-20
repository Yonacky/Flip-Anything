"""Read-only: can axis from silhouette tangency in the horizontal plane (no depth)."""
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
    dilate = int(sys.argv[10]) if len(sys.argv) > 10 else 2
    folder = ROOT.parent / 'logs' / 'experiments' / 'ds_control' / run
    k = json.loads((folder / "intrinsics.json").read_text(encoding="utf-8-sig"))["payload"]["metadata"]["factory_camera_param"]["color_intrinsic"]
    fx, fy, cx, cy = k["fx"], k["fy"], k["cx"], k["cy"]
    img = cv2.imread(str(folder / ("frame_%06d_color.png" % index)))
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (33, 55, 35), (95, 255, 255)) > 0
    green = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(green.astype(np.uint8), 8)
    i = max(range(1, n), key=lambda q: st[q, cv2.CC_STAT_AREA])
    gm = (lab == i).astype(np.uint8)
    if dilate > 0:
        gm = cv2.dilate(gm, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
    ys, xs = np.where(gm > 0)
    top, bot = int(ys.min()), int(ys.max())
    T = np.array(pose_matrix(pose))
    R_cam = T[:3, :3] @ R_TCP_CAM
    cam = T[:3, 3] + T[:3, :3] @ T_TCP_CAM

    def ray(u, v):
        d = R_cam @ np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        return d / np.linalg.norm(d)

    sols = []
    for v in range(top + 12, min(bot, h - 8), 8):
        row = np.where(gm[v] > 0)[0]
        if row.size < 60:
            continue
        u0, u1 = float(row.min()), float(row.max())
        h0 = ray(u0, v)[:2]; h1 = ray(u1, v)[:2]
        h0 /= np.linalg.norm(h0); h1 /= np.linalg.norm(h1)
        # signed distance from axis C to each tangent line = +-r
        # cross((C-cam), h) = +-r  ->  (Cx-cx)*(-hy) - (Cy-cy)*(-hx) = s
        A = np.array([[-h0[1], h0[0]], [-h1[1], h1[0]]])
        hm = h0 + h1
        hm /= np.linalg.norm(hm)
        cand = None
        for sgn in (1.0, -1.0):
            s = np.array([sgn * r_can, -sgn * r_can])
            try:
                C = np.linalg.solve(A, s + A @ cam[:2])
            except np.linalg.LinAlgError:
                continue
            vec = C - cam[:2]
            if float(vec @ hm) > 50.0:
                cand = C
                break
        if cand is None:
            continue
        sols.append((v, cand))
    if not sols:
        print(json.dumps({"frame": index, "error": "no rows"}))
        return
    arr = np.array([s[1] for s in sols])
    med = np.median(arr, axis=0)
    print(json.dumps({"frame": index, "rows": len(sols),
                      "axis_median": [round(float(med[0]), 1), round(float(med[1]), 1)],
                      "axis_std": [round(float(arr[:, 0].std()), 2), round(float(arr[:, 1].std()), 2)],
                      "axis_iqr": [round(float(np.percentile(arr[:, 0], 75) - np.percentile(arr[:, 0], 25)), 2),
                                   round(float(np.percentile(arr[:, 1], 75) - np.percentile(arr[:, 1], 25)), 2)],
                      "per_row": [[int(s[0]), round(float(s[1][0]), 1), round(float(s[1][1]), 1)] for s in sols[::4]]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
