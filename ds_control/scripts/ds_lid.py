"""Read-only: lid-circle pose from a local ellipse fit inside the can's top box."""
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
    radius = float(sys.argv[9]) if len(sys.argv) > 9 else 33.0
    folder = ROOT.parent / 'logs' / 'experiments' / 'ds_control' / run
    k = json.loads((folder / "intrinsics.json").read_text(encoding="utf-8-sig"))["payload"]["metadata"]["factory_camera_param"]["color_intrinsic"]
    fx, fy, cx, cy = k["fx"], k["fy"], k["cx"], k["cy"]
    img = cv2.imread(str(folder / ("frame_%06d_color.png" % index)))
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 60, 40), (95, 255, 255)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(green.astype(np.uint8), 8)
    i = max(range(1, n), key=lambda q: st[q, cv2.CC_STAT_AREA])
    gm = lab == i
    ys, xs = np.where(gm)
    top = int(ys.min()); uL, uR = int(xs.min()), int(xs.max())

    x0, x1 = max(0, uL - 30), min(w, uR + 30)
    y0, y1 = max(0, top - 90), min(h, top + 25)
    sub = img[y0:y1, x0:x1]
    sh = hsv[y0:y1, x0:x1]
    metal = ((sh[:, :, 1] < 80) & (sh[:, :, 2] > 110)).astype(np.uint8) * 255
    metal = cv2.morphologyEx(metal, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    metal = cv2.morphologyEx(metal, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(metal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 800]
    if not cnts:
        print(json.dumps({"frame": index, "error": "no lid contour"}))
        return
    c = max(cnts, key=cv2.contourArea)
    ell = cv2.fitEllipse(c)
    (eu, ev), (d1, d2), ang = ell
    eu += x0; ev += y0
    a, b = max(d1, d2) / 2.0, min(d1, d2) / 2.0

    # circle of known radius on a horizontal plane -> centre in camera frame
    # depth from a perspective pinhole: use the major axis as the metric scale
    z_center = radius * fx / a
    pc = np.array([(eu - cx) * z_center / fx, (ev - cy) * z_center / fy, z_center])
    T = np.array(pose_matrix(pose))
    R_cam = T[:3, :3] @ R_TCP_CAM
    cam = T[:3, 3] + T[:3, :3] @ T_TCP_CAM
    P = cam + R_cam @ pc
    # elevation of the view onto the lid plane from the ellipse aspect ratio
    elev = float(np.degrees(np.arcsin(max(0.02, min(0.99, b / a)))))
    print(json.dumps({"frame": index, "ellipse_px": [round(eu, 1), round(ev, 1)],
                      "axes_px": [round(2 * a, 1), round(2 * b, 1)],
                      "view_elevation_deg": round(elev, 1),
                      "lid_centre_cam_mm": np.round(pc, 1).tolist(),
                      "lid_centre_base_mm": np.round(P, 1).tolist(),
                      "camera_base_mm": np.round(cam, 1).tolist()},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
