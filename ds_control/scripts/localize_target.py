"""从当前 RGB-D 定位目标容器：候选绿色区域 -> 3D 尺寸检验（圆柱/罐）-> 输出基座系轴向与端点。
相机位姿假定：光轴与工具 Z 平行，光心在 TCP 沿工具 +Z 偏移 |t|（默认 110mm，可传参）。"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flip_anything.geometry import pose_matrix  # noqa: E402

RUN = Path(sys.argv[1]).resolve()
FRAME = int(sys.argv[2])
TCAM = float(sys.argv[3]) if len(sys.argv) > 3 else 110.0


def events():
    return [json.loads(l) for l in (RUN / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def intr():
    for e in events():
        if e["kind"] == "camera_started":
            return e["payload"]["metadata"]["factory_camera_param"]["color_intrinsic"]
    raise RuntimeError("no camera_started")


def pose_of(idx):
    last = None
    for e in events():
        if e["kind"] == "robot_sample":
            last = e["payload"]
        elif e["kind"] == "observation" and e["payload"].get("camera"):
            if int(Path(e["payload"]["camera"]["color_file"]).stem.split("_")[1]) == idx:
                return list(last["pose"])
    raise RuntimeError("pose not found")


I = intr()
pose = pose_of(FRAME)
R = np.array(pose_matrix(pose))[:3, :3]
C = np.array(pose[:3]) + R @ np.array([0.0, 0.0, TCAM])       # 光心（基座系）
Rcam = R                                                        # 光轴 = 工具 Z
bgr = cv2.imread(str(RUN / f"frame_{FRAME:06d}_color.png"), cv2.IMREAD_COLOR)
depth = np.load(RUN / f"frame_{FRAME:06d}_depth_raw.npy").astype(np.float64)
h, w = depth.shape
print("pose", [round(v, 2) for v in pose])
print("camera center (base)", np.round(C, 1).tolist(), "optical axis (base)", np.round(Rcam[:, 2], 3).tolist())

hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
cnt, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
report = {"frame": FRAME, "pose": pose, "camera_center_base": C.tolist(),
          "optical_axis_base": Rcam[:, 2].tolist(), "t_cam_assumed_mm": TCAM, "candidates": []}
for i in range(1, cnt):
    area = stats[i, cv2.CC_STAT_AREA]
    if area < 3000:
        continue
    x0, y0, ww, hh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    sel = (labels == i)
    ys, xs = np.nonzero(sel)
    zs = depth[ys, xs]
    ok = (zs > 30) & (zs < 2500)
    if ok.sum() < 500:
        continue
    xs, ys, zs = xs[ok], ys[ok], zs[ok]
    Xc = np.stack([(xs - I["cx"]) * zs / I["fx"], (ys - I["cy"]) * zs / I["fy"], zs], axis=1)
    Xb = (Rcam @ Xc.T).T + C
    ext = Xb.max(axis=0) - Xb.min(axis=0)
    # 主轴（PCA）
    cen = Xb.mean(axis=0)
    _, sv, vh = np.linalg.svd(Xb - cen)
    axis = vh[0]
    proj = (Xb - cen) @ axis
    entry = {"label": i, "area_px": int(area), "bbox": [int(x0), int(y0), int(ww), int(hh)],
             "centroid_px": [round(float(v), 1) for v in cents[i]],
             "n_points": int(ok.sum()), "extent_base_mm": np.round(ext, 1).tolist(),
             "pca_axis_base": np.round(axis, 3).tolist(), "pca_length_mm": round(float(proj.max() - proj.min()), 1),
             "center_base_mm": np.round(cen, 1).tolist(),
             "depth_median_mm": round(float(np.median(zs)), 1)}
    report["candidates"].append(entry)
    print(json.dumps(entry, ensure_ascii=False))
report["candidates"].sort(key=lambda e: -e["area_px"])
(RUN / f"target_localization_{FRAME:06d}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
# 可视化
vis = bgr.copy()
for e in report["candidates"][:4]:
    x0, y0, ww, hh = e["bbox"]
    cv2.rectangle(vis, (x0, y0), (x0 + ww, y0 + hh), (0, 0, 255), 3)
    cv2.putText(vis, f"{e['area_px']}px d={e['depth_median_mm']:.0f} len={e['pca_length_mm']:.0f}",
                (x0, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imwrite(str(RUN / f"target_candidates_{FRAME:06d}.png"), vis)
print("wrote target_candidates_%06d.png" % FRAME)