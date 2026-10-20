"""Back-project aligned depth into the colour optical frame and summarise the visible geometry."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(run, index):
    folder = ROOT / 'logs' / 'experiments' / 'legacy' / run
    depth = np.load(folder / f"frame_{index:06d}_depth_raw.npy")
    meta = None
    for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["kind"] == "camera_started":
            meta = event["payload"]["metadata"]
    return depth, meta


def main():
    run = sys.argv[1]
    index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    depth, meta = load(run, index)
    intrinsic = meta["factory_camera_param"]["color_intrinsic"]
    fx, fy, cx, cy = intrinsic["fx"], intrinsic["fy"], intrinsic["cx"], intrinsic["cy"]
    h, w = depth.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64)
    valid = z > 0
    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z
    pts = np.stack([x[valid], y[valid], z[valid]], axis=1)
    print(f"run={run} index={index} valid={valid.sum()} of {depth.size} ({100*valid.mean():.1f}%)")
    print("z mm percentiles:", {p: int(np.percentile(pts[:, 2], p)) for p in (1, 5, 25, 50, 75, 95, 99)})
    print("x mm range:", int(pts[:, 0].min()), int(pts[:, 0].max()))
    print("y mm range:", int(pts[:, 1].min()), int(pts[:, 1].max()))
    # coarse occupancy of visible depth by image region
    for name, sl in (("top_left", (slice(0, h // 2), slice(0, w // 2))),
                     ("top_right", (slice(0, h // 2), slice(w // 2, w))),
                     ("bottom_left", (slice(h // 2, h), slice(0, w // 2))),
                     ("bottom_right", (slice(h // 2, h), slice(w // 2, w))),
                     ("center", (slice(h // 3, 2 * h // 3), slice(w // 3, 2 * w // 3)))):
        block = depth[sl]
        ok = block[block > 0]
        print(f"  {name}: valid={ok.size} median_z={int(np.median(ok)) if ok.size else None}")
    out = ROOT / 'logs' / 'experiments' / 'legacy' / run / f"points_{index:06d}.npz"
    np.savez_compressed(out, points=pts.astype(np.float32))
    print("saved point cloud:", out)


if __name__ == "__main__":
    main()
