"""Single-shot camera bring-up test: enumerate, open, warm up, capture one frame."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from flip_anything.camera import OrbbecCamera
from flip_anything.config import load_config

cfg = load_config(Path(__file__).resolve().parents[2] / "configs/platform.json")
out = Path(cfg['recording']['root']) / "_camera_bringup"
out.mkdir(parents=True, exist_ok=True)
report = {}
try:
    report["enumerate"] = OrbbecCamera.enumerate()
except Exception as exc:
    report["enumerate_error"] = f"{type(exc).__name__}: {exc}"
cam = OrbbecCamera(cfg["camera"])
try:
    t0 = time.monotonic()
    report["open"] = cam.open()
    report["open_seconds"] = round(time.monotonic() - t0, 2)
    dropped = 0
    for _ in range(12):
        if cam.pipeline.wait_for_frames(2000) is None:
            dropped += 1
    report["warmup_dropped"] = dropped
    frame = cam.capture(out, 0)
    report["capture"] = frame
except Exception as exc:
    report["error"] = f"{type(exc).__name__}: {exc}"
finally:
    try:
        cam.close()
    except Exception:
        pass
print(json.dumps(report, ensure_ascii=False, indent=2, default=str))

