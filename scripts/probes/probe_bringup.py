"""Read-only bring-up probe: robot identity/state + camera open diagnostics. No motion."""
import json, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from flip_anything.config import load_config
from flip_anything.robot import XArmObserver

cfg = load_config(Path(__file__).resolve().parents[2] / "configs/platform.json")
out = {"robot": {}, "camera": {}}
rob = XArmObserver(cfg["robot"])
try:
    out["robot"]["open"] = rob.open()
    out["robot"]["read"] = rob.read()
except Exception as exc:
    out["robot"]["error"] = f"{type(exc).__name__}: {exc}"
finally:
    try: rob.close()
    except Exception: pass
print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
