import ipaddress
import json
from pathlib import Path


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("schema_version") != 1:
        raise ValueError("Unsupported platform schema_version")
    ipaddress.IPv4Address(cfg["robot"]["ip"])
    if cfg["robot"]["model"] != "xArm6":
        raise ValueError("This adapter targets xArm6")
    if cfg["robot"]["read_gripper"] and cfg["robot"]["gripper_connection"] != "xarm_rs485":
        raise ValueError("Gripper reads require confirmed xarm_rs485 connection")
    timeout = cfg["camera"]["timeout_ms"]
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 100 <= timeout <= 10000:
        raise ValueError("camera.timeout_ms must be 100..10000")
    skew = cfg["recording"]["max_rgbd_skew_ms"]
    if isinstance(skew, bool) or not isinstance(skew, (int, float)) or not 0 <= skew <= 1000:
        raise ValueError("Invalid max_rgbd_skew_ms")
    for section, key in (("recording", "root"), ("calibration", "file")):
        cfg[section][key] = str((path.parent / cfg[section][key]).resolve())
    if cfg["camera"].get("log_dir"):
        cfg["camera"]["log_dir"] = str((path.parent / cfg["camera"]["log_dir"]).resolve())
    return cfg
