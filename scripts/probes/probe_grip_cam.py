"""Gripper read state + camera open diagnosis (read-only, no motion, no activation)."""
import json, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from flip_anything.config import load_config
from xarm.wrapper import XArmAPI

cfg = load_config(Path(__file__).resolve().parents[2] / "configs/platform.json")
res = {}
arm = XArmAPI(cfg["robot"]["ip"], is_radian=False, do_not_open=True)
arm.connect()
try:
    res["connected"] = arm.connected
    res["axis"] = arm.axis
    res["is_simulation"] = bool(arm.is_simulation_robot)
    res["mode"] = arm.mode
    res["state"] = arm.state
    res["tcp_offset"] = list(arm.tcp_offset)
    res["tcp_load"] = arm.tcp_load
    res["version"] = arm.version
    res["robot_sn"] = arm.sn
    try:
        code, raw = arm.robotiq_get_status(check_baud=False)
        res["robotiq_get_status"] = {"code": code, "raw": list(raw) if raw is not None else None,
                                     "status": dict(arm.robotiq_status)}
    except Exception as exc:
        res["robotiq_get_status_error"] = f"{type(exc).__name__}: {exc}"
    for name in ("get_gripper_position", "get_gripper_err_code"):
        try:
            fn = getattr(arm, name)
            res[name] = fn()
        except Exception as exc:
            res[name + "_error"] = f"{type(exc).__name__}: {exc}"
finally:
    arm.disconnect()

# camera diagnosis
try:
    import cv2
    res["cv2"] = cv2.__version__
    res["cv2_dshow_build"] = bool(cv2.getBuildInformation().find("DirectShow") >= 0)
    for idx in range(0, 3):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        ok = cap.isOpened()
        entry = {"isOpened": ok}
        if ok:
            ret, frame = cap.read()
            entry["read"] = bool(ret)
            entry["shape"] = None if frame is None else list(frame.shape)
        else:
            entry["read"] = False
        cap.release()
        res.setdefault("cv2_dshow_index", {})[str(idx)] = entry
except Exception as exc:
    res["cv2_error"] = f"{type(exc).__name__}: {exc}"

try:
    from flip_anything.camera import OrbbecCamera
    res["orbbec_enumerate"] = OrbbecCamera.enumerate()
except Exception as exc:
    res["orbbec_enumerate_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
