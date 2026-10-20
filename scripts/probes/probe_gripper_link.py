"""Gripper link probe: robot tool port vs control-box RS485, timeouts, raw response codes."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from flip_anything.config import load_config
from xarm.wrapper import XArmAPI

cfg = load_config(Path(__file__).resolve().parents[2] / "configs/platform.json")
arm = XArmAPI(cfg["robot"]["ip"], is_radian=False, do_not_open=True)
arm.connect()
out = {}
arm.clean_error(); arm.clean_warn(); time.sleep(0.8)
arm.motion_enable(enable=True); arm.set_mode(0); arm.set_state(0); time.sleep(0.8)
frame = [0x09, 0x03, 0x07, 0xD0, 0x00, 0x03]
for label, target in (("robot", "robot"), ("control_box", "control_box")):
    entry = {"baud_before": arm.get_rs485_baudrate(target=target)}
    try:
        entry["set_timeout"] = arm.set_rs485_timeout(200, target=target)
    except Exception as exc:
        entry["set_timeout"] = f"{type(exc).__name__}: {exc}"
    try:
        entry["set_baud"] = arm.set_rs485_baudrate(115200, target=target)
    except Exception as exc:
        entry["set_baud"] = f"{type(exc).__name__}: {exc}"
    time.sleep(0.5)
    entry["baud_after"] = arm.get_rs485_baudrate(target=target)
    try:
        entry["raw"] = arm.set_rs485_data(frame, min_res_len=9, target=target, protocol="modbus_rtu")
    except Exception as exc:
        entry["raw"] = f"{type(exc).__name__}: {exc}"
    try:
        entry["raw_short"] = arm.set_rs485_data([0x09, 0x03, 0x07, 0xD0, 0x00, 0x01], min_res_len=5,
                                               target=target, protocol="modbus_rtu")
    except Exception as exc:
        entry["raw_short"] = f"{type(exc).__name__}: {exc}"
    entry["err_after"] = arm.get_err_warn_code()
    out[label] = entry
    arm.clean_error(); arm.clean_warn(); time.sleep(0.5)
out["err_final"] = arm.get_err_warn_code()
out["state_final"] = arm.get_state()
arm.disconnect()
print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
