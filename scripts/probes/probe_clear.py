"""Clear C19, re-enable motion, then read gripper/TCP state. No motion commands."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from flip_anything.config import load_config
from xarm.wrapper import XArmAPI

cfg = load_config(Path(__file__).resolve().parents[2] / "configs/platform.json")
arm = XArmAPI(cfg["robot"]["ip"], is_radian=False, do_not_open=True)
arm.connect()
out = {"tcp_offset_property": list(arm.tcp_offset)}
out["warn_before"] = arm.get_err_warn_code()
out["state_before"] = arm.get_state()
out["clean_error"] = arm.clean_error()
out["clean_warn"] = arm.clean_warn()
time.sleep(1.5)
out["err_after_clean"] = arm.get_err_warn_code()
out["enable"] = arm.motion_enable(enable=True)
out["set_mode"] = arm.set_mode(0)
out["set_state0"] = arm.set_state(0)
time.sleep(1.0)
out["state_after_enable"] = arm.get_state()
out["err_after_enable"] = arm.get_err_warn_code()
try:
    code, raw = arm.robotiq_get_status(check_baud=False)
    out["robotiq_status"] = {"code": code, "raw": list(raw) if raw else raw, "decoded": dict(arm.robotiq_status)}
except Exception as exc:
    out["robotiq_status_error"] = f"{type(exc).__name__}: {exc}"
for baud in (115200, 57600, 19200, 9600):
    try:
        out[f"set_baud_{baud}"] = arm.set_rs485_baudrate(baud, check=False) if "check" in arm.set_rs485_baudrate.__code__.co_varnames else arm.set_rs485_baudrate(baud)
    except Exception as exc:
        out[f"set_baud_{baud}"] = f"{type(exc).__name__}: {exc}"
    time.sleep(0.3)
    try:
        code, raw = arm.robotiq_get_status(check_baud=False)
        out[f"robotiq_at_{baud}"] = {"code": code, "raw": list(raw) if raw else raw}
    except Exception as exc:
        out[f"robotiq_at_{baud}"] = f"{type(exc).__name__}: {exc}"
out["err_end"] = arm.get_err_warn_code()
arm.disconnect()
print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
