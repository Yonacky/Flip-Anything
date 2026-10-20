"""Clear sticky warning, confirm TCP definition, then test the RS485 gripper link."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from flip_anything.config import load_config
from xarm.wrapper import XArmAPI

cfg = load_config(Path(__file__).resolve().parents[2] / "configs/platform.json")
arm = XArmAPI(cfg["robot"]["ip"], is_radian=False, do_not_open=True)
arm.connect()
out = {}
out["tcp_offset"] = list(arm.tcp_offset)
out["position_offset"] = list(arm.position_offset)
out["warn_before"] = arm.get_err_warn_code()
out["clean_warn"] = arm.clean_warn()
time.sleep(1.0)
out["warn_after_clean"] = arm.get_err_warn_code()
out["set_baud"] = arm.set_rs485_baudrate(115200)
time.sleep(0.3)
out["baud_now"] = arm.get_rs485_baudrate()
try:
    out["robotiq_after_baud"] = arm.robotiq_get_status(check_baud=False)
    out["robotiq_status"] = dict(arm.robotiq_status)
except Exception as exc:
    out["robotiq_error"] = f"{type(exc).__name__}: {exc}"
try:
    frame = [0x09, 0x03, 0x07, 0xD0, 0x00, 0x03]
    out["raw_rs485"] = arm.set_rs485_data(frame, min_res_len=9, target="robot", protocol="modbus_rtu")
except Exception as exc:
    out["raw_rs485_error"] = f"{type(exc).__name__}: {exc}"
out["warn_end"] = arm.get_err_warn_code()
arm.disconnect()
print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
