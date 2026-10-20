"""Read-only RS485/gripper link characterization. No activation, no motion, no writes."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from flip_anything.config import load_config
from xarm.wrapper import XArmAPI

cfg = load_config(Path(__file__).resolve().parents[2] / "configs/platform.json")
arm = XArmAPI(cfg["robot"]["ip"], is_radian=False, do_not_open=True)
arm.connect()
out = {"warn_before": arm.get_err_warn_code()}
try:
    out["rs485_baudrate"] = arm.get_rs485_baudrate()
except Exception as exc:
    out["rs485_baudrate_error"] = f"{type(exc).__name__}: {exc}"
try:
    out["rs485_timeout"] = arm.get_rs485_timeout()
except Exception as exc:
    out["rs485_timeout_error"] = f"{type(exc).__name__}: {exc}"
try:
    out["report_tcp_offset"] = list(arm.report._tcp_offset)
    out["report_tcp_load"] = [arm.report._tcp_load.mass, list(arm.report._tcp_load.center)]
except Exception as exc:
    out["report_tcp_error"] = f"{type(exc).__name__}: {exc}"
# Raw Modbus RTU request for Robotiq status (slave 9, fn 3, addr 0x07D0, 3 regs) at the current baud
try:
    frame = [0x09, 0x03, 0x07, 0xD0, 0x00, 0x03]
    out["raw_rs485_probe"] = arm.set_rs485_data(frame, min_res_len=9, target="robot", protocol="modbus_rtu")
except Exception as exc:
    out["raw_rs485_probe_error"] = f"{type(exc).__name__}: {exc}"
for baud in (115200, 19200, 9600, 57600, 38400):
    entry = {}
    try:
        entry["check"] = arm.checkset_modbus_baud(baud, check=False)
        time.sleep(0.3)
        frame = [0x09, 0x03, 0x07, 0xD0, 0x00, 0x03]
        entry["raw"] = arm.set_rs485_data(frame, min_res_len=9, target="robot", protocol="modbus_rtu")
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
    out[f"baud_{baud}"] = entry
out["warn_after"] = arm.get_err_warn_code()
arm.disconnect()
print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
