"""Establish a working RS485 gripper link (correct baud + timeout), then activate and open."""
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
out["set_timeout"] = arm.set_rs485_timeout(200, target="robot")
out["set_baud"] = arm.set_rs485_baudrate(115200, target="robot")
time.sleep(0.4)
out["baud"] = arm.get_rs485_baudrate(target="robot")
out["timeout"] = arm.get_rs485_timeout(target="robot")
def status(tag):
    code, raw = arm.set_rs485_data([0x09, 0x03, 0x07, 0xD0, 0x00, 0x03], min_res_len=9,
                                   target="robot", protocol="modbus_rtu")
    entry = {"code": code, "raw": list(raw) if raw else raw}
    if code == 0 and len(raw) >= 7:
        st = raw[3]
        entry["decoded"] = {"gOBJ": (st & 0xC0) >> 6, "gSTA": (st & 0x30) >> 4,
                            "gGTO": (st & 0x08) >> 3, "gACT": st & 0x01,
                            "gFLT": raw[5] & 0x0F if len(raw) >= 7 else None,
                            "gPR": raw[6] if len(raw) >= 7 else None,
                            "gPO": raw[7] if len(raw) >= 9 else None,
                            "gCU": raw[8] if len(raw) >= 9 else None}
    out[tag] = entry
    return entry
status("status_initial")
# Robotiq activation: clear rACT/rGTO + set rACT (0x01 0x00 0x00 0x00 0x00 0x00)
out["activate_write"] = arm.set_rs485_data([0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06,
                                            0x01, 0x00, 0x00, 0x00, 0x00, 0x00],
                                           min_res_len=8, target="robot", protocol="modbus_rtu")
for i in range(6):
    time.sleep(0.7)
    e = status(f"status_after_activate_{i}")
    if e.get("decoded", {}).get("gSTA") == 3:
        break
out["robotiq_get_status_call"] = None
try:
    code, raw = arm.robotiq_get_status(check_baud=False)
    out["robotiq_get_status_call"] = {"code": code, "raw": list(raw) if raw else raw,
                                      "decoded": dict(arm.robotiq_status)}
except Exception as exc:
    out["robotiq_get_status_call"] = f"{type(exc).__name__}: {exc}"
# open the gripper fully (pos 0)
out["open_write"] = arm.set_rs485_data([0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06,
                                        0x09, 0x00, 0x00, 0x00, 0x40, 0x0A],
                                       min_res_len=8, target="robot", protocol="modbus_rtu")
for i in range(6):
    time.sleep(0.6)
    e = status(f"status_after_open_{i}")
    if e.get("decoded", {}).get("gOBJ") in (1, 2, 3):
        break
out["err_end"] = arm.get_err_warn_code()
arm.disconnect()
print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
