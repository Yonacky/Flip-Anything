"""Read controller identity, TCP definition and load; optionally clear a sticky warning."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from flip_anything.config import load_config
from xarm.wrapper import XArmAPI

cfg = load_config(Path(__file__).resolve().parents[2] / "configs/platform.json")
arm = XArmAPI(cfg["robot"]["ip"], is_radian=False, do_not_open=True)
arm.connect()
res = {"connected": arm.connected, "axis": arm.axis, "version": arm.version,
       "controller_sn": arm.sn, "device_type": arm.device_type, "is_sim": bool(arm.is_simulation_robot)}
try:
    code, val = arm.get_tcp_offset()
    res["get_tcp_offset"] = [code, list(val) if val else val]
    res["tcp_offset_attribute"] = list(arm.tcp_offset)
except Exception as exc:
    res["get_tcp_offset_error"] = f"{type(exc).__name__}: {exc}"
try:
    code, val = arm.get_tcp_load()
    res["get_tcp_load"] = [code, val]
    res["tcp_load_attribute"] = arm.tcp_load
except Exception as exc:
    res["get_tcp_load_error"] = f"{type(exc).__name__}: {exc}"
res["err_warn_before"] = arm.get_err_warn_code()
res["state_before"] = arm.state
res["cmdnum_before"] = arm.cmd_num
if __import__("sys").argv[1:] == ["--clear"]:
    res["clean_error"] = arm.clean_error()
    res["clean_warn"] = arm.clean_warn()
    time.sleep(1.0)
    res["err_warn_after"] = arm.get_err_warn_code()
    res["state_after"] = arm.state
res["robotiq_status_attempt"] = None
try:
    code, raw = arm.robotiq_get_status(check_baud=False)
    res["robotiq_status_attempt"] = {"code": code, "raw": list(raw) if raw else raw,
                                     "status": dict(arm.robotiq_status)}
except Exception as exc:
    res["robotiq_status_attempt"] = f"{type(exc).__name__}: {exc}"
arm.disconnect()
print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
