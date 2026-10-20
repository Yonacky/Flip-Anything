"""Clear C31 and test a very gentle +Z translation to separate real impact from false collision."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from flip_anything.config import load_config
from xarm.wrapper import XArmAPI

cfg = load_config(Path(__file__).resolve().parents[2] / "configs/platform.json")
arm = XArmAPI(cfg["robot"]["ip"], is_radian=False, do_not_open=True)
arm.connect()
out = {}
out["err_before"] = arm.get_err_warn_code()
out["state_before"] = arm.get_state()
out["clean_error"] = arm.clean_error()
out["clean_warn"] = arm.clean_warn()
time.sleep(1.0)
out["err_cleared"] = arm.get_err_warn_code()
out["motion_enable"] = arm.motion_enable(enable=True)
out["set_mode"] = arm.set_mode(0)
out["set_state"] = arm.set_state(0)
time.sleep(1.0)
out["tcp_load_reported"] = arm.tcp_load
out["tcp_offset_property"] = list(arm.tcp_offset)
pose = list(arm.get_position(is_radian=False)[1])
out["pose_before"] = pose
target = list(pose)
target[2] += 20.0
out["move_gentle"] = arm.set_position(x=target[0], y=target[1], z=target[2], roll=target[3],
                                      pitch=target[4], yaw=target[5], speed=10, mvacc=20, wait=True, timeout=20)
out["err_after_move"] = arm.get_err_warn_code()
out["pose_after"] = list(arm.get_position(is_radian=False)[1])
out["state_after"] = arm.get_state()
out["set_state_stop"] = arm.set_state(4)
arm.disconnect()
print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
