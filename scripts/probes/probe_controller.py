"""Full read-only controller state characterization; no motion, no writes."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from flip_anything.config import load_config
from xarm.wrapper import XArmAPI

cfg = load_config(Path(__file__).resolve().parents[2] / "configs/platform.json")
arm = XArmAPI(cfg["robot"]["ip"], is_radian=False, do_not_open=True)
arm.connect()
res = {}
def grab(name, *args, **kw):
    try:
        fn = getattr(arm, name)
        out = fn(*args, **kw)
        res[name] = out if not hasattr(out, "tolist") else out.tolist()
    except Exception as exc:
        res[name + "_error"] = f"{type(exc).__name__}: {exc}"
try:
    for name in ("get_state", "get_err_warn_code", "get_cmdnum", "get_mode", "get_position",
                 "get_servo_angle", "get_joint_states", "get_tcp_offset", "get_tcp_load",
                 "get_tcp_maxacc", "get_joint_maxacc", "get_gripper_position", "get_gripper_err_code",
                 "get_ft_sensor_mode", "get_robot_sn", "get_version", "get_reduced_mode",
                 "get_reduced_states", "get_collision_sensitivity", "get_teach_sensitivity",
                 "get_safety_boundary", "get_tgpio_digital", "get_tgpio_analog", "get_linear_limit",
                 "get_mount_direction", "get_world_offset", "get_torque_limit", "get_error_warn_code"):
        grab(name)
    try:
        code, raw = arm.robotiq_get_status(check_baud=False)
        res["robotiq_get_status"] = {"code": code, "status": dict(arm.robotiq_status)}
    except Exception as exc:
        res["robotiq_get_status_error"] = f"{type(exc).__name__}: {exc}"
    try:
        res["gripper_type_check"] = arm.get_gripper_grip_velocity()
    except Exception as exc:
        res["gripper_type_check_error"] = f"{type(exc).__name__}: {exc}"
finally:
    arm.disconnect()
print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
