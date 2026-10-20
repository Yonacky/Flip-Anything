"""Take stock of the current physical situation: pose, flange position, table plane from depth."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from flip_anything.config import load_config
from flip_anything.geometry import pose_matrix
from xarm.wrapper import XArmAPI

cfg = load_config(Path(__file__).resolve().parents[2] / "configs/platform.json")
arm = XArmAPI(cfg["robot"]["ip"], is_radian=False, do_not_open=True)
arm.connect()
out = {}
out["err"] = arm.get_err_warn_code()
out["state"] = arm.get_state()
pose = list(arm.get_position(is_radian=False)[1])
joints = list(arm.get_servo_angle(is_radian=False)[1])[:6]
out["pose_tcp"] = pose
out["joints"] = joints
out["tcp_offset"] = list(arm.tcp_offset)
T = np.asarray(pose_matrix(pose))
flange = T[:3, 3] - T[:3, 2] * 174.0
out["flange_position_estimate"] = flange.tolist()
out["tool_z_axis"] = T[:3, 2].tolist()
out["tool_x_axis"] = T[:3, 0].tolist()
out["tool_y_axis"] = T[:3, 1].tolist()
try:
    code, fk = arm.get_forward_kinematics(joints, input_is_radian=False, return_is_radian=False)
    out["forward_kinematics"] = {"code": code, "pose": list(fk)}
except Exception as exc:
    out["forward_kinematics_error"] = f"{type(exc).__name__}: {exc}"
arm.disconnect()
print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
