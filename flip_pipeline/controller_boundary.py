"""Only SI SE(3) <-> xArm controller pose boundary; no SDK or motion calls.

Controller pose is [x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg]. Its fixed-axis
RPY convention is Rz(yaw) @ Ry(pitch) @ Rx(roll), SciPy *lowercase* ``xyz``.
Conversion does not certify an interpolation, a reachable pose or a free path.
"""
from copy import deepcopy
import math
import warnings

import numpy as np

from .calibration import validate_transform


FRAME = "base"
UNITS = "mm_deg"
ORIENTATION_CONVENTION = "extrinsic_xyz_rpy"
MOTION_SPEED_UNITS = {"cartesian_translation": "mm_s", "joint": "deg_s"}


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _pose(value, name="pose"):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 6:
        raise ValueError(f"{name} must contain six numeric mm/degree values")
    return np.array([_number(x, name) for x in value], dtype=float)


def _convention(frame, units, orientation_convention):
    if frame != FRAME or units != UNITS or orientation_convention != ORIENTATION_CONVENTION:
        raise ValueError("xArm boundary requires frame=base, units=mm_deg, orientation_convention=extrinsic_xyz_rpy")


def controller_pose_to_matrix(pose, *, frame=FRAME, units=UNITS, orientation_convention=ORIENTATION_CONVENTION):
    """Convert a raw controller receipt into base_from_tool, metres/radians."""
    _convention(frame, units, orientation_convention)
    p = _pose(pose)
    T = np.eye(4)
    roll, pitch, yaw = np.deg2rad(p[3:])
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    T[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                  [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                  [-sp, cp * sr, cp * cr]]
    T[:3, 3] = p[:3] / 1000.0
    return validate_transform(T)


def matrix_to_controller_move(base_from_tool, *, grip_required, motion_space, speed, **optional_fields):
    """Build a typed command from SI geometry, with explicit interpolation/speed.

    The existing worker chooses Cartesian translation only below 0.5 degrees of
    orientation change; otherwise it uses joint interpolation. At authorization
    and again before dispatch call validate_motion_against_start with a fresh
    controller pose. Do not assume the Cartesian endpoints define a joint path.
    """
    T = validate_transform(base_from_tool, "base_from_tool")
    from scipy.spatial.transform import Rotation
    if motion_space not in MOTION_SPEED_UNITS:
        raise ValueError("motion_space must be cartesian_translation or joint")
    protected = {"action", "pose", "frame", "units", "orientation_convention", "grip_required",
                 "motion_space", "speed", "speed_units", "near_euler_singularity"}
    if protected & optional_fields.keys():
        raise ValueError("cannot override converted pose or unit metadata")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", UserWarning)
        rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    singular = bool(seen) or abs(math.cos(math.radians(float(rpy[1])))) < 1e-6
    p = np.concatenate((T[:3, 3] * 1000.0, rpy))
    # At singularity SciPy fixes the third angle to zero; pose remains equivalent.
    # Its near-singular cutoff can lose <= 1e-7 rad, not a different Euler order.
    if not np.allclose(controller_pose_to_matrix(p), T, rtol=0, atol=3e-7):
        raise ValueError("controller pose conversion failed SE(3) roundtrip")
    request = {"action": "move", "pose": p.tolist(), "frame": FRAME, "units": UNITS,
               "orientation_convention": ORIENTATION_CONVENTION,
               "grip_required": grip_required, "motion_space": motion_space,
               "speed": speed, "speed_units": MOTION_SPEED_UNITS[motion_space],
               "near_euler_singularity": singular, **optional_fields}
    return validate_controller_request(request)


def validate_controller_request(request):
    """Reject ambiguous unit conventions before the legacy worker sees a payload.

    Gripper registers use their own 0..255 integer domain, not mm/degree units.
    Optional expected_pose is always the same raw controller pose convention.
    """
    if not isinstance(request, dict) or request.get("action") not in {"move", "grip"}:
        raise ValueError("only move/grip commands are authorized by this boundary")
    if not isinstance(request.get("grip_required"), bool):
        raise ValueError("grip_required must be explicit boolean")
    checked = deepcopy(request)
    if request["action"] == "move":
        _convention(request.get("frame"), request.get("units"), request.get("orientation_convention"))
        checked["pose"] = _pose(request.get("pose")).tolist()
        motion = request.get("motion_space")
        if motion not in MOTION_SPEED_UNITS or request.get("speed_units") != MOTION_SPEED_UNITS[motion]:
            raise ValueError("motion_space and speed_units must match actual controller interpolation")
        checked["speed"] = _number(request.get("speed"), "speed", positive=True)
        for name in ("max_translation_mm", "max_rotation_deg", "max_joint_deg"):
            if name in request:
                checked[name] = _number(request[name], name, positive=True)
        if "near_euler_singularity" in request and not isinstance(request["near_euler_singularity"], bool):
            raise ValueError("near_euler_singularity must be a boolean")
    else:
        for name, default in (("pos", None), ("speed", 40), ("force", 10)):
            value = request.get(name, default)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
                raise ValueError(f"Robotiq {name} must be an integer register value 0..255")
        if "units" in request and request["units"] != "robotiq_registers":
            raise ValueError("gripper units must be robotiq_registers when declared")
        if "speed_units" in request and request["speed_units"] != "robotiq_registers":
            raise ValueError("gripper speed is a register value, not mm_s or deg_s")
    if "expected_pose" in request:
        checked["expected_pose"] = _pose(request["expected_pose"], "expected_pose").tolist()
    if "timeout_s" in request:
        checked["timeout_s"] = _number(request["timeout_s"], "timeout_s", positive=True)
    if "capture" in request and not isinstance(request["capture"], bool):
        raise ValueError("capture must be a boolean")
    return checked


def validate_motion_against_start(request, start_pose):
    """Verify the declared motion/speed branch matches existing worker behavior.

    The 0.5 degree boundary deliberately matches session.run_step. This check is
    required again on the latest pose immediately before motion, because pose
    drift can change which branch the legacy worker selects.
    """
    checked = validate_controller_request(request)
    if checked["action"] != "move":
        return {"action": "grip", "motion_space": None}
    start = controller_pose_to_matrix(start_pose)
    end = controller_pose_to_matrix(checked["pose"])
    cosine = float((np.trace(start[:3, :3].T @ end[:3, :3]) - 1.0) / 2.0)
    rotation_deg = float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    if abs(rotation_deg - 0.5) < 1e-6:
        raise ValueError("orientation change too close to legacy interpolation threshold; choose an unambiguous segment")
    actual = "cartesian_translation" if rotation_deg < 0.5 else "joint"
    if checked["motion_space"] != actual:
        raise ValueError(f"declared motion_space {checked['motion_space']} differs from worker branch {actual}")
    return {"action": "move", "motion_space": actual, "speed_units": MOTION_SPEED_UNITS[actual],
            "rotation_deg": rotation_deg, "translation_mm": float(np.linalg.norm(end[:3, 3] - start[:3, 3]) * 1000)}
