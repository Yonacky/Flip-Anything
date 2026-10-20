"""Explicit, bounded joint demonstration; never clears faults or gripper state."""
import argparse
import json
import time
import hashlib
from pathlib import Path

from flip_anything.config import load_config
from flip_anything.robot import checked
from flip_anything.trial import Trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--joint", type=int, choices=(1, 5), default=5)
    parser.add_argument("--expected-j5", "--expected-angle", dest="expected_angle", required=True, type=float)
    parser.add_argument("--observe", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Requires explicit --execute after scene and state inspection")
    cfg = load_config("configs/platform.json")
    trial = Trial(cfg["recording"]["root"], cfg, "hardware",
                  context={"task": "user_authorized_joint_out_and_back", "joint": args.joint, "delta_deg": 5,
                           "speed_deg_s": 3, "acceleration_deg_s2": 10,
                           "expected_angle": args.expected_angle,
                           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    from xarm.wrapper import XArmAPI
    arm = XArmAPI(cfg["robot"]["ip"], is_radian=False, do_not_open=True)
    status, error, prepared = "failed", None, False
    camera = None
    joint_index = args.joint - 1

    def call(name, *pos, **kwargs):
        trial.event("sdk_request", method=name, args=list(pos), kwargs=kwargs)
        result = getattr(arm, name)(*pos, **kwargs)
        trial.event("sdk_result", method=name, result=result)
        code = result[0] if isinstance(result, tuple) else result
        if code != 0:
            raise RuntimeError(f"{name}: SDK code {code}")
        return result

    def sample():
        angles = list(checked(arm.get_servo_angle(is_radian=False), "get_servo_angle"))[:6]
        faults = checked(arm.get_err_warn_code(), "get_err_warn_code")
        state = checked(arm.get_state(), "get_state")
        trial.event("robot_sample", joint_deg=angles, state=state, error_warn=faults)
        if faults[0] or faults[1]:
            raise RuntimeError(f"Controller error/warning: {faults}")
        return angles, state

    def move(target):
        # One command at a time; bounded monitoring stops on interruption/fault/timeout.
        call("set_servo_angle", servo_id=args.joint, angle=target, speed=3, mvacc=10,
             is_radian=False, wait=False)
        deadline = time.monotonic() + 10
        stable = 0
        while time.monotonic() < deadline:
            angles, state = sample()
            if state in (4, 5):
                raise RuntimeError(f"Controller stopped during command: state={state}")
            if any(abs(angles[i] - initial[i]) > 0.3 for i in range(6) if i != joint_index):
                raise RuntimeError("Unexpected motion on another joint")
            stable = stable + 1 if abs(angles[joint_index] - target) < 0.15 and state != 1 else 0
            if stable >= 2:
                return angles
            time.sleep(0.1)
        raise TimeoutError("Joint target not reached within 10s")

    try:
        arm.connect()
        if not arm.connected or arm.axis != 6:
            raise RuntimeError("Expected connected xArm6")
        trial.event("controller_mode", simulation_robot=arm.is_simulation_robot)
        if arm.is_simulation_robot:
            raise RuntimeError("Controller is in simulation mode; refusing demonstration")
        if args.observe:
            from flip_anything.camera import OrbbecCamera
            camera = OrbbecCamera(cfg["camera"])
            trial.event("camera_started", metadata=camera.open())
            trial.event("observation", phase="before", camera=camera.capture(trial.path, 0))
        initial, state = sample()
        if state not in (0, 2, 4, 5):
            raise RuntimeError(f"Unexpected initial controller state {state}")
        if abs(initial[joint_index] - args.expected_angle) > 0.5:
            raise RuntimeError("Joint changed since scene inspection; obtain fresh precheck")
        target = initial[joint_index] + 5
        joints = list(initial)
        joints[joint_index] = target
        if checked(arm.is_joint_limit(joints, is_radian=False), "is_joint_limit"):
            raise RuntimeError("Target exceeds joint limits")
        current_pose = checked(arm.get_position(is_radian=False), "get_position")
        target_pose = checked(arm.get_forward_kinematics(joints, input_is_radian=False, return_is_radian=False), "get_forward_kinematics")
        displacement = sum((target_pose[i] - current_pose[i])**2 for i in range(3))**0.5
        trial.event("preflight", initial_joint_deg=initial, target_joint_deg=joints,
                    current_pose_mm_deg=current_pose, target_pose_mm_deg=target_pose,
                    endpoint_displacement_mm=displacement,
                    limitation="joint limits/FK only; no automatic collision or table-plane validation")
        bound = 60 if args.joint == 1 else 30
        if displacement > bound:
            raise RuntimeError(f"TCP displacement exceeds {bound}mm demonstration bound")
        # Stop/flush existing queue before changing modes or enabling. Never auto-resume old commands.
        prepared = True
        call("set_state", 4)
        call("motion_enable", True)
        call("set_mode", 0)
        call("set_state", 0)
        sample()
        peak = move(target)
        if camera:
            trial.event("observation", phase="peak", camera=camera.capture(trial.path, 1))
        hold_end = time.monotonic() + 3
        while time.monotonic() < hold_end:
            _, state = sample()
            if state in (4, 5):
                raise RuntimeError("Stopped during peak hold; no return command sent")
            time.sleep(0.1)
        final = move(initial[joint_index])
        trial.event("demo_completed", joint=args.joint, initial_angle_deg=initial[joint_index],
                    peak_angle_deg=peak[joint_index], final_angle_deg=final[joint_index],
                    bottle_task_outcome="not_executed")
        if camera:
            trial.event("observation", phase="returned", camera=camera.capture(trial.path, 2))
        status = "completed"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        trial.event("error", message=error)
    finally:
        if prepared and arm.connected:
            try:
                call("set_state", 4)
            except Exception as exc:
                status = "failed"
                error = (error or "") + f"; stop failed: {exc}"
                trial.event("stop_error", message=str(exc))
        if camera:
            try:
                camera.close()
            except Exception as exc:
                status = "failed"
                error = (error or "") + f"; camera cleanup failed: {exc}"
        arm.disconnect()
        trial.finish(status, error)
    print(json.dumps({"run": str(trial.path), "status": status, "error": error}, ensure_ascii=False))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
