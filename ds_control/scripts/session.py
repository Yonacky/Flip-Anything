"""DeepSeek 现场会话执行器：单连接持有机械臂与腕部相机，按文件命令执行有界步骤。

设计依据（skill.md）：
- 一次持物过程中保持控制连接，并持续只读轮询夹爪状态；
- 检查抓取只用只读状态请求，绝不为了"确认夹紧"重发夹爪位置指令；
- 显式传 auto_enable=False，持物期间不让检查路径触发复位/重新激活/松爪；
- 每段动作有界、可核对；出现滑移、接触丢失、控制器异常或外部停止即中止当前阶段，
  不自动接着执行余下路径；
- 记录实际命令、反馈、原始图像、位姿与时间戳。
"""
import argparse
import hashlib
import os
import json
import math
import time
from pathlib import Path

import numpy as np
from xarm.wrapper import XArmAPI

from flip_anything.camera import OrbbecCamera
from flip_anything.config import load_config
from flip_anything.geometry import pose_matrix
from flip_anything.gripper import require_existing_grasp
from flip_anything.robot import checked
from flip_anything.trial import Trial
from flip_pipeline.supervision import execute_guarded, validate_grip_transition, halt_run_tasks, validate_live_acquisition
from flip_pipeline.controller_boundary import validate_motion_against_start
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

# 硬上限：任何命令都不能越过这些界限
HARD_MAX_TRANSLATION_MM = 100.0
HARD_MAX_ROTATION_DEG = 95.0
HARD_MAX_JOINT_DEG = 110.0
HARD_MAX_LINEAR_SPEED = 25.0
HARD_MAX_JOINT_SPEED = 10.0
IDLE_STATES = (0, 2, 4)


class Rejected(Exception):
    """规划阶段拒绝该命令；未向控制器下发任何动作。"""


def rotation_angle_deg(a, b):
    ra = np.array(pose_matrix(list(a)))[:3, :3]
    rb = np.array(pose_matrix(list(b)))[:3, :3]
    cosine = (np.trace(ra.T @ rb) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def unwrap_joints(q, reference):
    out = list(q)
    for index in range(len(out)):
        candidates = [out[index] + 360.0 * k for k in (-1, 0, 1)]
        out[index] = min(candidates, key=lambda value: abs(value - reference[index]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--duration-s", type=float, default=7200.0)
    parser.add_argument("--purpose", default="deepseek_workspace_execution")
    args = parser.parse_args()

    cfg = load_config("configs/platform.json")
    context = {
        "purpose": args.purpose,
        "object_id": "target_requires_current_observation",
        "success_definition": "inverted on table, released, stable; record support evidence and any human intervention",
        "model_id": "deepseek-flash via DSH",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    trial = Trial(cfg["recording"]["root"], cfg, "hardware", args.run_id, context)
    folder = trial.path
    inbox = folder / "commands"
    inbox.mkdir(parents=True, exist_ok=True)

    arm = XArmAPI(cfg["robot"]["ip"], do_not_open=True, is_radian=False)
    cam = OrbbecCamera(cfg["camera"])
    frame_index = 0
    handled = set()
    fatal = False

    def call(method, **kwargs):
        if (folder / "STOP").exists() and not (method == "set_state" and kwargs.get("state") == 4):
            raise Rejected("STOP is latched; refusing SDK mutation")
        trial.event("sdk_request", method=method, kwargs=kwargs)
        result = getattr(arm, method)(**kwargs)
        trial.event("sdk_result", method=method, result=result)
        code = result[0] if isinstance(result, tuple) else result
        if code != 0:
            raise RuntimeError(f"{method} returned SDK code {code}")
        return result

    def read_gripper():
        try:
            checked(arm.robotiq_get_status(check_baud=False), "robotiq_get_status")
            return dict(arm.robotiq_status), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def state(log=False, want_gripper=True):
        sample = {
            "sample_start_monotonic_ns": time.monotonic_ns(),
            "state": checked(arm.get_state(), "get_state"),
            "error_warn": list(checked(arm.get_err_warn_code(), "get_err_warn_code")),
            "pose": list(checked(arm.get_position(is_radian=False), "get_position")),
            "joint_deg": list(checked(arm.get_servo_angle(is_radian=False), "get_servo_angle"))[:6],
            "cmd_num": checked(arm.get_cmdnum(), "get_cmdnum"),
        }
        sample["gripper"], sample["gripper_error"] = read_gripper() if want_gripper else (None, None)
        sample["sample_end_monotonic_ns"] = time.monotonic_ns()
        sample["host_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if log:
            trial.event("robot_sample", **sample)
        write_result("live_state.json", sample)
        if sample["error_warn"][0]:
            raise RuntimeError(f"Controller error code {sample['error_warn'][0]}")
        if sample["error_warn"][1]:
            sample["warning_latched"] = sample["error_warn"][1]
        return sample

    def capture(label):
        nonlocal frame_index
        for _ in range(8):
            if cam.pipeline.wait_for_frames(1500) is None:
                raise TimeoutError("Camera frame timeout while settling")
            state(log=False)
        frame = cam.capture(folder, frame_index)
        frame_index += 1
        trial.event("observation", phase=label, camera=frame)
        return frame

    def write_result(name, payload):
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        target = folder / name
        # The fixed-name temp file intermittently fails with OSError 22 on this host and
        # used to kill the whole session; retry with unique names and never abort the run
        # just because a status snapshot could not be written.
        for attempt in range(6):
            temporary = folder / (name + f".{os.getpid()}.{attempt}.tmp")
            try:
                temporary.write_text(text, encoding="utf-8")
                temporary.replace(target)
                return
            except OSError:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                time.sleep(0.03 * (attempt + 1))
        try:
            target.write_text(text, encoding="utf-8")
        except OSError:
            pass

    def guard(payload, before):
        target = np.array(payload["pose"], dtype=float)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise Rejected("Target pose must be six finite numbers")
        expected = payload.get("expected_pose")
        if expected is not None:
            if np.linalg.norm(np.array(expected[:3]) - np.array(before["pose"][:3])) > 2.0:
                raise Rejected("Position differs from the expected start pose")
            if rotation_angle_deg(expected, before["pose"]) > 2.0:
                raise Rejected("Orientation differs from the expected start pose")
        translation = float(np.linalg.norm(target[:3] - np.array(before["pose"][:3])))
        rotation = rotation_angle_deg(target, before["pose"])
        limit_translation = min(float(payload.get("max_translation_mm", 60.0)), HARD_MAX_TRANSLATION_MM)
        limit_rotation = min(float(payload.get("max_rotation_deg", 60.0)), HARD_MAX_ROTATION_DEG)
        if translation > limit_translation:
            raise Rejected(f"Translation step {translation:.1f} mm exceeds {limit_translation} mm")
        if rotation > limit_rotation:
            raise Rejected(f"Rotation step {rotation:.1f} deg exceeds {limit_rotation} deg")
        return target, translation, rotation

    def solve_joints(target, reference):
        result = arm.get_inverse_kinematics(target.tolist(), input_is_radian=False,
                                            return_is_radian=False, ref_angles=reference)
        trial.event("sdk_result", method="get_inverse_kinematics", result=result)
        if result[0] != 0 or not result[1]:
            result = arm.get_inverse_kinematics(target.tolist(), input_is_radian=False,
                                                return_is_radian=False)
            trial.event("sdk_result", method="get_inverse_kinematics_retry", result=result)
        if result[0] != 0 or not result[1]:
            raise Rejected(f"Inverse kinematics failed with code {result[0]}")
        joints = unwrap_joints(list(result[1])[:6], reference)
        return joints, result[1]

    def run_step(action, payload):
        if (folder / "STOP").exists():
            raise Rejected("STOP latched before dispatch")
        before = state(log=True)
        validate_motion_against_start(payload, before["pose"])
        validate_grip_transition(WORKSPACE_ROOT, payload, before)
        validate_live_acquisition(WORKSPACE_ROOT, payload, before)
        if before["state"] == 5:
            raise Rejected("Controller configuration changed; inspect before a new task")
        if before["state"] not in IDLE_STATES:
            raise Rejected(f"Controller state {before['state']} is not an idle state")
        if before["error_warn"][1]:
            raise Rejected("Controller warning requires diagnosis; not auto-cleared")
        expected = payload.get("expected_pose")
        if expected is None or len(expected) != 6:
            raise Rejected("expected_pose is required")
        if np.linalg.norm(np.array(expected[:3])-np.array(before["pose"][:3])) > 2 or rotation_angle_deg(expected,before["pose"]) > 2:
            raise Rejected("Expected start pose changed")
        if action == "grip" and not all(0 <= int(payload.get(k, default)) <= 255 for k,default in (("speed",40),("force",10))):
            raise Rejected("Invalid gripper speed/force")
        if payload.get("grip_required"):
            required = require_existing_grasp(arm)
            trial.event("grasp_precheck", gripper=required)

        if action == "grip":
            position = int(payload["pos"])
            if not 0 <= position <= 255:
                raise Rejected("Gripper position must be 0..255")
            if before["gripper"] is None or before["gripper"]["gACT"] != 1 or before["gripper"]["gSTA"] != 3:
                raise Rejected("Gripper not activated; refusing position command")
            joints, target, cartesian = None, None, False
        else:
            target, translation, rotation = guard(payload, before)
            joints, _ = solve_joints(target, before["joint_deg"])
            delta = max(abs(np.array(joints) - np.array(before["joint_deg"])))
            if delta > min(float(payload.get("max_joint_deg", 90.0)), HARD_MAX_JOINT_DEG):
                raise Rejected(f"Joint step {delta:.1f} deg too large")
            if checked(arm.is_joint_limit(joints, is_radian=False), "is_joint_limit"):
                raise Rejected("Target joint angles hit the joint limit")
            cartesian = rotation < 0.5
            trial.event("step_plan", target=target.tolist(), joints=joints,
                        translation_mm=translation, rotation_deg=rotation, cartesian=cartesian)

        call("set_state", state=4)
        call("motion_enable", enable=True)
        call("set_mode", mode=0)
        call("set_state", state=0)
        try:
            if action == "grip":
                trial.event("gripper_command", pos=position, auto_enable=False,
                            speed=int(payload.get("speed", 40)), force=int(payload.get("force", 10)))
                call("robotiq_set_position", pos=position, speed=int(payload.get("speed", 40)),
                     force=int(payload.get("force", 10)), wait=False, check_baud=False, auto_enable=False)
            elif cartesian:
                speed = min(float(payload.get("speed", 15.0)), HARD_MAX_LINEAR_SPEED)
                call("set_position", x=float(target[0]), y=float(target[1]), z=float(target[2]),
                     roll=float(target[3]), pitch=float(target[4]), yaw=float(target[5]),
                     speed=speed, mvacc=min(2 * speed, 100.0), is_radian=False, wait=False)
            else:
                speed = min(float(payload.get("speed", 6.0)), HARD_MAX_JOINT_SPEED)
                call("set_servo_angle", angle=joints, speed=speed, mvacc=min(2 * speed, 20.0),
                     is_radian=False, wait=False)

            deadline = time.monotonic() + float(payload.get("timeout_s", 40.0))
            stable = 0
            time.sleep(0.25)
            while time.monotonic() < deadline:
                sample = state(log=True)
                if sample["state"] in (4, 5):
                    raise RuntimeError("External stop detected; aborting this step")
                if (folder / "STOP").exists():
                    raise RuntimeError("Workspace STOP requested")
                if payload.get("grip_required") and sample["gripper"] is None:
                    raise RuntimeError("Held-object feedback unavailable")
                if payload.get("grip_required") and sample["gripper"] is not None:
                    grip = sample["gripper"]
                    if grip["gFLT"] or grip["gOBJ"] not in (1, 2):
                        raise RuntimeError(f"Held-object status lost: {grip}")
                if action == "grip":
                    grip = sample["gripper"]
                    if grip is None:
                        raise RuntimeError("Gripper status unavailable while commanding it")
                    if grip["gFLT"]:
                        raise RuntimeError(f"Gripper fault {grip['gFLT']}")
                    reached = grip["gOBJ"] in (1, 2, 3)
                else:
                    reached = (np.linalg.norm(np.array(sample["pose"][:3]) - target[:3]) < 0.8
                               and max(abs(np.array(sample["joint_deg"]) - np.array(joints))) < 0.4)
                stable = stable + 1 if reached else 0
                if stable >= 3:
                    break
                time.sleep(0.12)
            else:
                raise TimeoutError("Step did not converge before timeout")
        finally:
            call("set_state", state=4)
        result = {"state": state(log=True)}
        if payload.get("capture", True):
            result["camera"] = capture("after_" + action)
        return result

    def command(payload):
        action = payload.get("action")
        trial.event("command", request=payload)
        if action == "status":
            return {"state": state(log=True)}
        if action == "observe":
            return {"state": state(log=True), "camera": capture(payload.get("label", "observe"))}
        if action == "gripper_check":
            grip, error = read_gripper()
            return {"gripper": grip, "gripper_error": error}
        if action == "gripper_activate":
            if payload.get("object_held") is not False:
                raise Rejected("Activation only allowed when an empty gripper is explicitly declared")
            precheck, precheck_error = read_gripper()
            trial.event("gripper_activate_precheck", gripper=precheck, error=precheck_error)
            if precheck_error or precheck is None or precheck.get("gOBJ") in (1,2) or precheck.get("gFLT"):
                raise Rejected("Cannot initialize with missing/faulted/possibly held-object feedback")
            call("robotiq_set_activate", wait=True, timeout=5, check_baud=False)
            return {"state": state(log=True), "activated": True}
        if action == "shutdown":
            return {"shutdown": True}
        if action in ("move", "grip"):
            return execute_guarded(WORKSPACE_ROOT, args.run_id, payload, lambda: run_step(action, payload))
        raise Rejected(f"Unknown action {action!r}")

    try:
        arm.connect()
        startup = {"axis": arm.axis, "version": arm.version, "connected": arm.connected,
                   "is_simulation": arm.is_simulation_robot,
                   "tcp_offset": list(arm.tcp_offset or []), "tcp_load": arm.tcp_load}
        trial.event("controller_info", **startup)
        if not arm.connected or arm.axis != 6 or arm.is_simulation_robot:
            raise RuntimeError(f"Expected a real xArm6, got {startup}")
        for reader in ("get_collision_sensitivity", "get_tcp_maxacc", "get_joint_maxacc"):
            try:
                trial.event("controller_setting", name=reader, value=getattr(arm, reader)())
            except Exception as exc:
                trial.event("controller_setting", name=reader, error=str(exc))
        opening = state(log=True)
        trial.event("camera_started", metadata=cam.open())
        print(json.dumps({"ready": str(folder), "controller": startup}), flush=True)
        capture("session_start")
        state(log=True)

        expiry = time.monotonic() + args.duration_s
        last_log = 0.0
        while time.monotonic() < expiry:
            if (folder / "STOP").exists():
                break
            now = time.monotonic()
            state(log=(now - last_log > 1.0))
            if now - last_log > 1.0:
                last_log = now
            pending = [f for f in sorted(inbox.glob("*.json")) if f.name not in handled]
            if pending:
                path = pending[0]
                handled.add(path.name)
                try:
                    try:
                        payload = json.loads(path.read_text(encoding="utf-8-sig"))
                    except Exception as exc:
                        raise Rejected(f"Malformed command file: {type(exc).__name__}: {exc}")
                    result = command(payload)
                    result["ok"] = True
                except Rejected as exc:
                    result = {"ok": False, "rejected": True, "error": f"{type(exc).__name__}: {exc}"}
                    trial.event("command_rejected", **result)
                except Exception as exc:
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    trial.event("error", **result)
                    fatal = True
                    try:
                        call("set_state", state=4)
                    except Exception:
                        pass
                write_result("result_" + path.name, result)
                print(json.dumps({"command": path.name, "ok": result.get("ok"),
                                  "error": result.get("error")}), flush=True)
                if fatal or result.get("shutdown"):
                    break
            time.sleep(0.2)
    except BaseException as exc:
        fatal = True
        trial.event("session_error", message=f"{type(exc).__name__}: {exc}")
        print(json.dumps({"session_error": f"{type(exc).__name__}: {exc}"}), flush=True)
    finally:
        if arm.connected:
            try:
                call("set_state", state=4)
            except Exception:
                pass
        for name, cleanup in (("camera",cam.close),("robot",arm.disconnect)):
            try:
                cleanup()
            except Exception as exc:
                fatal = True
                try:
                    trial.event("cleanup_error", component=name, error=str(exc))
                except Exception:
                    print(json.dumps({"cleanup_error":name,"error":str(exc)}),flush=True)
        if fatal or (folder / "STOP").exists():
            try:
                halt_run_tasks(WORKSPACE_ROOT, args.run_id, "Session fault or external stop; inspect and call human")
            except Exception as exc:
                trial.event("pipeline_halt_error", error=str(exc))
        trial.finish("failed" if fatal else "completed", None)


if __name__ == "__main__":
    main()
