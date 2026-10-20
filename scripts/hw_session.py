"""Resident hardware session: one process owns the arm and wrist camera.

Commands are JSON files dropped into <run>/commands; results and observations are
written back as files so the caller never needs piped stdio. No implicit motion:
every step names its expected start pose and is bounds-checked before it is sent.
"""
import hashlib, json, math, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flip_anything.camera import OrbbecCamera
from flip_anything.config import load_config
from flip_anything.geometry import pose_matrix
from flip_anything.robot import checked
from flip_anything.trial import Trial
from xarm.wrapper import XArmAPI


def stamp():
    return {"utc": datetime.now(timezone.utc).isoformat(), "host_monotonic_ns": time.monotonic_ns()}


class Session:
    def __init__(self, run_id, expiry_s, allow_motion):
        self.cfg = load_config(str(ROOT / "configs/platform.json"))
        self.allow_motion = allow_motion
        self.trial = Trial(self.cfg["recording"]["root"], self.cfg, "hardware", run_id, {
            "purpose": "autonomous_flip_session",
            "object_id": "green_can",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "allow_motion": allow_motion,
        })
        self.folder = self.trial.path
        self.inbox = self.folder / "commands"
        self.inbox.mkdir(exist_ok=True)
        self.arm = XArmAPI(self.cfg["robot"]["ip"], do_not_open=True, is_radian=False)
        self.cam = OrbbecCamera(self.cfg["camera"])
        self.index = 0
        self.done = set()
        self.moving_enabled = False
        self._gripper_ready = False
        self.expiry = time.monotonic() + expiry_s

    # ---- primitives -------------------------------------------------
    def call(self, method, **kw):
        self.trial.event("sdk_request", method=method, kwargs=kw)
        result = getattr(self.arm, method)(**kw)
        self.trial.event("sdk_result", method=method, result=result)
        code = result[0] if isinstance(result, tuple) else result
        if code != 0:
            raise RuntimeError(f"{method}: SDK code {code}")
        return result

    # ---- gripper: direct Modbus RTU over the tool RS485 port --------
    # The SDK's robotiq helpers write the baud rate on every read (an activation cache reset
    # risk while carrying) and fail with the tool port's default 30 ms Modbus timeout.
    # This client sets the link once, then talks to the gripper with explicit frames only.
    ROBOTIQ_BAUD = 115200
    ROBOTIQ_TIMEOUT_MS = 200

    def gripper_link(self):
        if self._gripper_ready:
            return
        self.trial.event("gripper_link_setup", baud=self.ROBOTIQ_BAUD, timeout_ms=self.ROBOTIQ_TIMEOUT_MS)
        self.call("set_rs485_timeout", timeout=self.ROBOTIQ_TIMEOUT_MS, target="robot")
        self.call("set_rs485_baudrate", baud=self.ROBOTIQ_BAUD, target="robot")
        time.sleep(0.3)
        self._gripper_ready = True

    @staticmethod
    def decode_gripper(raw):
        if not raw or len(raw) < 7:
            return None
        st = raw[3]
        return {"gOBJ": (st & 0xC0) >> 6, "gSTA": (st & 0x30) >> 4, "gGTO": (st & 0x08) >> 3,
                "gACT": st & 0x01, "kFLT": (raw[5] & 0xF0) >> 4 if len(raw) > 5 else None,
                "gFLT": raw[5] & 0x0F if len(raw) > 5 else None,
                "gPR": raw[6] if len(raw) > 6 else None,
                "gPO": raw[7] if len(raw) > 7 else None,
                "gCU": raw[8] if len(raw) > 8 else None}

    def gripper_status(self, retries=3):
        self.gripper_link()
        last = "no attempt"
        for _ in range(retries):
            code, raw = self.arm.set_rs485_data([0x09, 0x03, 0x07, 0xD0, 0x00, 0x03], min_res_len=7,
                                                target="robot", protocol="modbus_rtu")
            if code == 0:
                decoded = self.decode_gripper(list(raw))
                if decoded is not None:
                    self.trial.event("gripper_status", raw=list(raw), decoded=decoded)
                    return decoded
                last = f"short response {list(raw)}"
            else:
                last = f"code {code} raw {list(raw) if raw else raw}"
            time.sleep(0.2)
        raise RuntimeError(f"Gripper status read failed: {last}")

    def gripper_write(self, pos, speed=0x40, force=0x0A, require_activated=True):
        """Send one motion request. Never re-activates an already activated gripper."""
        if not 0 <= pos <= 0xFF:
            raise ValueError("Gripper position register must be 0..255")
        before = self.gripper_status()
        if require_activated and (before["gACT"] != 1 or before["gSTA"] != 3):
            raise RuntimeError(f"Gripper not activated (gACT={before['gACT']}, gSTA={before['gSTA']}); "
                               "refusing implicit activation")
        self.trial.event("gripper_write", pos=pos, speed=speed, force=force, before=before)
        code, raw = self.arm.set_rs485_data([0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06,
                                             0x09, 0x00, 0x00, pos, speed, force],
                                            min_res_len=8, target="robot", protocol="modbus_rtu")
        return {"code": code, "raw": list(raw) if raw else raw, "before": before}

    def gripper_activate(self):
        """Explicit activation, only ever called before the gripper holds anything."""
        self.gripper_link()
        before = self.gripper_status()
        if before["gACT"] == 1 and before["gSTA"] == 3:
            return {"already_active": True, "status": before}
        code, raw = self.arm.set_rs485_data([0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06,
                                             0x01, 0x00, 0x00, 0x00, 0x00, 0x00],
                                            min_res_len=8, target="robot", protocol="modbus_rtu")
        deadline = time.monotonic() + 8.0
        status = before
        while time.monotonic() < deadline:
            time.sleep(0.5)
            status = self.gripper_status()
            if status["gSTA"] == 3 and status["gACT"] == 1:
                break
        self.trial.event("gripper_activate", code=code, raw=list(raw) if raw else raw, status=status)
        if status["gSTA"] != 3 or status["gACT"] != 1:
            raise RuntimeError(f"Gripper activation did not complete: {status}")
        return {"code": code, "raw": list(raw) if raw else raw, "status": status}

    def state(self, log=False):
        st = checked(self.arm.get_state(), "get_state")
        faults = checked(self.arm.get_err_warn_code(), "get_err_warn_code")
        pose = list(checked(self.arm.get_position(is_radian=False), "get_position"))
        joints = list(checked(self.arm.get_servo_angle(is_radian=False), "get_servo_angle"))[:6]
        sample = {"state": st, "error_warn": faults, "pose": pose, "joint_deg": joints, "gripper": None,
                  **stamp()}
        try:
            sample["gripper"] = self.gripper_status()
        except Exception as exc:
            sample["gripper_error"] = f"{type(exc).__name__}: {exc}"
        if log:
            self.trial.event("robot_sample", **sample)
        (self.folder / "live_state.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")
        return sample

    def guard_faults(self, sample, require_clean=False):
        """Record fault flags. Observations proceed regardless; motion requires a cleared state."""
        error, warn = sample["error_warn"]
        if error or warn:
            self.trial.event("controller_fault_flag", error_code=error, warn_code=warn)
        if require_clean and (error or warn):
            raise RuntimeError(f"Controller not clean: error {error}, warning {warn}")

    def clear_faults(self, attempts=3):
        """C19 (end-module communication) is sticky and blocks motion; clear it before enabling."""
        for _ in range(attempts):
            s = self.state()
            error, warn = s["error_warn"]
            if not error and not warn:
                return s
            self.trial.event("clear_fault_attempt", error_code=error, warn_code=warn)
            try:
                self.call("clean_error")
                self.call("clean_warn")
            except Exception as exc:
                self.trial.event("clear_fault_error", message=str(exc))
            time.sleep(1.0)
        return self.state()

    def capture(self, label, warmup=8):
        for _ in range(warmup):
            if self.cam.pipeline.wait_for_frames(1500) is None:
                raise TimeoutError("Camera frame timeout during warmup")
        before = self.state(True)
        self.guard_faults(before)
        frame = self.cam.capture(self.folder, self.index)
        self.index += 1
        after = self.state(True)
        record = {"label": label, "pose_before": before["pose"], "pose_after": after["pose"],
                  "joints_before": before["joint_deg"], "state_before": before["state"],
                  "error_warn": before["error_warn"],
                  "gripper": before["gripper"], "camera": frame, "index": self.index - 1, **stamp()}
        self.trial.event("observation", **record)
        return record

    def ensure_motion_ready(self):
        s = self.state()
        if s["error_warn"][0] or s["error_warn"][1]:
            s = self.clear_faults()
        self.guard_faults(s, require_clean=True)
        if s["state"] not in (0, 2, 4):
            raise RuntimeError(f"Controller not in idle/stop state: {s['state']}")
        if self.moving_enabled:
            return s
        self.call("set_state", state=4)
        self.call("motion_enable", enable=True)
        self.call("set_mode", mode=0)
        self.call("set_state", state=0)
        self.moving_enabled = True
        return s



    def check_expected(self, request, sample):
        exp = request.get("expected_pose")
        if exp is None:
            raise ValueError("Command requires expected_pose")
        dpos = float(np.linalg.norm(np.asarray(sample["pose"][:3]) - np.asarray(exp[:3])))
        if dpos > request.get("pos_tol_mm", 2.0):
            raise RuntimeError(f"Start position differs from expected by {dpos:.2f} mm")
        r0 = np.asarray(pose_matrix(sample["pose"]))[:3, :3]
        r1 = np.asarray(pose_matrix(list(exp)))[:3, :3]
        angle = math.degrees(math.acos(float(np.clip((np.trace(r0.T @ r1) - 1) / 2, -1, 1))))
        if angle > request.get("rot_tol_deg", 1.0):
            raise RuntimeError(f"Start orientation differs from expected by {angle:.2f} deg")

    def settle(self, action, target_pose, joints=None, timeout=40.0):
        deadline = time.monotonic() + timeout
        stable = 0
        last = None
        while time.monotonic() < deadline:
            s = self.state(True)
            self.guard_faults(s)
            if s["state"] in (4, 5):
                raise RuntimeError("External stop detected; aborting step")
            if action == "grip":
                g = s.get("gripper") or {}
                if g.get("gFLT"):
                    raise RuntimeError(f"Gripper fault {g['gFLT']}")
                reached = g.get("gOBJ") in (1, 2, 3)
            else:
                dpos = float(np.linalg.norm(np.asarray(s["pose"][:3]) - np.asarray(target_pose[:3])))
                r0 = np.asarray(pose_matrix(s["pose"]))[:3, :3]
                r1 = np.asarray(pose_matrix(list(target_pose)))[:3, :3]
                angle = math.degrees(math.acos(float(np.clip((np.trace(r0.T @ r1) - 1) / 2, -1, 1))))
                joint_err = max(abs(np.asarray(s["joint_deg"]) - np.asarray(joints))) if joints is not None else 0.0
                reached = dpos < 0.8 and angle < 0.6 and joint_err < 0.4 and s["state"] != 1
            stable = stable + 1 if reached else 0
            last = s
            if stable >= 3:
                return s
            time.sleep(0.12)
        raise TimeoutError(f"Step timeout for {action}; last={None if last is None else last['pose']}")

    def wait_stopped(self, timeout=60.0, settle_s=0.6):
        """Wait for commanded motion to finish (SDK state returns to stop) without faking precision."""
        deadline = time.monotonic() + timeout
        quiet_since = None
        last = None
        while time.monotonic() < deadline:
            time.sleep(0.15)
            s = self.state(True)
            self.guard_faults(s)
            if s["state"] in (4, 5):
                raise RuntimeError("External stop detected; aborting step")
            last = s
            if s["state"] != 1:
                quiet_since = quiet_since or time.monotonic()
                if time.monotonic() - quiet_since >= settle_s:
                    return s
            else:
                quiet_since = None
        raise TimeoutError(f"Motion did not stop in {timeout:.0f}s; last pose {None if last is None else last['pose']}")

    def do_move(self, request, sample):
        target = np.asarray(request["pose"], dtype=float)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("pose must be six finite numbers")
        step = float(np.linalg.norm(target[:3] - np.asarray(sample["pose"][:3])))
        if step > request.get("max_step_mm", 160.0):
            raise RuntimeError(f"Translation step {step:.1f} mm exceeds bound")
        r0 = np.asarray(pose_matrix(sample["pose"]))[:3, :3]
        r1 = np.asarray(pose_matrix(target.tolist()))[:3, :3]
        angle = math.degrees(math.acos(float(np.clip((np.trace(r0.T @ r1) - 1) / 2, -1, 1))))
        if angle > request.get("max_step_deg", 95.0):
            raise RuntimeError(f"Rotation step {angle:.1f} deg exceeds bound")
        mode = request.get("mode", "linear")
        speed = float(request.get("speed", 15))
        targets = [target]
        # Optional extra waypoints, each independently bounded against the previous one.
        for extra in request.get("via", []):
            targets.append(np.asarray(extra, dtype=float))
        executed = []
        self.ensure_motion_ready()
        previous = np.asarray(sample["pose"], dtype=float)
        for index, waypoint in enumerate(targets):
            if not np.isfinite(waypoint).all() or waypoint.shape != (6,):
                raise ValueError("via waypoints must be six finite numbers")
            w_step = float(np.linalg.norm(waypoint[:3] - previous[:3]))
            if w_step > request.get("max_step_mm", 160.0):
                raise RuntimeError(f"Waypoint {index} translation {w_step:.1f} mm exceeds bound")
            w0 = np.asarray(pose_matrix(previous.tolist()))[:3, :3]
            w1 = np.asarray(pose_matrix(waypoint.tolist()))[:3, :3]
            w_angle = math.degrees(math.acos(float(np.clip((np.trace(w0.T @ w1) - 1) / 2, -1, 1))))
            if w_angle > request.get("max_step_deg", 95.0):
                raise RuntimeError(f"Waypoint {index} rotation {w_angle:.1f} deg exceeds bound")
            joints = None
            use_joint = mode == "joint" or w_angle > 0.05
            if use_joint:
                joints = list(checked(self.arm.get_inverse_kinematics(waypoint.tolist(), input_is_radian=False,
                                                                     return_is_radian=False), "IK"))[:6]
                base_joints = executed[-1]["joints"] if executed else list(sample["joint_deg"])
                for i in (0, 3, 5):
                    cands = [joints[i] + 360 * k for k in (-1, 0, 1) if -360 <= joints[i] + 360 * k <= 360]
                    if cands:
                        joints[i] = min(cands, key=lambda v: abs(v - base_joints[i]))
                jump = max(abs(np.asarray(joints) - np.asarray(base_joints)))
                if jump > request.get("max_joint_step_deg", 110.0):
                    raise RuntimeError(f"Waypoint {index} joint step {jump:.1f} deg exceeds bound")
                if checked(self.arm.is_joint_limit(joints, is_radian=False), "is_joint_limit"):
                    raise RuntimeError(f"Waypoint {index} joints out of limits")
            try:
                if use_joint:
                    self.call("set_servo_angle", angle=joints, speed=min(speed, 40), mvacc=min(2 * speed, 200),
                              is_radian=False, wait=False)
                else:
                    self.call("set_position", x=float(waypoint[0]), y=float(waypoint[1]), z=float(waypoint[2]),
                              roll=float(waypoint[3]), pitch=float(waypoint[4]), yaw=float(waypoint[5]),
                              speed=min(speed, 40), mvacc=min(2 * speed, 200), is_radian=False, wait=False)
                final = self.wait_stopped(timeout=request.get("timeout_s", 60.0),
                                          settle_s=request.get("settle_s", 0.6))
            finally:
                self.call("set_state", state=4)
            executed.append({"target": waypoint.tolist(), "joints": joints, "final_pose": final["pose"],
                             "final_joints": final["joint_deg"], "state": final["state"],
                             "gripper": final["gripper"]})
            previous = np.asarray(final["pose"], dtype=float)
        last = executed[-1]
        return {"steps": executed, "final_pose": last["final_pose"], "final_joints": last["final_joints"],
                "state": last["state"], "gripper": last["gripper"]}

    def do_grip(self, request, sample):
        pos = int(request["pos"])
        if not 0 <= pos <= 255:
            raise ValueError("Grip position register must be 0..255 (0 open, 255 closed)")
        g = sample.get("gripper") or {}
        if g.get("gACT") != 1 or g.get("gSTA") != 3:
            raise RuntimeError("Gripper not activated; refusing to auto-activate")
        self.ensure_motion_ready()
        previous = self.gripper_status()
        started = time.monotonic()
        wrote = self.gripper_write(pos, speed=int(request.get("speed", 0x40)), force=int(request.get("force", 0x0A)))
        target_tol = int(request.get("pos_tol", 8))
        deadline = time.monotonic() + float(request.get("timeout_s", 20.0))
        last = previous
        settled = 0
        while time.monotonic() < deadline:
            time.sleep(0.25)
            s = self.state(True)
            self.guard_faults(s)
            if s["state"] in (4, 5):
                raise RuntimeError("External stop detected; aborting grip")
            last = s.get("gripper") or last
            if last.get("gFLT"):
                raise RuntimeError(f"Gripper fault {last['gFLT']}")
            # gOBJ 0 means still moving; 1/2 mean contact stop, 3 means stopped at target
            contact = last.get("gOBJ") in (1, 2)
            at_target = last.get("gOBJ") == 3 and abs(int(last.get("gPO") or 0) - pos) <= target_tol
            settled = settled + 1 if (contact or at_target) else 0
            if settled >= 3:
                break
        else:
            raise TimeoutError(f"Grip did not settle: {last}")
        self.call("set_state", state=4)
        return {"requested": pos, "write": wrote, "initial": previous, "final": last,
                "elapsed_s": round(time.monotonic() - started, 2)}

    def do_gripper_activate(self, request, sample):
        if not self.allow_motion:
            raise RuntimeError("Activation not allowed in observation mode")
        self.ensure_motion_ready()
        try:
            result = self.gripper_activate()
        finally:
            self.call("set_state", state=4)
        return {"activation": result, "gripper": self.state()["gripper"]}

    def do_probe(self, request, sample):
        """Bounded joint-space probe: move named joints to given absolute angles, capture each pose.

        Used to establish observation geometry (where a rotation sends the camera's view) with
        small, individually bounded steps. Only joints listed in 'joints' are changed.
        """
        if not self.allow_motion:
            raise RuntimeError("Motion not allowed in observation mode")
        base = list(sample["joint_deg"])
        steps = request.get("steps") or []
        if not steps:
            raise ValueError("probe needs a non-empty 'steps' list")
        limit = float(request.get("max_joint_step_deg", 30.0))
        records = []
        for index, step in enumerate(steps):
            target_joints = list(base)
            for key, value in (step.get("joints") or {}).items():
                target_joints[int(key)] = float(value)
            jump = max(abs(np.asarray(target_joints) - np.asarray(base)))
            if jump > limit:
                raise RuntimeError(f"Probe step {index} joint change {jump:.1f} deg exceeds bound")
            if checked(self.arm.is_joint_limit(target_joints, is_radian=False), "is_joint_limit"):
                raise RuntimeError(f"Probe step {index} joints out of limits")
            self.trial.event("probe_step", index=index, joints=target_joints, label=step.get("label"))
            self.ensure_motion_ready()
            try:
                self.call("set_servo_angle", angle=target_joints, speed=min(float(request.get("speed", 8)), 30),
                          mvacc=20, is_radian=False, wait=False)
                stopped = self.wait_stopped(timeout=float(request.get("timeout_s", 40.0)),
                                            settle_s=float(request.get("settle_s", 0.8)))
            finally:
                self.call("set_state", state=4)
            shot = self.capture(step.get("label", f"probe_{index}")) if step.get("capture", True) else None
            records.append({"index": index, "commanded_joints": target_joints,
                            "reached_joints": stopped["joint_deg"], "pose": stopped["pose"],
                            "state": stopped["state"], "observation": shot})
            base = list(stopped["joint_deg"])
        return records

    def do_pose(self, request, sample):
        """Cartesian motion with the gripper holding pose known only as a rotation about the tool Z.

        The target is given as tool_rotation_deg (rotation about the current tool Z) plus an optional
        cartesian delta, so a fixed-TCP wrist rotation can be commanded without guessing tool/base frames.
        """
        if not self.allow_motion:
            raise RuntimeError("Motion not allowed in observation mode")
        start = np.asarray(sample["pose"], dtype=float)
        delta = np.asarray(request.get("delta_mm", [0.0, 0.0, 0.0]), dtype=float)
        rot_deg = float(request.get("tool_rotation_deg", 0.0))
        if abs(rot_deg) > float(request.get("max_tool_rotation_deg", 60.0)):
            raise RuntimeError(f"Tool rotation {rot_deg:.1f} deg exceeds bound")
        if float(np.linalg.norm(delta)) > float(request.get("max_step_mm", 60.0)):
            raise RuntimeError(f"Translation step {np.linalg.norm(delta):.1f} mm exceeds bound")
        r_start = np.asarray(pose_matrix(start.tolist()))[:3, :3]
        axis = r_start[:, 2] if request.get("axis", "tool_z") == "tool_z" else r_start[:, 0]
        theta = np.radians(rot_deg)
        kx, ky, kz = axis / np.linalg.norm(axis)
        skew = np.asarray([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
        r_delta = np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)
        target_rot = r_delta @ r_start
        rvec = np.asarray(cv2.Rodrigues(target_rot)[0]).reshape(3)
        target = start.copy()
        target[:3] = start[:3] + delta
        if not np.isfinite(target).all():
            raise ValueError("target pose is not finite")
        self.ensure_motion_ready()
        try:
            self.call("set_position_aa", axis_angle_pose=[float(v) for v in list(target[:3]) + list(rvec)],
                      speed=min(float(request.get("speed", 20)), 40), mvacc=60, is_radian=False, wait=False)
            stopped = self.wait_stopped(timeout=float(request.get("timeout_s", 60.0)),
                                        settle_s=float(request.get("settle_s", 0.8)))
        finally:
            self.call("set_state", state=4)
        shot = self.capture(request.get("label", "pose")) if request.get("capture", True) else None
        return {"commanded": {"start": start.tolist(), "delta_mm": delta.tolist(),
                              "tool_rotation_deg": rot_deg, "axis": request.get("axis", "tool_z"),
                              "rotation_vector": rvec.tolist()},
                "reached_pose": stopped["pose"], "reached_joints": stopped["joint_deg"],
                "state": stopped["state"], "observation": shot}

    # ---- main loop ---------------------------------------------------
    def run(self):
        arm, cam = self.arm, self.cam
        try:
            arm.connect()
            if not arm.connected or arm.axis != 6 or arm.is_simulation_robot:
                raise RuntimeError("Expected a real xArm6")
            self.trial.event("robot_connected", metadata={
                "axis": arm.axis, "firmware": arm.version, "sn": arm.sn,
                "tcp_offset_mm_deg": list(arm.tcp_offset), "tcp_load": arm.tcp_load,
                "is_simulation": bool(arm.is_simulation_robot)})
            self.trial.event("camera_started", metadata=cam.open())
            self.clear_faults()
            self.gripper_link()
            self.state(True)
            self.capture("session_start")
            print(json.dumps({"ready": str(self.folder), "allow_motion": self.allow_motion}), flush=True)
            while not self.stop and time.monotonic() < self.expiry:
                pending = [f for f in sorted(self.inbox.glob("*.json")) if f.name not in self.done]
                if pending:
                    f = pending[0]
                    self.done.add(f.name)
                    self.process(f)
                else:
                    self.state(log=False)
                    time.sleep(0.2)
            self.trial.event("session_window_elapsed")
            return "completed"
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.trial.event("error", message=self.error, traceback=traceback.format_exc())
            return "failed"
        finally:
            try:
                if arm.connected:
                    self.call("set_state", state=4)
            except Exception:
                pass
            try:
                cam.close()
            except Exception:
                pass
            try:
                arm.disconnect()
            except Exception:
                pass
            self.trial.finish("completed" if self.error is None else "failed", self.error)

    error = None
    stop = False

    def process(self, path):
        name = path.stem
        try:
            request = json.loads(path.read_text(encoding="utf-8-sig"))
            self.trial.event("command", name=name, request=request)
            before = self.state(True)
            self.guard_faults(before)
            action = request.get("action")
            if action == "inspect":
                result = {"state": before, "observation": self.capture(request.get("label", "inspect"))}
            elif action == "state":
                result = {"state": before}
            elif action == "probe":
                result = {"state": before, "probes": self.do_probe(request, before)}
            elif action == "pose":
                result = {"state": before, "motion": self.do_pose(request, before)}
            elif action == "shutdown":
                result = {"shutdown": True}
            elif action in ("move", "grip"):
                if not self.allow_motion:
                    raise RuntimeError("Motion not allowed in observation mode")
                self.check_expected(request, before)
                result = self.do_move(request, before) if action == "move" else self.do_grip(request, before)
                result = {"before": before, "after": result, "observation": self.capture("after_" + name)}
            elif action == "gripper_activate":
                result = self.do_gripper_activate(request, before)
                result = {"before": before, "after": result}
            else:
                raise ValueError(f"Unknown action {action!r}")
            payload = {"ok": True, **result}
        except Exception as exc:
            payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()}
            self.trial.event("error", command=name, **payload)
            try:
                self.call("set_state", state=4)
            except Exception:
                pass
        (self.folder / f"result_{name}.json").write_text(json.dumps(payload, indent=2, default=str),
                                                         encoding="utf-8")
        if payload.get("shutdown"):
            self.stop = True


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--expiry-s", type=float, default=3600)
    p.add_argument("--allow-motion", action="store_true")
    a = p.parse_args()
    s = Session(a.run_id, a.expiry_s, a.allow_motion)
    status = s.run()
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
