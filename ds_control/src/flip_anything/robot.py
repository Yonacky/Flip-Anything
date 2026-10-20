import time


def checked(result, operation):
    code, value = result
    if code != 0:
        raise RuntimeError(f"{operation} failed with SDK code {code}")
    return value


class XArmObserver:
    """Read-only adapter: never enables motors, clears faults or activates gripper."""

    def __init__(self, cfg, api_factory=None):
        self.cfg = cfg
        self.api_factory = api_factory
        self.arm = None

    def open(self):
        if self.api_factory is None:
            from xarm.wrapper import XArmAPI
            self.api_factory = XArmAPI
        self.arm = self.api_factory(self.cfg["ip"], is_radian=False, do_not_open=True)
        try:
            self.arm.connect()
            if not self.arm.connected:
                raise ConnectionError("xArm connection did not become ready")
            if self.arm.axis != 6:
                raise RuntimeError(f"Expected xArm6, SDK reports {self.arm.axis} axes")
        except BaseException:
            self.close()
            raise
        return {"model": "xArm6", "ip": self.cfg["ip"], "axis": self.arm.axis,
                "firmware": self.arm.version, "tcp_offset_mm_deg": list(self.arm.tcp_offset),
                "tcp_load_reported": self.arm.tcp_load}

    def read(self):
        start = time.monotonic_ns()
        pose = checked(self.arm.get_position(is_radian=False), "get_position")
        joints = checked(self.arm.get_servo_angle(is_radian=False), "get_servo_angle")
        state = checked(self.arm.get_state(), "get_state")
        faults = checked(self.arm.get_err_warn_code(), "get_err_warn_code")
        result = {"sample_start_monotonic_ns": start, "pose_base_mm_deg": list(pose),
                  "joint_deg": list(joints)[:6], "state": state, "error_warn": faults,
                  "gripper": None, "wrench": None}
        if self.cfg["read_gripper"]:
            # check_baud=False avoids SDK's automatic baud-rate write during a read.
            raw = checked(self.arm.robotiq_get_status(check_baud=False), "robotiq_get_status")
            result["gripper"] = {"raw_response": list(raw), "status": dict(self.arm.robotiq_status)}
        result["sample_end_monotonic_ns"] = time.monotonic_ns()
        return result

    def close(self):
        if self.arm is not None:
            arm, self.arm = self.arm, None
            arm.disconnect()


class MockRobot:
    def __init__(self, cfg):
        pass

    def open(self):
        return {"model": "mock_xArm6", "axis": 6, "firmware": "synthetic"}

    def read(self):
        now = time.monotonic_ns()
        return {"sample_start_monotonic_ns": now, "sample_end_monotonic_ns": now,
                "pose_base_mm_deg": [300, 0, 300, 180, 0, 0], "joint_deg": [0] * 6,
                "state": 0, "error_warn": [0, 0], "gripper": None, "wrench": None}

    def close(self):
        pass
