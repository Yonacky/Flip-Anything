"""Grasp checks must never actuate, reset or activate a loaded gripper."""
from .robot import checked


def require_existing_grasp(arm):
    checked(arm.robotiq_get_status(check_baud=False), "robotiq_get_status")
    status = dict(arm.robotiq_status)
    if status["gACT"] != 1 or status["gSTA"] != 3:
        raise RuntimeError("Gripper is not activated; no automatic activation while carrying")
    if status["gFLT"] != 0:
        raise RuntimeError(f"Gripper fault {status['gFLT']}; grasp check does not send corrective commands")
    if status["gGTO"] != 1 or status["gOBJ"] not in (1, 2):
        raise RuntimeError("No detected held object; no motion allowed")
    return status
