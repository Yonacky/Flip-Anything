"""Recover clearance: clear C31, lower collision sensitivity, then climb in small bounded +Z steps."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from flip_anything.config import load_config
from xarm.wrapper import XArmAPI

cfg = load_config(Path(__file__).resolve().parents[2] / "configs/platform.json")
arm = XArmAPI(cfg["robot"]["ip"], is_radian=False, do_not_open=True)
arm.connect()
log = []
def snap(tag):
    pose = list(arm.get_position(is_radian=False)[1])
    err = arm.get_err_warn_code()[1]
    st = arm.get_state()[1]
    log.append({"tag": tag, "pose": pose, "err_warn": err, "state": st})
    return pose, err, st

snap("connect")
for name, value in (("clean_error", None), ("clean_warn", None)):
    getattr(arm, name)()
time.sleep(0.8)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(0)
time.sleep(0.8)
snap("enabled")
for tag, fn in (("collision_sensitivity_1", lambda: arm.set_collision_sensitivity(1)),
                ("tcp_maxacc_200", lambda: arm.set_tcp_maxacc(200)),
                ("joint_maxacc_20", lambda: arm.set_joint_maxacc(20))):
    try:
        log.append({"tag": tag, "result": fn()})
    except Exception as exc:
        log.append({"tag": tag, "error": f"{type(exc).__name__}: {exc}"})

pose, err, st = snap("before_lift")
if err[0] or err[1]:
    arm.clean_error(); arm.clean_warn(); time.sleep(0.8)
    arm.motion_enable(enable=True); arm.set_mode(0); arm.set_state(0); time.sleep(0.5)
    pose, err, st = snap("after_reclean")
start = list(pose)
for step in range(1, 21):
    target = list(start)
    target[2] = start[2] + 20.0 * step
    code = arm.set_position(x=target[0], y=target[1], z=target[2], roll=target[3], pitch=target[4],
                            yaw=target[5], speed=15, mvacc=30, wait=True, timeout=25)
    pose_now, err, st = snap(f"lift_{step}")
    log.append({"tag": f"lift_{step}_code", "code": code})
    if err[0] or err[1] or st in (4, 5):
        log.append({"tag": "abort", "reason": f"err={err} state={st}"})
        break
arm.set_state(4)
snap("final")
arm.disconnect()
print(json.dumps(log, ensure_ascii=False, indent=2, default=str))
