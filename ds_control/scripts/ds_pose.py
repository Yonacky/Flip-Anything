"""Compute a wrist tilt target pose about a tool-frame axis. Read-only: prints numbers only."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_axis(axis, deg):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    a = math.radians(deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * (k @ k)


def euler_to_matrix(rpy):
    r, p, y = np.radians(rpy)
    return rot_z(y) @ rot_y(p) @ rot_x(r)


def matrix_to_euler(m):
    p = math.atan2(-m[2, 0], math.hypot(m[0, 0], m[1, 0]))
    y = math.atan2(m[1, 0], m[0, 0])
    r = math.atan2(m[2, 1], m[2, 2])
    return np.degrees([r, p, y])


def angle_between(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(a.T @ b) - 1.0) / 2.0))))


def main():
    pose = json.loads(sys.argv[1])
    axis = json.loads(sys.argv[2])
    ang = float(sys.argv[3])
    m = euler_to_matrix(np.array(pose[3:6]))
    new = m @ rot_axis(axis, ang)
    rpy = matrix_to_euler(new)
    err = np.abs(euler_to_matrix(rpy) - new).max()
    target = [round(float(v), 4) for v in list(pose[:3]) + list(rpy)]
    print("axis=%s angle=%+.1f deg" % (axis, ang))
    print("recon err %.2e  rotation step %.2f deg" % (err, angle_between(m, new)))
    print("tool z (base) %s" % np.round(new[:, 2], 4))
    print("target pose %s" % target)
    Path(ROOT / "scene" / "next_action.json").write_text(
        json.dumps({"action": "move", "pose": target, "grip_required": False, "capture": True,
                    "max_rotation_deg": 35.0, "max_translation_mm": 5.0,
                    "speed": 6.0, "timeout_s": 45.0}, ensure_ascii=False), encoding="utf8")
    print("wrote scene/next_action.json")


if __name__ == "__main__":
    main()
