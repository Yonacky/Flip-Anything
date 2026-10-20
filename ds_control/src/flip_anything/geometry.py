"""Explicit optical-camera -> TCP -> base transforms, millimetres and degrees."""
import json
import math
from pathlib import Path


def multiply(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def pose_matrix(pose):
    if len(pose) != 6 or not all(math.isfinite(x) for x in pose):
        raise ValueError("pose requires six finite x,y,z,roll,pitch,yaw values")
    x, y, z, roll, pitch, yaw = pose
    r, p, y_angle = map(math.radians, (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y_angle), math.sin(y_angle)
    # xArm fixed-axis roll/pitch/yaw: Rz(yaw) Ry(pitch) Rx(roll).
    return [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, x],
            [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, y],
            [-sp, cp*sr, cp*cr, z], [0, 0, 0, 1]]


def validate_transform(t):
    if not isinstance(t, list) or len(t) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in t):
        raise ValueError("Transform must be a 4x4 matrix")
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for row in t for x in row):
        raise ValueError("Transform contains non-finite or non-numeric values")
    if any(abs(t[3][j] - [0, 0, 0, 1][j]) > 1e-6 for j in range(4)):
        raise ValueError("Invalid homogeneous last row")
    for i in range(3):
        for j in range(3):
            if abs(sum(t[k][i]*t[k][j] for k in range(3)) - (1 if i == j else 0)) > 1e-5:
                raise ValueError("Rotation must be orthonormal")
    determinant = (t[0][0]*(t[1][1]*t[2][2]-t[1][2]*t[2][1])
                   -t[0][1]*(t[1][0]*t[2][2]-t[1][2]*t[2][0])
                   +t[0][2]*(t[1][0]*t[2][1]-t[1][1]*t[2][0]))
    if abs(determinant - 1) > 1e-5:
        raise ValueError("Rotation determinant must be +1")
    return t


def load_hand_eye(path, camera_serial):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("status") != "verified" or data.get("translation_unit") != "mm":
        raise ValueError("Hand-eye calibration must be verified, schema 1, millimetres")
    if not camera_serial or data.get("camera_serial") != camera_serial:
        raise ValueError("Calibration camera serial does not match")
    if not data.get("tcp_definition") or not data.get("verified_at"):
        raise ValueError("Calibration needs TCP definition and verification date")
    return validate_transform(data["T_tcp_camera"])


def camera_point_to_base(point_mm, pose_base_mm_deg, t_tcp_camera):
    if len(point_mm) != 3 or not all(math.isfinite(x) for x in point_mm):
        raise ValueError("Point must contain three finite values")
    transform = multiply(pose_matrix(pose_base_mm_deg), validate_transform(t_tcp_camera))
    point = list(point_mm) + [1]
    return [sum(transform[i][j] * point[j] for j in range(4)) for i in range(3)]
