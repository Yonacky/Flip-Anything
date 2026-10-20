"""生成剩余探针命令：起点位姿 + 规格串 "t:dx,dy,dz;rx:deg;ry:deg"。"""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'src')
from flip_anything.geometry import pose_matrix


def rpy_from_matrix(R):
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    return [math.degrees(math.atan2(R[2, 1], R[2, 2])), math.degrees(pitch),
            math.degrees(math.atan2(R[1, 0], R[0, 0]))]


def rot(axis, deg):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


run = Path(sys.argv[1]); pose = json.loads(sys.argv[2])
prefix = sys.argv[3]; specs = [s for s in sys.argv[4].split(';') if s]
for i, spec in enumerate(specs, start=1):
    kind, _, value = spec.partition(':')
    expected = list(pose)
    if kind == 't':
        T = np.array(pose_matrix(pose)); T[:3, 3] += np.array([float(v) for v in value.split(',')])
        pose = [T[0, 3], T[1, 3], T[2, 3]] + pose[3:]
    else:
        R = np.array(pose_matrix(pose))[:3, :3] @ rot(kind[1], float(value))
        pose = pose[:3] + rpy_from_matrix(R)
    cmd = {'action': 'move', 'pose': [round(v, 6) for v in pose],
           'expected_pose': [round(v, 6) for v in expected], 'speed': 12,
           'max_translation_mm': 60, 'max_rotation_deg': 60, 'capture': True, 'grip_required': False}
    name = f'{prefix}{i:02d}_{kind}{value.replace(",", "_")}'
    (run / 'commands' / f'{name}.json').write_text(json.dumps(cmd, ensure_ascii=False), encoding='utf-8')
    print(name, [round(v, 3) for v in pose])
