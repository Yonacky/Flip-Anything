"""重定位阶段规划：抬高 -> 转成工具Z朝下 -> 横移到罐轴上方。"""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'src')
from flip_anything.geometry import pose_matrix


def rpy_from_matrix(R):
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    return [math.degrees(math.atan2(R[2, 1], R[2, 2])), math.degrees(pitch),
            math.degrees(math.atan2(R[1, 0], R[0, 0]))]


def slerp(Ra, Rb, t):
    qa = rot_to_quat(Ra); qb = rot_to_quat(Rb)
    if np.dot(qa, qb) < 0:
        qb = -qb
    q = qa*(1-t) + qb*t
    q = q/np.linalg.norm(q)
    return quat_to_rot(q)


def rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr+1.0)*2; q = [0.25*S, (R[2,1]-R[1,2])/S, (R[0,2]-R[2,0])/S, (R[1,0]-R[0,1])/S]
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = math.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2; q = [(R[2,1]-R[1,2])/S, 0.25*S, (R[0,1]+R[1,0])/S, (R[0,2]+R[2,0])/S]
    elif R[1,1] > R[2,2]:
        S = math.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2; q = [(R[0,2]-R[2,0])/S, (R[0,1]+R[1,0])/S, 0.25*S, (R[1,2]+R[2,1])/S]
    else:
        S = math.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2; q = [(R[1,0]-R[0,1])/S, (R[0,2]+R[2,0])/S, (R[1,2]+R[2,1])/S, 0.25*S]
    return np.array(q)


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


run = Path(sys.argv[1]); start = json.loads(sys.argv[2]); target_pos = json.loads(sys.argv[3])
name = sys.argv[4]
T0 = np.array(pose_matrix(start)); R0 = T0[:3, :3]; p0 = T0[:3, 3]
Rt = np.array(pose_matrix(list(target_pos)))[:3, :3]
pt = np.array(list(target_pos)[:3], float)
cmds = []
pose = list(start)
# 1) 先抬高
for i in range(1, 3):
    nxt = p0.copy(); nxt[2] = p0[2] + (pt[2]-p0[2])*i/2.0
    cmds.append(('up%d' % i, list(nxt) + pose[3:]))
    pose = list(nxt) + pose[3:]
# 2) 再转成朝下（slerp 两步）
Rc0 = np.array(pose_matrix(pose))[:3, :3]
for i in (1, 2, 3):
    Ri = slerp(R0, Rt, i/3.0)
    cmds.append(('rot%d' % i, list(pt * 0 + np.array(pose[:3])) + rpy_from_matrix(Ri)))
    pose = list(pose[:3]) + rpy_from_matrix(Ri)
# 3) 横移到罐轴上方（保持新姿态）
cur = np.array(pose[:3], float)
for i in (1, 2, 3):
    nxt = cur + (pt - cur) * i / 3.0
    cmds.append(('lat%d' % i, list(nxt) + pose[3:]))
    pose = list(nxt) + pose[3:]

prev = list(start)
for tag, tgt in cmds:
    cmd = {'action': 'move', 'pose': [round(v, 5) for v in tgt],
           'expected_pose': [round(v, 5) for v in prev], 'speed': 40,
           'max_translation_mm': 90, 'max_rotation_deg': 70, 'capture': True, 'grip_required': False}
    fn = f'{name}_{tag}'
    (run/'commands'/(fn + '.json')).write_text(json.dumps(cmd), encoding='utf-8')
    print(fn, [round(v, 2) for v in tgt])
    prev = tgt
