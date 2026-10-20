"""稳健转姿态：反复以不超过 maxstep 的角度向目标姿态逼近。"""
import json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, 'src')
from flip_anything.geometry import pose_matrix

run = Path(sys.argv[1]); target = json.loads(sys.argv[2]); maxstep = float(sys.argv[3])
prefix = sys.argv[4] if len(sys.argv) > 4 else 'rot'
keep_pos = len(sys.argv) > 5 and sys.argv[5] == 'keep'


def rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr+1.0)*2; return np.array([0.25*S, (R[2,1]-R[1,2])/S, (R[0,2]-R[2,0])/S, (R[1,0]-R[0,1])/S])
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = math.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2; return np.array([(R[2,1]-R[1,2])/S, 0.25*S, (R[0,1]+R[1,0])/S, (R[0,2]+R[2,0])/S])
    if R[1,1] > R[2,2]:
        S = math.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2; return np.array([(R[0,2]-R[2,0])/S, (R[0,1]+R[1,0])/S, 0.25*S, (R[1,2]+R[2,1])/S])
    S = math.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2; return np.array([(R[1,0]-R[0,1])/S, (R[0,2]+R[2,0])/S, (R[1,2]+R[2,1])/S, 0.25*S])


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def rpy_of(R):
    return [math.degrees(math.atan2(R[2, 1], R[2, 2])),
            math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0])))),
            math.degrees(math.atan2(R[1, 0], R[0, 0]))]


def angle_between(Ra, Rb):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Ra.T @ Rb)-1)/2))))


def read_pose():
    for _ in range(60):
        try:
            d = json.loads((run/'live_state.json').read_text(encoding='utf-8'))
            if d.get('pose'):
                return d['pose']
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError('no live pose')


def send(name, payload, timeout=90):
    (run/'commands'/(name+'.json')).write_text(json.dumps(payload), encoding='utf-8')
    end = time.time()+timeout
    while time.time() < end:
        f = run/('result_'+name+'.json')
        if f.exists():
            return json.loads(f.read_text(encoding='utf-8-sig'))
        time.sleep(0.2)
    raise RuntimeError('timeout '+name)


Rt = np.array(pose_matrix(list(target)))[:3, :3]
step_i = 0
while True:
    pose = read_pose()
    Rc = np.array(pose_matrix(pose))[:3, :3]
    ang = angle_between(Rc, Rt)
    if ang < 1.5:
        print(f'到位，剩余 {ang:.2f} deg')
        break
    frac = min(1.0, maxstep/ang)
    qa = rot_to_quat(Rc); qb = rot_to_quat(Rt)
    if np.dot(qa, qb) < 0:
        qb = -qb
    q = qa*(1-frac) + qb*frac
    q = q/np.linalg.norm(q)
    Rn = quat_to_rot(q)
    pos = pose[:3] if keep_pos else list(target)[:3]
    tgt = [round(v, 5) for v in pos] + rpy_of(Rn)
    step_i += 1
    r = send(f'{prefix}{step_i}', {'action': 'move', 'pose': tgt, 'expected_pose': pose, 'speed': 15,
                                   'max_translation_mm': 90, 'max_rotation_deg': 70,
                                   'capture': False, 'grip_required': False})
    print(f'{prefix}{step_i}: 剩余 {ang:6.2f} -> ok={r.get("ok")} {r.get("error") or ""}')
    if not r.get('ok'):
        break
print('done')
