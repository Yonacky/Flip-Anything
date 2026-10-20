"""抬起 -> 绕工具Z轴翻转180°（全程校验持物状态）。"""
import json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, 'src')
from flip_anything.geometry import pose_matrix

run = Path(sys.argv[1]); TAG = time.strftime('%H%M%S')


def send(name, payload, timeout=120):
    name = f'{TAG}_{name}'
    tmp = run/'commands'/(name + '.json.tmp')
    tmp.write_text(json.dumps(payload), encoding='utf-8')
    tmp.replace(run/'commands'/(name + '.json'))
    res = run/('result_' + name + '.json')
    end = time.time() + timeout
    while time.time() < end:
        if res.exists():
            return json.loads(res.read_text(encoding='utf-8-sig'))
        time.sleep(0.2)
    raise RuntimeError('timeout ' + name)


def rpy(R):
    return [math.degrees(math.atan2(R[2, 1], R[2, 2])),
            math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0])))),
            math.degrees(math.atan2(R[1, 0], R[0, 0]))]


def quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr+1.0)*2; return np.array([0.25*S, (R[2,1]-R[1,2])/S, (R[0,2]-R[2,0])/S, (R[1,0]-R[0,1])/S])
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = math.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2; return np.array([(R[2,1]-R[1,2])/S, 0.25*S, (R[0,1]+R[1,0])/S, (R[0,2]+R[2,0])/S])
    if R[1,1] > R[2,2]:
        S = math.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2; return np.array([(R[0,2]-R[2,0])/S, (R[0,1]+R[1,0])/S, 0.25*S, (R[1,2]+R[2,1])/S])
    S = math.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2; return np.array([(R[1,0]-R[0,1])/S, (R[0,2]+R[2,0])/S, (R[1,2]+R[2,1])/S, 0.25*S])


def mat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def ang(A, B):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(A.T @ B)-1)/2))))


state = json.loads((run/'live_state.json').read_text(encoding='utf-8'))
pose = state['pose']
print('起始 pose', np.round(pose, 1), ' gOBJ', state['gripper']['gOBJ'])

# ---- 抬起 ----
for k in range(1, 6):
    cur = np.array(pose[:3], float)
    tgt = [cur[0], cur[1], min(250.0, cur[2] + 50.0)] + list(pose[3:])
    if abs(tgt[2] - cur[2]) < 0.3:
        break
    r = send(f'up{k}', {'action': 'move', 'pose': [round(v, 4) for v in tgt], 'expected_pose': pose,
                        'speed': 20, 'max_translation_mm': 60, 'max_rotation_deg': 70,
                        'capture': True, 'grip_required': True})
    print(f'抬起{k} ok={r.get("ok")} {r.get("error") or ""} pose={np.round(r["state"]["pose"],1) if r.get("ok") else None} '
          f'gOBJ={r["state"]["gripper"]["gOBJ"] if r.get("ok") else None}')
    if not r.get('ok'):
        sys.exit(1)
    pose = r['state']['pose']

# ---- 绕工具Z轴翻转180° ----
R0 = np.array(pose_matrix(pose))[:3, :3]
Rz180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)
Rt = R0 @ Rz180
step = 0
while True:
    cur = np.array(pose_matrix(pose))[:3, :3]
    a = ang(cur, Rt)
    if a < 1.5:
        print('翻转完成，剩余 %.2f deg' % a); break
    frac = min(1.0, 45.0/a)
    qa, qb = quat(cur), quat(Rt)
    if np.dot(qa, qb) < 0:
        qb = -qb
    q = qa*(1-frac) + qb*frac; q = q/np.linalg.norm(q)
    step += 1
    tgt = list(pose[:3]) + rpy(mat(q))
    r = send(f'flip{step}', {'action': 'move', 'pose': [round(v, 4) for v in tgt], 'expected_pose': pose,
                             'speed': 12, 'max_translation_mm': 60, 'max_rotation_deg': 70,
                             'capture': True, 'grip_required': True})
    ok = r.get('ok')
    print(f'翻转{step}: 剩余 {a:6.2f} -> ok={ok} {r.get("error") or ""} '
          f'gOBJ={r["state"]["gripper"]["gOBJ"] if ok else None}')
    if not ok:
        sys.exit(1)
    pose = r['state']['pose']
json.dump({'pose': pose}, open(run/f'after_flip_{TAG}.json', 'w'), indent=2)
print('翻转后 pose', np.round(pose, 2))
