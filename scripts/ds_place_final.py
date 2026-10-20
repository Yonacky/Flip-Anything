"""最终落放：降到留约3mm余量 -> 保持不动张爪 -> 沿接近轴退回。"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, 'src')
from flip_anything.geometry import pose_matrix
run = Path(sys.argv[1]); place_z = float(sys.argv[2]); TAG = time.strftime('%H%M%S')
zd = np.array(pose_matrix([0.0, 0.0, 0.0, -180.0, -85.0, 0.0]))[:3, 2]


def send(name, payload, timeout=120, hold=False):
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


pose = json.loads((run/'live_state.json').read_text(encoding='utf-8'))['pose']
print('起始 pose', np.round(pose, 1))
# 分两小步下降到 place_z
k = 0
while np.array(pose[:3], float)[2] - place_z > 0.4:
    k += 1
    cur = np.array(pose[:3], float)
    nz = max(place_z, cur[2] - 6.0)
    tgt = [cur[0], cur[1], nz] + list(pose[3:])
    r = send(f'last{k}', {'action': 'move', 'pose': [round(v, 4) for v in tgt], 'expected_pose': pose,
                          'speed': 6, 'max_translation_mm': 30, 'max_rotation_deg': 70,
                          'capture': True, 'grip_required': True})
    if not r.get('ok'):
        print('下降失败', r.get('error')); sys.exit(1)
    pose = r['state']['pose']
    print('  降 -> z=%.2f gOBJ=%d' % (pose[2], r['state']['gripper']['gOBJ']))
# 保持不动，张爪
r = send('release', {'action': 'grip', 'pos': 3, 'speed': 30, 'force': 10,
                     'expected_pose': pose, 'capture': True, 'grip_required': False})
print('张爪 ok=%s gPO=%s gOBJ=%s' % (r.get('ok'), r['state']['gripper']['gPO'] if r.get('ok') else None,
                                     r['state']['gripper']['gOBJ'] if r.get('ok') else None))
pose = r['state']['pose']
time.sleep(1.5)
# 沿接近轴退回（不扫碰罐子）
for k in range(1, 4):
    cur = np.array(pose[:3], float) - 50.0*zd
    r = send(f'ret{k}', {'action': 'move', 'pose': [round(v, 4) for v in cur] + list(pose[3:]),
                         'expected_pose': pose, 'speed': 15, 'max_translation_mm': 60,
                         'max_rotation_deg': 70, 'capture': True, 'grip_required': False})
    if not r.get('ok'):
        print('退回失败', r.get('error')); break
    pose = r['state']['pose']
    print('  退回%d -> %s' % (k, np.round(pose[:3], 1)))
json.dump({'pose': pose}, open(run/f'placed_{TAG}.json', 'w'), indent=2)
