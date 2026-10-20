"""唯一命名的分步前进 + 轻夹检测（避免与旧结果文件重名）。"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, 'src')
from flip_anything.geometry import pose_matrix

run = Path(sys.argv[1]); axis_xy = json.loads(sys.argv[2]); h_grasp = float(sys.argv[3])
RPY = [180.0, 85.0, 180.0]
zd = np.array(pose_matrix([0.0, 0.0, 0.0] + RPY))[:3, 2]
axis = np.array([axis_xy[0], axis_xy[1], h_grasp], float)
TAG = time.strftime('%H%M%S')


def send(name, payload, timeout=90):
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
print('起始 TCP', np.round(pose[:3], 1), ' 沿接近轴到罐轴距离', round(float((axis-np.array(pose[:3])) @ zd), 1), 'mm')
g = send('open', {'action': 'grip', 'pos': 3, 'force': 10, 'speed': 80, 'expected_pose': pose,
                  'capture': False, 'grip_required': False})
print('张开 ok=%s' % g.get('ok'))
pose = g['state']['pose']
found = None
for k in range(1, 12):
    tgt = np.array(pose[:3], float) + 18.0*zd
    r = send(f'fwd{k}', {'action': 'move', 'pose': [round(v, 4) for v in tgt] + RPY,
                         'expected_pose': pose, 'speed': 15, 'max_translation_mm': 40,
                         'max_rotation_deg': 70, 'capture': True, 'grip_required': False})
    if not r.get('ok'):
        print('前进 %d 失败: %s' % (k, r.get('error'))); break
    pose = r['state']['pose']
    dist = float((axis - np.array(pose[:3], float)) @ zd)
    c = send(f'chk{k}', {'action': 'grip', 'pos': 120, 'force': 20, 'speed': 30,
                         'expected_pose': pose, 'capture': True, 'grip_required': False})
    if not c.get('ok'):
        print('  第%d步 闭合失败 %s' % (k, c.get('error'))); break
    st = c['state']['gripper']
    det = (st['gOBJ'] in (1, 2)) or st['gPO'] < 115
    print('第%d步 TCP=%s 距罐轴 %5.1fmm gPO=%d gOBJ=%d gap=%.1fmm %s' % (
        k, np.round(pose[:3], 1), dist, st['gPO'], st['gOBJ'], (227-st['gPO'])*0.381,
        'DETECTED' if det else 'empty'))
    if det:
        found = k; break
    o = send(f'op{k}', {'action': 'grip', 'pos': 3, 'force': 10, 'speed': 80,
                        'expected_pose': pose, 'capture': False, 'grip_required': False})
    pose = o['state']['pose']

if found is None:
    print('未检测到罐身'); sys.exit(1)
tgt = np.array(pose[:3], float) + 10.0*zd
r = send('deeper', {'action': 'move', 'pose': [round(v, 4) for v in tgt] + RPY, 'expected_pose': pose,
                    'speed': 10, 'max_translation_mm': 40, 'max_rotation_deg': 70,
                    'capture': True, 'grip_required': False})
if r.get('ok'):
    pose = r['state']['pose']
    print('加深到', np.round(pose[:3], 1))
gr = send('final', {'action': 'grip', 'pos': 120, 'force': 45, 'speed': 25,
                    'expected_pose': pose, 'capture': True, 'grip_required': False})
st = gr['state']['gripper']
print('正式夹紧: gPO=%d gOBJ=%d gFLT=%d gap=%.1fmm' % (st['gPO'], st['gOBJ'], st['gFLT'], (227-st['gPO'])*0.381))
json.dump({'tcp': pose[:3], 'gripper': st}, open(run/f'grasp_{TAG}.json', 'w'), indent=2)
