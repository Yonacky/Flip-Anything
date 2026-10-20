"""侧抓（无需试探）：把 TCP 直接走到罐轴上，使罐轴落在两指中间，然后夹紧。"""
import json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, 'src')
from flip_anything.geometry import pose_matrix

run = Path(sys.argv[1])
axis = json.loads(sys.argv[2])          # [x, y]
h_grasp = float(sys.argv[3])
RPY = [180.0, 85.0, 180.0]
zd = np.array(pose_matrix([0.0, 0.0, 0.0] + RPY))[:3, 2]
target = np.array([axis[0], axis[1], h_grasp], float)
print('接近轴', np.round(zd, 4), ' 目标 TCP(罐轴处)', np.round(target, 2))


def read_pose():
    for _ in range(80):
        try:
            d = json.loads((run/'live_state.json').read_text(encoding='utf-8'))
            if d.get('pose'):
                return d['pose']
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError('no pose')


def send(name, payload, timeout=90):
    (run/'commands'/(name+'.json')).write_text(json.dumps(payload), encoding='utf-8')
    end = time.time()+timeout
    while time.time() < end:
        f = run/('result_'+name+'.json')
        if f.exists():
            return json.loads(f.read_text(encoding='utf-8-sig'))
        time.sleep(0.2)
    raise RuntimeError('timeout '+name)


def move_to(tcp, name, speed=20):
    tcp = np.array(tcp, float)
    for seg in range(8):
        pose = read_pose()
        cur = np.array(pose[:3], float)
        delta = tcp - cur
        dist = float(np.linalg.norm(delta))
        if dist < 0.4:
            return {'ok': True}
        step = tcp if dist <= 60.0 else cur + delta*(60.0/dist)
        r = send(f'{name}_{seg}', {'action': 'move', 'pose': [round(v, 4) for v in step] + RPY,
                                   'expected_pose': pose, 'speed': speed, 'max_translation_mm': 70,
                                   'max_rotation_deg': 70, 'capture': True, 'grip_required': False})
        if not r.get('ok'):
            return r
    return {'ok': False, 'error': 'segments'}


# 先到罐轴前 60mm 的过渡点，再精确进入
pre = target - 60.0*zd
r = move_to(pre, 'approach1')
print('过渡点 -> ok=%s %s' % (r.get('ok'), r.get('error') or ''))
if not r.get('ok'):
    sys.exit(1)
r = move_to(target, 'approach2', speed=10)
print('进入抓取位 -> ok=%s %s' % (r.get('ok'), r.get('error') or ''))
if not r.get('ok'):
    sys.exit(1)
pose = read_pose()
print('到位 TCP =', np.round(pose[:3], 2))

g = send('close_grasp', {'action': 'grip', 'pos': 140, 'force': 45, 'speed': 40,
                         'expected_pose': pose, 'capture': True, 'grip_required': False})
st = g['state']['gripper'] if g.get('ok') else None
print('夹紧结果 ok=%s %s' % (g.get('ok'), g.get('error') or ''))
if st:
    print(f'  gPO={st["gPO"]} gOBJ={st["gOBJ"]} gFLT={st["gFLT"]} gGTO={st["gGTO"]} gACT={st["gACT"]}')
    width = (227 - st['gPO']) * 0.381
    print(f'  两指开口估计 ≈ {width:.1f} mm (罐径约 48-53 mm)')
json.dump({'tcp': list(pose[:3]), 'gripper': st}, open(run/'grasp_record.json', 'w'), indent=2)
