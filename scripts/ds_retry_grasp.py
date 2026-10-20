"""张开 -> 走到罐轴 -> 夹紧（每步留图）"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, 'src')
from flip_anything.geometry import pose_matrix
run = Path(sys.argv[1]); target = np.array(json.loads(sys.argv[2]), float)
RPY = [180.0, 85.0, 180.0]


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
    tmp = run/'commands'/(name+'.json.tmp')
    tmp.write_text(json.dumps(payload), encoding='utf-8')
    tmp.replace(run/'commands'/(name+'.json'))
    end = time.time()+timeout
    while time.time() < end:
        f = run/('result_'+name+'.json')
        if f.exists():
            return json.loads(f.read_text(encoding='utf-8-sig'))
        time.sleep(0.2)
    raise RuntimeError('timeout '+name)


# 1) 确保张开
pose = read_pose()
g = send('g_open', {'action': 'grip', 'pos': 3, 'force': 10, 'speed': 80,
                    'expected_pose': pose, 'capture': True, 'grip_required': False})
print('张开 ok=%s gPO=%s' % (g.get('ok'), g['state']['gripper']['gPO'] if g.get('ok') else None))
# 2) 分段走到罐轴
pose = read_pose()
for seg in range(8):
    cur = np.array(pose[:3], float); delta = target - cur
    dist = float(np.linalg.norm(delta))
    if dist < 0.4:
        break
    step = target if dist <= 50.0 else cur + delta*(50.0/dist)
    r = send(f'go_{seg}', {'action': 'move', 'pose': [round(v, 4) for v in step] + RPY,
                           'expected_pose': pose, 'speed': 18, 'max_translation_mm': 60,
                           'max_rotation_deg': 70, 'capture': True, 'grip_required': False})
    print(f'  移动 {seg}: -> {np.round(step,1)} ok={r.get("ok")} {r.get("error") or ""}')
    if not r.get('ok'):
        sys.exit(1)
    pose = r['state']['pose']
print('到位 TCP =', np.round(pose[:3], 2), ' 夹爪 gPO =', json.loads((run/'live_state.json').read_text(encoding='utf-8'))['gripper']['gPO'])
# 3) 夹紧
c = send('g_close', {'action': 'grip', 'pos': 140, 'force': 45, 'speed': 40,
                     'expected_pose': pose, 'capture': True, 'grip_required': False})
st = c['state']['gripper'] if c.get('ok') else None
print('夹紧 ok=%s %s' % (c.get('ok'), c.get('error') or ''))
if st:
    print('  gPO=%d gOBJ=%d gFLT=%d gap=%.1f mm' % (st['gPO'], st['gOBJ'], st['gFLT'], (227-st['gPO'])*0.381))
    print('  判定:', 'GOT OBJECT' if (st['gOBJ'] in (1, 2) or st['gPO'] < 130) else 'EMPTY')
