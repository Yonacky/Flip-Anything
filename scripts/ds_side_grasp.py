"""侧向推进 + 轻闭合试探，找到罐子并完成低位侧抓。"""
import json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, 'src')
from flip_anything.geometry import pose_matrix

run = Path(sys.argv[1])
axis_xy = json.loads(sys.argv[2])      # [x, y] 罐轴
h_grasp = float(sys.argv[3])           # 抓取高度(基面 z)
stand = float(sys.argv[4])             # 起始站距
RPY = [180.0, 85.0, 180.0]
R = np.array(pose_matrix([0.0, 0.0, 0.0] + RPY))[:3, :3]
zd = R[:, 2]                            # 工具 Z(接近轴)方向
print('接近轴方向', np.round(zd, 4))
target_grasp = np.array([axis_xy[0], axis_xy[1], h_grasp])
start_tcp = target_grasp - stand*zd


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


def move_to(tcp, name, speed=25):
    tcp = np.array(tcp, float)
    for seg in range(6):
        pose = read_pose()
        cur = np.array(pose[:3], float)
        delta = tcp - cur
        dist = float(np.linalg.norm(delta))
        if dist < 0.5:
            return {'ok': True}
        step = tcp if dist <= 80.0 else cur + delta*(80.0/dist)
        r = send(f'{name}_{seg}', {'action': 'move', 'pose': [round(v, 4) for v in step] + RPY,
                                   'expected_pose': pose, 'speed': speed, 'max_translation_mm': 90,
                                   'max_rotation_deg': 70, 'capture': False, 'grip_required': False})
        if not r.get('ok'):
            return r
    return {'ok': False, 'error': 'too many segments'}


def grip(pos, name, force=10, speed=60, hold=False):
    pose = read_pose()
    r = send(name, {'action': 'grip', 'pos': pos, 'speed': speed, 'force': force,
                    'expected_pose': pose, 'capture': hold, 'grip_required': False})
    return r


# 1) 对准罐轴
r = move_to(start_tcp, 'align')
print('对准 -> ok=%s %s' % (r.get('ok'), r.get('error') or ''))
assert r.get('ok')

# 2) 逐步推进 + 轻闭合试探
detected_at = None
tcp = start_tcp.copy()
for k in range(1, 8):
    tcp = tcp + 18.0*zd
    r = move_to(tcp, f'adv{k}')
    if not r.get('ok'):
        print(f'推进 {k} 失败: {r.get("error")}')
        break
    g1 = grip(110, f'tc{k}', force=10, speed=60)
    st = g1['state']['gripper'] if g1.get('ok') else None
    if st is None:
        print(f'  第{k}步 闭合失败 {g1.get("error")}'); break
    det = st['gOBJ'] in (1, 2) or st['gPO'] < 105
    print(f'第{k}步 推进后 TCP={np.round(tcp,1)}  闭合 gPO={st["gPO"]:3d} gOBJ={st["gOBJ"]} gFLT={st["gFLT"]} '
          + ('检测到罐子 ✓' if det else '空'))
    grip(3, f'op{k}', force=10, speed=80)
    if det:
        detected_at = k
        break

if detected_at is None:
    print('未检测到罐子'); sys.exit(1)

# 3) 再多推进 10mm 以获得更深的夹持，然后正式夹紧
tcp = tcp + 8.0*zd
r = move_to(tcp, 'adv_final')
print('加深 -> ok=%s' % r.get('ok'))
g = grip(140, 'grasp', force=45, speed=40, hold=True)
st = g['state']['gripper']
print(f'夹紧: gPO={st["gPO"]} gOBJ={st["gOBJ"]} gFLT={st["gFLT"]} gGTO={st["gGTO"]} gACT={st["gACT"]}')
print('抓取点 TCP =', np.round(tcp, 2))
