"""从正上方逐步下降 + 闭合试探，测定抓取中心相对 TCP 原点的偏置。"""
import json, sys, time
from pathlib import Path

run = Path(sys.argv[1])
x, y = float(sys.argv[2]), float(sys.argv[3])
z_start, z_stop, step = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
RPY = [180.0, 0.0, 0.0]


def read_pose(timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            d = json.loads((run/'live_state.json').read_text(encoding='utf-8'))
            if d.get('pose'):
                return d['pose']
        except Exception:
            pass
        time.sleep(0.15)
    raise RuntimeError('cannot read live pose')


def send(name, payload, timeout=90.0):
    (run/'commands'/(name + '.json')).write_text(json.dumps(payload), encoding='utf-8')
    end = time.time() + timeout
    while time.time() < end:
        f = run/('result_' + name + '.json')
        if f.exists():
            return json.loads(f.read_text(encoding='utf-8-sig'))
        time.sleep(0.2)
    raise RuntimeError('timeout ' + name)


z = z_start
while z >= z_stop - 1e-6:
    pose = read_pose()
    tgt = [x, y, round(z, 3)] + RPY
    if abs(pose[0]-x) > 0.05 or abs(pose[1]-y) > 0.05 or abs(pose[2]-z) > 0.05:
        r = send(f'desc_z{int(round(z))}', {'action': 'move', 'pose': tgt, 'expected_pose': pose,
                                            'speed': 30, 'max_translation_mm': 90, 'max_rotation_deg': 70,
                                            'capture': False, 'grip_required': False})
        if not r.get('ok'):
            print(f'z={z:7.1f} 移动失败: {r.get("error")}')
            break
    pose = read_pose()
    g1 = send(f'close_z{int(round(z))}', {'action': 'grip', 'pos': 255, 'speed': 60, 'force': 10,
                                          'expected_pose': pose, 'capture': False, 'grip_required': False})
    if not g1.get('ok'):
        print(f'z={z:7.1f} 闭合失败: {g1.get("error")}')
        break
    st = g1['state']['gripper']
    detected = st['gOBJ'] in (1, 2) or st['gPO'] < 200
    print(f'z={z:7.1f}  闭合后 gPO={st["gPO"]:3d} gOBJ={st["gOBJ"]} gFLT={st["gFLT"]}  -> '
          + ('检测到罐子 ✓' if detected else '空'))
    g2 = send(f'open_z{int(round(z))}', {'action': 'grip', 'pos': 3, 'speed': 80, 'force': 10,
                                         'expected_pose': read_pose(), 'capture': False, 'grip_required': False})
    if not g2.get('ok'):
        print(f'z={z:7.1f} 张开失败: {g2.get("error")}')
        break
    if detected:
        print(f'*** 首个检测高度 z={z:.1f} mm（基座系）')
        break
    z -= step
else:
    print('未检测到罐子，已到下限')
