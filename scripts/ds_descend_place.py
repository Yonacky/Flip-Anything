"""下降接近桌面（每步校验持物状态），停在计算接触位上方约15mm处等人工确认。"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, 'src')
from flip_anything.geometry import pose_matrix
run = Path(sys.argv[1]); stop_z = float(sys.argv[2]); TAG = time.strftime('%H%M%S')
RPY = [-180.0, -85.0, 0.0]
zd = np.array(pose_matrix([0.0, 0.0, 0.0] + RPY))[:3, 2]


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


pose = json.loads((run/'live_state.json').read_text(encoding='utf-8'))['pose']
print('起始 pose', np.round(pose, 1))
k = 0
while True:
    cur = np.array(pose[:3], float)
    if cur[2] - stop_z < 0.5:
        break
    k += 1
    nz = max(stop_z, cur[2] - 40.0)
    tgt = [cur[0], cur[1], nz] + list(pose[3:])
    r = send(f'dn{k}', {'action': 'move', 'pose': [round(v, 4) for v in tgt], 'expected_pose': pose,
                        'speed': 12, 'max_translation_mm': 60, 'max_rotation_deg': 70,
                        'capture': True, 'grip_required': True})
    if not r.get('ok'):
        print('下降%d 失败: %s' % (k, r.get('error'))); sys.exit(1)
    pose = r['state']['pose']
    st = r['state']['gripper']
    print('下降%d -> z=%.1f gOBJ=%d gPO=%d frame=%s' % (k, pose[2], st['gOBJ'], st['gPO'],
          r.get('camera', {}).get('color_file')))
json.dump({'pose': pose}, open(run/f'near_table_{TAG}.json', 'w'), indent=2)
print('停在 pose', np.round(pose, 2))
print('  相机高度(基座) = %.1f mm' % (pose[2] - 70.1))
print('  罐口距桌估计 = %.1f mm' % ((pose[2] + 3.84) - 99.0))
