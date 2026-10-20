"""Invert the held can about the tool Z axis in bounded steps (method from the
2026-10-18 case, re-derived from this round's own start pose and mount)."""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb
import goto


def rpy_of(R):
    return [math.degrees(math.atan2(R[2, 1], R[2, 2])),
            math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0])))),
            math.degrees(math.atan2(R[1, 0], R[0, 0]))]


def main():
    step_deg = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
    hold = (sys.argv[2] if len(sys.argv) > 2 else 'hold') == 'hold'
    run = tb.run_dir()
    ref = goto.last_result(run)
    pose = json.loads(ref.read_text(encoding='utf-8-sig'))['state']['pose']
    R0 = np.array(tb.pose_matrix(pose))[:3, :3]
    pos = np.array(pose[:3], dtype=float)
    n = int(round(180.0 / abs(step_deg)))
    R = R0
    for k in range(1, n + 1):
        a = math.radians(step_deg)
        Rz = np.array([[math.cos(a), -math.sin(a), 0.0],
                       [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
        R = R @ Rz
        tgt = [round(float(v), 4) for v in pos] + [round(float(v), 4) for v in rpy_of(R)]
        payload = {'action': 'move', 'pose': tgt, 'grip_required': hold, 'capture': True,
                   'max_translation_mm': 6.0, 'max_rotation_deg': max(30.0, abs(step_deg) + 12.0),
                   'max_joint_deg': 95.0, 'speed': 5.0, 'timeout_s': 60.0}
        cmd = tb.ROOT / 'scene' / f'flipz_{int(step_deg)}_{k}.json'
        cmd.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        res = goto.execute(cmd, ref)
        r = res.get('result', {})
        st = r.get('state') or {}
        g = st.get('gripper') or {}
        print(json.dumps({'step': k, 'of': n, 'target_rpy': tgt[3:], 'ok': r.get('ok'),
                          'error': r.get('error'),
                          'pose': [round(float(v), 1) for v in st.get('pose', [])] if st else None,
                          'gOBJ': g.get('gOBJ'), 'gPO': g.get('gPO'),
                          'frame': (r.get('camera') or {}).get('color_file')},
                         ensure_ascii=False), flush=True)
        if not r.get('ok'):
            return 1
        ref = Path(res['result_file'])
    print('flip complete')
    return 0


if __name__ == '__main__':
    sys.exit(main())
