"""Bounded multi-step move helper that keeps using the single session."""

import json, subprocess, sys

from pathlib import Path

import numpy as np



ROOT = Path(r'G:\Flip Anything\ds_control')

sys.path.insert(0, str(ROOT / 'src'))

from flip_anything.geometry import pose_matrix

PY = Path(r'G:\Flip Anything\.venv\Scripts\python.exe')





def run_dir():

    return ROOT.parent / 'logs' / 'experiments' / 'ds_control' / json.loads((ROOT / 'active.json').read_text())['run']





def results(run):

    return [p for p in sorted(run.glob('result_*.json'), key=lambda q: q.stat().st_mtime)

            if 'state' in json.loads(p.read_text(encoding='utf-8-sig'))]





def ang(a, b):

    Ra = np.array(pose_matrix(list(a)))[:3, :3]

    Rb = np.array(pose_matrix(list(b)))[:3, :3]

    return float(np.degrees(np.arccos(max(-1.0, min(1.0, (np.trace(Ra.T @ Rb) - 1) / 2)))))





def execute(cmd, ref):

    out = subprocess.run([str(PY), '-X', 'utf8', 'scripts/control.py', 'execute', str(cmd),

                          '--from-result', str(ref)], cwd=str(ROOT), capture_output=True, text=True,

                         encoding='utf-8', errors='replace')

    for line in reversed(out.stdout.strip().splitlines()):

        if line.strip().startswith('{'):

            return json.loads(line.strip())

    raise RuntimeError(out.stdout[-1500:] + out.stderr[-1500:])





def main():

    target = [float(v) for v in sys.argv[1:7]]

    label = sys.argv[7]

    grip_required = len(sys.argv) > 8 and sys.argv[8].lower() == 'true'

    speed = float(sys.argv[9]) if len(sys.argv) > 9 else 10.0

    run = run_dir()

    ref = results(run)[-1]

    for seg in range(6):

        cur = json.loads(ref.read_text(encoding='utf-8-sig'))['state']['pose']

        d = np.linalg.norm(np.array(cur[:3]) - np.array(target[:3]))

        a = ang(cur, target)

        if d < 0.4 and a < 0.4:

            print(json.dumps({'done': True, 'pose': cur}))

            return 0

        frac = min(1.0, 55.0 / max(d, 1e-6), 55.0 / max(a, 1e-6))

        # keep the commanded orientation identical (wrapped to the current representation)

        rpy = list(target[3:])

        for j, i in enumerate((3, 4, 5)):

            while rpy[j] - cur[i] > 180.0:

                rpy[j] -= 360.0

            while rpy[j] - cur[i] < -180.0:

                rpy[j] += 360.0

        step = [cur[i] + (target[i] - cur[i]) * frac for i in range(3)] + rpy

        payload = {'action': 'move', 'pose': [round(v, 4) for v in step],

                   'grip_required': bool(grip_required), 'capture': True,

                   'max_translation_mm': 75.0, 'max_rotation_deg': 70.0, 'max_joint_deg': 95.0,

                   'speed': speed, 'timeout_s': 60.0}

        cmd = ROOT / 'scene' / ('mv_%s_%d.json' % (label, seg))

        cmd.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')

        res = execute(cmd, ref)

        r = res.get('result', {})

        print(json.dumps({'seg': seg, 'ok': r.get('ok'), 'error': r.get('error'),

                          'pose': (r.get('state') or {}).get('pose'),

                          'grip': (r.get('state') or {}).get('gripper'),

                          'frame': (r.get('camera') or {}).get('color_file')}, ensure_ascii=False))

        if not r.get('ok'):

            return 1

        ref = Path(res['result_file'])

    return 0





if __name__ == '__main__':

    sys.exit(main())

