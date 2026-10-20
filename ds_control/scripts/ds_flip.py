"""Flip the held can 180 deg about the tool Z axis in bounded steps."""
import json, math, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(r'G:\Flip Anything\ds_control')
sys.path.insert(0, str(ROOT / 'src'))
from flip_anything.geometry import pose_matrix
PY = Path(r'G:\Flip Anything\.venv\Scripts\python.exe')


def rpy_of(R):
    return [math.degrees(math.atan2(R[2, 1], R[2, 2])),
            math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0])))),
            math.degrees(math.atan2(R[1, 0], R[0, 0]))]


def run_dir():
    return ROOT.parent / 'logs' / 'experiments' / 'ds_control' / json.loads((ROOT / 'active.json').read_text())['run']


def ref_result(run):
    good = [p for p in sorted(run.glob('result_*.json'), key=lambda q: q.stat().st_mtime)
            if 'state' in json.loads(p.read_text(encoding='utf-8-sig'))]
    return good[-1]


def execute(cmd, ref):
    out = subprocess.run([str(PY), '-X', 'utf8', 'scripts/control.py', 'execute', str(cmd),
                          '--from-result', str(ref)], cwd=str(ROOT), capture_output=True, text=True,
                         encoding='utf-8', errors='replace')
    for line in reversed(out.stdout.strip().splitlines()):
        if line.strip().startswith('{'):
            return json.loads(line.strip())
    raise RuntimeError(out.stdout[-1500:] + out.stderr[-1500:])


def main():
    run = run_dir()
    ref = ref_result(run)
    pose = json.loads(ref.read_text(encoding='utf-8-sig'))['state']['pose']
    R0 = np.array(pose_matrix(pose))[:3, :3]
    pos = pose[:3]
    for k, deg in enumerate((45, 45, 45, 45), start=1):
        a = math.radians(deg)
        Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        Rk = R0 @ Rz
        tgt = [round(v, 4) for v in pos] + [round(v, 4) for v in rpy_of(Rk)]
        payload = {'action': 'move', 'pose': tgt, 'grip_required': True, 'capture': True,
                   'max_translation_mm': 8.0, 'max_rotation_deg': 50.0, 'max_joint_deg': 95.0,
                   'speed': 6.0, 'timeout_s': 60.0}
        cmd = ROOT / 'scene' / ('flip_%d.json' % k)
        cmd.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        res = execute(cmd, ref)
        r = res.get('result', {})
        st = r.get('state') or {}
        g = st.get('gripper') or {}
        print(json.dumps({'step': k, 'ok': r.get('ok'), 'error': r.get('error'),
                          'pose': st.get('pose'), 'gOBJ': g.get('gOBJ'), 'gPO': g.get('gPO'),
                          'frame': (r.get('camera') or {}).get('color_file')}, ensure_ascii=False))
        if not r.get('ok'):
            return 1
        ref = Path(res['result_file'])
        R0 = Rk
    print('flip complete')
    return 0


if __name__ == '__main__':
    sys.exit(main())

