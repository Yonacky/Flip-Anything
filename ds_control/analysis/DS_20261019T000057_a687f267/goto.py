"""Move the TCP to a base-frame pose in bounded, receipt-checked steps (empty or held)."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb

PY = Path(r'G:\Flip Anything\.venv\Scripts\python.exe')


def expmap(axis, ang):
    axis = axis / np.linalg.norm(axis)
    vx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * vx + (1 - math.cos(ang)) * (vx @ vx)


def axis_angle(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if abs(math.pi - ang) < 1e-6:
        w, V = np.linalg.eigh(R + np.eye(3))
        axis = V[:, np.argmax(w)]
        return axis / np.linalg.norm(axis), ang
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(ang))
    return axis, ang


def slerp(Ra, Rb, t):
    axis, ang = axis_angle(Ra.T @ Rb)
    if ang < 1e-9:
        return Ra.copy()
    return Ra @ expmap(axis, ang * t)


def last_result(run):
    good = [p for p in sorted(run.glob('result_*.json'), key=lambda q: q.stat().st_mtime)
            if 'state' in json.loads(p.read_text(encoding='utf-8-sig'))]
    return good[-1]


def execute(cmd, ref):
    out = subprocess.run([str(PY), '-X', 'utf8', 'scripts/control.py', 'execute', str(cmd),
                          '--from-result', str(ref)], cwd=str(tb.ROOT), capture_output=True,
                         text=True, encoding='utf-8', errors='replace')
    for line in reversed(out.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith('{'):
            try:
                return json.loads(line)
            except Exception:
                continue
    raise RuntimeError(f'execute failed: {out.stdout[-1500:]} {out.stderr[-1500:]}')


def move_to(tag, tgt_pose, grip_required, step_mm=70.0, step_deg=50.0, speed=8.0, ref=None):
    run = tb.run_dir()
    ref = ref or last_result(run)
    cur = json.loads(ref.read_text(encoding='utf-8-sig'))['state']['pose']
    R0 = np.array(tb.pose_matrix(cur))[:3, :3]
    p0 = np.array(cur[:3], dtype=float)
    Rt = np.array(tb.pose_matrix([float(v) for v in tgt_pose]))[:3, :3]
    pt = np.array([float(v) for v in tgt_pose[:3]])
    dist = float(np.linalg.norm(pt - p0))
    ang = math.degrees(axis_angle(R0.T @ Rt)[1])
    n = max(1, int(math.ceil(dist / step_mm)), int(math.ceil(ang / step_deg)))
    print(json.dumps({'tag': tag, 'from': [round(float(v), 1) for v in cur],
                      'to': [round(float(v), 1) for v in tgt_pose],
                      'delta_mm': round(dist, 1), 'delta_deg': round(ang, 1), 'steps': n},
                     ensure_ascii=False), flush=True)
    for k in range(1, n + 1):
        t = k / n
        Rk = slerp(R0, Rt, t)
        pk = p0 + (pt - p0) * t
        tgt = [round(float(v), 4) for v in pk] + [round(float(v), 4) for v in tb.rpy_of(Rk)]
        payload = {'action': 'move', 'pose': tgt, 'grip_required': bool(grip_required),
                   'capture': True,
                   'max_translation_mm': max(15.0, min(90.0, dist / n + 15.0)),
                   'max_rotation_deg': max(15.0, min(95.0, ang / n + 15.0)),
                   'max_joint_deg': 95.0, 'speed': speed, 'timeout_s': 60.0}
        cmd = tb.ROOT / 'scene' / f'{tag}_{k}.json'
        cmd.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        res = execute(cmd, ref)
        r = res.get('result', {})
        st = r.get('state') or {}
        g = st.get('gripper') or {}
        print(json.dumps({'tag': tag, 'step': k, 'of': n, 'ok': r.get('ok'), 'error': r.get('error'),
                          'pose': [round(float(v), 1) for v in st.get('pose', [])] if st else None,
                          'gOBJ': g.get('gOBJ'), 'gPO': g.get('gPO'),
                          'frame': (r.get('camera') or {}).get('color_file')},
                         ensure_ascii=False), flush=True)
        if not r.get('ok'):
            return 1, ref
        ref = Path(res['result_file'])
    return 0, ref


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pose', required=True, help='x,y,z,roll,pitch,yaw')
    p.add_argument('--grip-required', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--tag', default='goto')
    p.add_argument('--step-mm', type=float, default=70.0)
    p.add_argument('--step-deg', type=float, default=50.0)
    p.add_argument('--speed', type=float, default=8.0)
    args = p.parse_args()
    tgt = [float(v) for v in args.pose.split(',')]
    if not args.execute:
        run = tb.run_dir()
        cur = json.loads(last_result(run).read_text(encoding='utf-8-sig'))['state']['pose']
        print(json.dumps({'target': tgt, 'current': cur,
                          'delta_mm': round(float(np.linalg.norm(np.array(tgt[:3]) - np.array(cur[:3]))), 1)},
                         ensure_ascii=False, indent=1))
        return 0
    code, _ = move_to(args.tag, tgt, args.grip_required, args.step_mm, args.step_deg, args.speed)
    return code


if __name__ == '__main__':
    sys.exit(main())
