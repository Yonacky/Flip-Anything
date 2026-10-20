"""Aim the wrist camera at a base-frame point and move there in bounded steps.

Read-only until --execute. The camera is re-aimed with the minimal rotation that
takes the current optical axis onto the requested line of sight; a bounded list of
extra rolls about the optical axis is tried in order, because a wrist joint limit
rejects some rolls (a rejection sends no motion). Camera->TCP comes from the mount
recorded for this round; IK is left to the executor.
"""
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


def rot_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.eye(3)
    return expmap(v / s, math.atan2(s, float(a @ b)))


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


def candidate(run, C, P, roll_deg):
    mount = tb.load_mount(tb.PRIOR_MOUNT)
    ref = last_result(run)
    cur = json.loads(ref.read_text(encoding='utf-8-sig'))['state']['pose']
    R0 = np.array(tb.pose_matrix(cur))[:3, :3]
    p0 = np.array(cur[:3], dtype=float)
    R_cam_cur = R0 @ mount['R_tcp_cam']
    z_des = (P - C) / np.linalg.norm(P - C)
    R_cam_des = rot_between(R_cam_cur[:, 2], z_des) @ R_cam_cur
    if abs(roll_deg) > 1e-9:
        R_cam_des = R_cam_des @ expmap(np.array([0.0, 0.0, 1.0]), math.radians(roll_deg))
    R_tool_des = R_cam_des @ mount['R_tcp_cam'].T
    tcp_des = C - R_tool_des @ mount['t_tcp_cam_mm']
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R0.T @ R_tool_des) - 1) / 2))))
    return {'ref': ref, 'cur': cur, 'R0': R0, 'p0': p0, 'R_des': R_tool_des,
            'tcp_des': tcp_des, 'ang': ang, 'dist': float(np.linalg.norm(tcp_des - p0)),
            'roll': roll_deg}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--camera', required=True, help='desired camera origin x,y,z in base mm')
    p.add_argument('--target', required=True, help='base point the optical axis should hit')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--tag', default='aim')
    p.add_argument('--roll-list', default='0')
    p.add_argument('--step-mm', type=float, default=60.0)
    p.add_argument('--step-deg', type=float, default=45.0)
    p.add_argument('--speed', type=float, default=12.0)
    args = p.parse_args()

    run = tb.run_dir()
    C = np.array([float(v) for v in args.camera.split(',')])
    P = np.array([float(v) for v in args.target.split(',')])
    rolls = [float(v) for v in args.roll_list.split(',') if v.strip()]
    print(json.dumps({'camera_target': C.tolist(), 'aim_point': P.tolist(),
                      'rolls': rolls}, ensure_ascii=False))
    last_error = None
    for roll in rolls:
        c = candidate(run, C, P, roll)
        print(json.dumps({'roll': roll, 'tcp_target': np.round(c['tcp_des'], 2).tolist(),
                          'rpy': [round(v, 2) for v in tb.rpy_of(c['R_des'])],
                          'delta_mm': round(c['dist'], 1), 'delta_deg': round(c['ang'], 1)},
                         ensure_ascii=False), flush=True)
        if not args.execute:
            continue
        ref = c['ref']
        n = max(1, int(math.ceil(c['dist'] / args.step_mm)), int(math.ceil(c['ang'] / args.step_deg)))
        ok = True
        for k in range(1, n + 1):
            t = k / n
            Rk = slerp(c['R0'], c['R_des'], t)
            pk = c['p0'] + (c['tcp_des'] - c['p0']) * t
            tgt = [round(float(v), 4) for v in pk] + [round(float(v), 4) for v in tb.rpy_of(Rk)]
            payload = {'action': 'move', 'pose': tgt, 'grip_required': False, 'capture': True,
                       'max_translation_mm': max(20.0, min(90.0, c['dist'] / n + 15.0)),
                       'max_rotation_deg': max(20.0, min(95.0, c['ang'] / n + 15.0)),
                       'max_joint_deg': 95.0, 'speed': args.speed, 'timeout_s': 60.0}
            cmd = tb.ROOT / 'scene' / f"{args.tag}_r{int(roll)}_{k}.json"
            cmd.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
            res = execute(cmd, ref)
            r = res.get('result', {})
            print(json.dumps({'roll': roll, 'step': k, 'of': n, 'ok': r.get('ok'),
                              'error': r.get('error'), 'pose': (r.get('state') or {}).get('pose'),
                              'frame': (r.get('camera') or {}).get('color_file')},
                             ensure_ascii=False), flush=True)
            if not r.get('ok'):
                ok = False
                last_error = r.get('error')
                break
            ref = Path(res['result_file'])
        if ok:
            print('arrived with roll', roll)
            return 0
    print('no candidate accepted; last error:', last_error)
    return 1


if __name__ == '__main__':
    sys.exit(main())
