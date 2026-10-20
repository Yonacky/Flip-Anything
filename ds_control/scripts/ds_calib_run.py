"""Drive the arm through calibration poses inside the single session and record RGB-D + pose."""

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

    rid = json.loads((ROOT / 'active.json').read_text())['run']

    return ROOT.parent / 'logs' / 'experiments' / 'ds_control' / rid





def last_result(run):

    good = [p for p in sorted(run.glob('result_*.json'), key=lambda q: q.stat().st_mtime)

            if 'state' in json.loads(p.read_text(encoding='utf-8-sig'))]

    return good[-1]





def execute(cmd, ref):

    out = subprocess.run([str(PY), '-X', 'utf8', 'scripts/control.py', 'execute', str(cmd),

                          '--from-result', str(ref)], cwd=str(ROOT), capture_output=True, text=True,

                         encoding='utf-8', errors='replace')

    for line in reversed(out.stdout.strip().splitlines()):

        line = line.strip()

        if line.startswith('{'):

            try:

                return json.loads(line)

            except Exception:

                continue

    raise RuntimeError(f'execute failed: {out.stdout[-2000:]} {out.stderr[-2000:]}')





def ang(a, b):

    Ra = np.array(pose_matrix(list(a)))[:3, :3]

    Rb = np.array(pose_matrix(list(b)))[:3, :3]

    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Ra.T @ Rb) - 1) / 2))))





def main():

    run = run_dir()

    base_pos = [400.0, -12.0, 170.0]

    base_rpy = [180.0, 89.5, 180.0]

    R0 = np.array(pose_matrix(base_pos + base_rpy))[:3, :3]



    def rot(axis, deg):

        if axis == 'z':

            A = np.array([[math.cos(math.radians(deg)), -math.sin(math.radians(deg)), 0],

                          [math.sin(math.radians(deg)), math.cos(math.radians(deg)), 0], [0, 0, 1]])

        else:

            A = np.array([[math.cos(math.radians(deg)), 0, math.sin(math.radians(deg))], [0, 1, 0],

                          [-math.sin(math.radians(deg)), 0, math.cos(math.radians(deg))]])

        return list(rpy_of(R0 @ A))



    plan = [('base', base_pos, base_rpy),

            ('ty1', [base_pos[0], base_pos[1] + 45.0, base_pos[2]], base_rpy),

            ('tx1', [base_pos[0] + 45.0, base_pos[1], base_pos[2]], base_rpy),

            ('tz1', [base_pos[0], base_pos[1], base_pos[2] + 45.0], base_rpy)]

    for nm, ax, dg in (('roll_p50', 'z', 50), ('roll_m50', 'z', -50), ('roll_p90', 'z', 90),

                       ('roll_m90', 'z', -90), ('pitch_p20', 'y', 20), ('pitch_m20', 'y', -20)):

        plan.append(('return_' + nm, base_pos, base_rpy))

        plan.append((nm, base_pos, rot(ax, dg)))



    mpath = run / 'calib_manifest.json'

    entries = json.loads(mpath.read_text()) if mpath.exists() else []

    have = {e['name'] for e in entries}

    ref = last_result(run)

    for nm, pos, r in plan:

        if nm in have:

            continue

        cur = json.loads(ref.read_text(encoding='utf-8-sig'))['state']['pose']

        tgt = [round(v, 4) for v in pos] + [round(v, 4) for v in r]

        if np.linalg.norm(np.array(cur[:3]) - np.array(pos)) < 0.3 and ang(cur, tgt) < 0.3:

            entries.append({'name': nm, 'pose': cur, 'frame': None, 'run': str(run)})

            mpath.write_text(json.dumps(entries, indent=1))

            continue

        payload = {'action': 'move', 'pose': tgt, 'grip_required': False, 'capture': True,

                   'max_translation_mm': 90.0, 'max_rotation_deg': 95.0, 'max_joint_deg': 95.0,

                   'speed': 8.0, 'timeout_s': 60.0}

        cmd = ROOT / 'scene' / ('cal_' + nm + '.json')

        cmd.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')

        res = execute(cmd, ref)

        ok = res.get('result', {}).get('ok')

        print(f'{nm:10s} ok={ok} {res.get("result", {}).get("error", "")}', flush=True)

        if not ok:

            print(json.dumps(res, ensure_ascii=False)[:600])

            return 1

        ref = Path(res['result_file'])

        st = res['result']['state']

        cam = res['result'].get('camera') or {}

        entries.append({'name': nm, 'pose': st['pose'], 'frame': cam.get('color_file', ''),

                        'run': str(run)})

        mpath.write_text(json.dumps(entries, indent=1), encoding='utf-8')

    print('manifest', mpath, len(entries))

    return 0





if __name__ == '__main__':

    sys.exit(main())

