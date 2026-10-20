"""Record the roll_m90 pose, then two pitch poses; appends to the calibration manifest."""

import json, math, subprocess, sys

from pathlib import Path

import numpy as np



ROOT = Path(r'G:\Flip Anything\ds_control')

sys.path.insert(0, str(ROOT / 'src'))

from flip_anything.geometry import pose_matrix

PY = Path(r'G:\Flip Anything\.venv\Scripts\python.exe')

OLD = ROOT.parent / 'logs' / 'experiments' / 'ds_control' / 'DS_20261018T203231_b2ac5b99' / 'calib_manifest.json'





def rpy_of(R):

    return [math.degrees(math.atan2(R[2, 1], R[2, 2])),

            math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0])))),

            math.degrees(math.atan2(R[1, 0], R[0, 0]))]





def run_dir():

    return ROOT.parent / 'logs' / 'experiments' / 'ds_control' / json.loads((ROOT / 'active.json').read_text())['run']





def last_result(run):

    good = [p for p in sorted(run.glob('result_*.json'), key=lambda q: q.stat().st_mtime)

            if 'state' in json.loads(p.read_text(encoding='utf-8-sig'))]

    if not good:

        raise RuntimeError('no reference result yet')

    return good[-1]





def execute(cmd, ref):

    out = subprocess.run([str(PY), '-X', 'utf8', 'scripts/control.py', 'execute', str(cmd),

                          '--from-result', str(ref)], cwd=str(ROOT), capture_output=True, text=True,

                         encoding='utf-8', errors='replace')

    for line in reversed(out.stdout.strip().splitlines()):

        if line.strip().startswith('{'):

            return json.loads(line.strip())

    raise RuntimeError(out.stdout[-1500:] + out.stderr[-1500:])





def move(run, ref, name, pos, rpy, mrot=95.0, mtran=50.0):

    payload = {'action': 'move', 'pose': [round(v, 4) for v in pos] + [round(v, 4) for v in rpy],

               'grip_required': False, 'capture': True, 'max_translation_mm': mtran,

               'max_rotation_deg': mrot, 'max_joint_deg': 95.0, 'speed': 8.0, 'timeout_s': 60.0}

    cmd = ROOT / 'scene' / ('cal_' + name + '.json')

    cmd.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')

    res = execute(cmd, ref)

    ok = res.get('result', {}).get('ok')

    print(f'{name:16s} ok={ok} {res.get("result", {}).get("error", "")}', flush=True)

    if not ok:

        raise SystemExit(1)

    return Path(res['result_file']), res['result']





def main():

    run = run_dir()

    entries = json.loads(OLD.read_text())

    already = {e['name'] for e in entries}

    ref = last_result(run)

    base_pos = [400.0, -12.0, 170.0]

    base_rpy = [180.0, 89.5, 180.0]

    R0 = np.array(pose_matrix(base_pos + base_rpy))[:3, :3]



    # 1) record the current pose as roll_m90 if it is still there

    cur = json.loads(ref.read_text(encoding='utf-8-sig'))['state']['pose']

    if abs(abs(cur[3]) - 90.5) < 5 and abs(cur[4]) < 5 and 'roll_m90' not in already:

        out = subprocess.run([str(PY), '-X', 'utf8', 'scripts/control.py', 'observe'],

                             cwd=str(ROOT), capture_output=True, text=True, encoding='utf-8',

                             errors='replace')

        res = None

        for line in reversed(out.stdout.strip().splitlines()):

            if line.strip().startswith('{'):

                res = json.loads(line.strip())

                break

        if res is None:

            raise RuntimeError(out.stdout[-1500:] + out.stderr[-1500:])

        ref = Path(res['result_file'])

        st, cam = res['result']['state'], res['result']['camera']

        entries.append({'name': 'roll_m90', 'pose': st['pose'], 'frame': cam['color_file'],

                        'run': str(run)})

        OLD.write_text(json.dumps(entries, indent=1), encoding='utf-8')

        print('recorded roll_m90 at', np.round(st['pose'], 2), flush=True)



    # 2) back to base, then two pitch poses

    ref, _ = move(run, ref, 'return_m90', base_pos, base_rpy)

    for name, deg in (('pitch_p20', 20.0), ('pitch_m20', -20.0)):

        a = math.radians(deg)

        Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])

        ref, r = move(run, ref, name, base_pos, list(rpy_of(R0 @ Ry)), mrot=40.0)

        st, cam = r['state'], r['camera']

        entries.append({'name': name, 'pose': st['pose'], 'frame': cam['color_file'], 'run': str(run)})

        OLD.write_text(json.dumps(entries, indent=1), encoding='utf-8')

        ref, _ = move(run, ref, 'return_' + name, base_pos, base_rpy, mrot=40.0)

    print('entries', len(entries), 'at', OLD)

    return 0





if __name__ == '__main__':

    sys.exit(main())

