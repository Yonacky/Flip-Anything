"""Deeper, firmer re-grasp: open, advance, close harder. Stops on any failure."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb
import goto


def grip(pos, speed, force, tag, ref):
    payload = {'action': 'grip', 'pos': int(pos), 'speed': int(speed), 'force': int(force),
               'grip_required': False, 'capture': True}
    cmd = tb.ROOT / 'scene' / f'{tag}.json'
    cmd.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    res = goto.execute(cmd, ref)
    r = res.get('result', {})
    st = r.get('state') or {}
    g = st.get('gripper') or {}
    print(json.dumps({'tag': tag, 'ok': r.get('ok'), 'error': r.get('error'),
                      'gOBJ': g.get('gOBJ'), 'gPO': g.get('gPO'), 'gCU': g.get('gCU'),
                      'frame': (r.get('camera') or {}).get('color_file')}, ensure_ascii=False),
          flush=True)
    return (Path(res['result_file']) if r.get('ok') else None)


def main():
    advance = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    pos = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    force = int(sys.argv[3]) if len(sys.argv) > 3 else 45
    run = tb.run_dir()
    ref = goto.last_result(run)
    ref = grip(3, 40, 15, 'regrasp_open', ref)
    if ref is None:
        return 1
    cur = json.loads(ref.read_text(encoding='utf-8-sig'))['state']['pose']
    tgt = [round(cur[0] + advance, 3), round(cur[1], 3), round(cur[2], 3)] + [round(v, 3) for v in cur[3:]]
    code, ref = goto.move_to('regrasp_adv', tgt, False, step_mm=max(20.0, advance), speed=4.0, ref=ref)
    if code:
        return code
    ref = grip(pos, 35, force, f'regrasp_close{pos}', ref)
    if ref is None:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
