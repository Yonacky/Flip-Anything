"""从各 run 的事件/结果/命令文件重建 (run, frame, pose) 序列。"""
import json, sys
from pathlib import Path

runs = [Path(p) for p in sys.argv[2:]]
entries = []
for run in runs:
    last_pose = None
    baseline = None
    for line in (run / 'events.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['kind'] == 'robot_sample':
            last_pose = row['payload']['pose']
        elif row['kind'] == 'observation' and row['payload']['phase'] == 'session_start':
            cam = row['payload']['camera']
            baseline = {'run': str(run), 'frame': cam['color_file'].replace('_color.png', ''),
                        'pose': last_pose, 'name': 'session_start'}
    if baseline is not None:
        if baseline['pose'] is None:
            cmds = sorted((run / 'commands').glob('*.json'), key=lambda p: p.name)
            baseline['pose'] = json.loads(cmds[0].read_text(encoding='utf-8-sig'))['expected_pose']
        entries.append(baseline)
    for path in sorted(run.glob('result_*.json'), key=lambda p: p.name):
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if not data.get('ok') or 'camera' not in data:
            print('skip (not ok):', path.name)
            continue
        entries.append({'run': str(run), 'frame': data['camera']['color_file'].replace('_color.png', ''),
                        'pose': data['state']['pose'], 'name': path.name[7:-5]})
Path(sys.argv[1]).write_text(json.dumps(entries, indent=2), encoding='utf-8')
for e in entries:
    print(f"{e['name']:16s} {Path(e['run']).name:24s} {e['frame']}  pose={[round(v,2) for v in e['pose']]}")
