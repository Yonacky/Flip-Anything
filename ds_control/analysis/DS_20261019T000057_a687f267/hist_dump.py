"""Dump the pose/gripper sequence of a historical run (read-only reference)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tb

run = tb.LOGS / (sys.argv[1] if len(sys.argv) > 1 else 'DS_20261018T210456_00845c9c')
mount = tb.load_mount(tb.PRIOR_MOUNT)
for path in sorted(run.glob('result_*.json'), key=lambda p: p.stat().st_mtime):
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    st = data.get('state')
    if not st:
        print(path.name, 'ok=', data.get('ok'), 'rejected=', data.get('rejected'),
              data.get('error'))
        continue
    g = st.get('gripper') or {}
    R_cam, cam = tb.camera_pose(st['pose'], mount)
    print(json.dumps({'file': path.name[7:22], 'ok': data.get('ok'),
                      'pose': [round(v, 1) for v in st['pose']],
                      'cam': [round(float(v), 1) for v in cam],
                      'gOBJ': g.get('gOBJ'), 'gPO': g.get('gPO'), 'gFLT': g.get('gFLT'),
                      'frame': (data.get('camera') or {}).get('color_file')}, ensure_ascii=False))
