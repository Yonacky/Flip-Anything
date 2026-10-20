"""写一条 move 命令。用法: run_dir name pose_json expected_json [speed] [max_trans] [max_rot]"""
import json, sys
from pathlib import Path
run, name, pose, expected = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), json.loads(sys.argv[4])
speed = float(sys.argv[5]) if len(sys.argv) > 5 else 12.0
mt = float(sys.argv[6]) if len(sys.argv) > 6 else 60.0
mr = float(sys.argv[7]) if len(sys.argv) > 7 else 60.0
cmd = {"action": "move", "pose": pose, "expected_pose": expected, "speed": speed,
       "max_translation_mm": mt, "max_rotation_deg": mr, "capture": True, "grip_required": False}
Path(run, "commands", name + ".json").write_text(json.dumps(cmd), encoding="utf-8")
print("wrote", name, pose)
