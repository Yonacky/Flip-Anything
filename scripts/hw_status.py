"""Print the current TCP pose / joints / gripper and the latest live state."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hw_cmd
run = hw_cmd.current_run()
folder = ROOT / 'logs' / 'experiments' / 'legacy' / run
print("run:", run)
state_file = folder / "live_state.json"
if state_file.exists():
    print(json.dumps(json.loads(state_file.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
else:
    print("no live_state.json yet")
print("results:", [p.name for p in sorted(folder.glob("result_*.json"))])

