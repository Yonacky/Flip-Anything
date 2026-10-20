"""Read-only: extract gripper command encodings used in previous runs (for parameter reference)."""
import json
from pathlib import Path

for run in sorted((Path(__file__).resolve().parents[2] / 'logs' / 'experiments' / 'ds_control').glob("DS_*")):
    ev = run / "events.jsonl"
    if not ev.exists():
        continue
    rows = [json.loads(l) for l in ev.read_text(encoding="utf8").splitlines() if l.strip()]
    seen = []
    for r in rows:
        if r["kind"] == "gripper_command":
            seen.append(r["payload"])
        if r["kind"] == "robot_sample" and r["payload"].get("gripper"):
            g = r["payload"]["gripper"]
            seen.append({"sample": {k: g[k] for k in ("gOBJ", "gPO", "gACT", "gSTA", "gFLT")}})
    if seen:
        print(run.name)
        for s in seen[:6]:
            print("   ", json.dumps(s, ensure_ascii=False))

