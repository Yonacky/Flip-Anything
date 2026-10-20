import json
from pathlib import Path
root = Path(__file__).resolve().parents[2] / 'logs' / 'experiments' / 'ds_control'
for run in sorted(root.glob("DS_*")):
    ev = run / "events.jsonl"
    if not ev.exists():
        continue
    rows = [json.loads(l) for l in ev.read_text(encoding="utf8").splitlines() if l.strip()]
    # first sample state, and any set_state calls
    first = next((r for r in rows if r["kind"] == "robot_sample"), None)
    fs = first["payload"]["state"] if first else None
    writes = []
    for i, r in enumerate(rows):
        if r["kind"] == "sdk_request" and r["payload"].get("method") in ("set_state", "motion_enable", "clean_error", "clean_warn", "set_mode"):
            res = None
            for r2 in rows[i+1:i+4]:
                if r2["kind"] == "sdk_result" and r2["payload"].get("method") == r["payload"]["method"]:
                    res = r2["payload"].get("result")
                    break
            writes.append((r["utc"][11:19], r["payload"]["method"], r["payload"].get("kwargs"), res))
    states = sorted(set(r["payload"]["state"] for r in rows if r["kind"] == "robot_sample"))
    print(run.name, "first_state", fs, "states", states, "writes", len(writes))
    for w in writes[:8]:
        print("   ", w)
