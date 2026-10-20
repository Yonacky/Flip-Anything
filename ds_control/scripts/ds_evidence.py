"""Read-only evidence extraction for the run record. Sends nothing."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT.parent / 'logs' / 'experiments' / 'ds_control' / "DS_20261016T220611_8b322867"
rows = [json.loads(l) for l in (RUN / "events.jsonl").read_text(encoding="utf8").splitlines() if l.strip()]
print("events:", len(rows))
prev = None
print("\n-- controller state transitions (session view) --")
for r in rows:
    if r["kind"] != "robot_sample":
        continue
    p = r["payload"]
    key = (p["state"], tuple(round(v, 2) for v in p["pose"]), tuple(round(v, 2) for v in p["joint_deg"]))
    if key != prev:
        print(r["utc"][11:23], "state", p["state"], "pose", [round(v, 2) for v in p["pose"]],
              "joints", [round(v, 1) for v in p["joint_deg"]])
        prev = key
print("\n-- commands and outcomes --")
for r in rows:
    k = r["kind"]
    if k == "command":
        print(r["utc"][11:23], "COMMAND", json.dumps(r["payload"]["request"], ensure_ascii=False)[:180])
    elif k in ("command_rejected", "error", "session_error", "trial_finished"):
        print(r["utc"][11:23], k.upper(), json.dumps(r["payload"], ensure_ascii=False)[:220])
print("\n-- observations --")
for r in rows:
    if r["kind"] == "observation":
        c = r["payload"]["camera"]
        print(r["utc"][11:23], r["payload"]["phase"], c["color_file"], c["depth_file"])
