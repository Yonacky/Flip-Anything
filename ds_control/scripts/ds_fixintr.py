import json
from pathlib import Path
run = Path(r"G:/Flip Anything/logs/experiments/ds_control/DS_20261016T222430_cc837571")
rows = [json.loads(l) for l in (run / "events.jsonl").read_text(encoding="utf8").splitlines() if l.strip()]
meta = next(r["payload"]["metadata"] for r in rows if r["kind"] == "camera_started")
(run / "_intrinsics.txt").write_text(json.dumps({"payload": {"metadata": meta}}, ensure_ascii=False), encoding="utf8")
print("intrinsics written; valid frames:", sorted(p.name for p in run.glob("frame_*_color.png")))
