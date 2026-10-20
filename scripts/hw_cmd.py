"""Send commands to a resident hw_session process and wait for file results."""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'logs' / 'experiments' / 'legacy'


def current_run():
    return (RUNS / "_current_run.txt").read_text(encoding="ascii").strip()


def send(action, name=None, run=None, wait=True, timeout=180.0, **fields):
    run = run or current_run()
    folder = RUNS / run
    inbox = folder / "commands"
    inbox.mkdir(parents=True, exist_ok=True)
    if name is None:
        existing = sorted(p.stem for p in inbox.glob("*.json"))
        index = len(existing) + 1
        while f"c{index:03d}_{action}" in existing:
            index += 1
        name = f"c{index:03d}_{action}"
    payload = {"action": action, **fields}
    path = inbox / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    if not wait:
        return {"sent": name, "run": run}
    result_path = folder / f"result_{name}.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        time.sleep(0.5)
    return {"ok": False, "error": f"timeout waiting for {result_path}", "sent": name}


def summarize(result, keys=None):
    if not isinstance(result, dict):
        return result
    if not result.get("ok", True):
        return {"ok": False, "error": result.get("error"), "traceback": result.get("traceback", "")[-1200:]}
    out = {"ok": True}
    for key, value in result.items():
        if key in ("ok", "traceback"):
            continue
        if key == "observation" and isinstance(value, dict):
            out["observation"] = {k: value.get(k) for k in
                                  ("label", "index", "color_file", "pose_before", "state_before", "rgbd_skew_ms")}
        elif key == "state" and isinstance(value, dict):
            out["state"] = {k: value.get(k) for k in ("state", "error_warn", "pose", "joint_deg", "gripper")}
        elif key == "gripper" and isinstance(value, dict):
            out["gripper"] = value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = value
    if isinstance(result.get("before"), dict) and isinstance(result.get("after"), dict):
        out["before_pose"] = result["before"].get("pose")
        out["before_gripper"] = result["before"].get("gripper")
    return out


if __name__ == "__main__":
    args = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"action": "state"}
    print(json.dumps(summarize(send(**args)), ensure_ascii=False, indent=2, default=str))
