import argparse
import json
import math
import sys
import time
from pathlib import Path

from .camera import MockCamera, OrbbecCamera
from .config import load_config
from .robot import MockRobot, XArmObserver
from .trial import Trial, software_versions


def record(cfg, args, robot_factory=None, camera_factory=None):
    hardware = args.backend == "hardware"
    robot_factory = robot_factory or (XArmObserver if hardware else MockRobot)
    camera_factory = camera_factory or (OrbbecCamera if hardware else MockCamera)
    trial = Trial(cfg["recording"]["root"], cfg, args.backend, args.run_id,
                  {"object_id": args.object_id, "purpose": "observation_infrastructure", "mode": args.devices})
    robot = robot_factory(cfg["robot"]) if args.devices in ("robot", "both") else None
    camera = camera_factory(cfg["camera"]) if args.devices in ("camera", "both") else None
    error = None
    status = "completed"
    try:
        if robot:
            trial.event("robot_connected", metadata=robot.open())
        if camera:
            trial.event("camera_started", metadata=camera.open())
        calibration = Path(cfg["calibration"]["file"])
        if calibration.exists():
            # Snapshot even unverified calibration; capture does not use it to move hardware.
            trial.manifest["calibration_snapshot"] = json.loads(calibration.read_text(encoding="utf-8"))
            trial._write_manifest()
        for index in range(args.frames):
            before = robot.read() if robot else None
            frame = camera.capture(trial.path, index) if camera else None
            after = robot.read() if robot and camera else None
            if frame and frame["rgbd_skew_ms"] is not None and frame["rgbd_skew_ms"] > cfg["recording"]["max_rgbd_skew_ms"]:
                raise RuntimeError(f"RGB-D skew exceeds configured limit: {frame['rgbd_skew_ms']}ms")
            trial.event("observation", index=index, robot_before=before, camera=frame, robot_after=after,
                        synchronization="sequential_host_samples; no camera-robot clock calibration")
            if index + 1 < args.frames:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        status, error = "interrupted", "KeyboardInterrupt"
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        trial.event("error", message=error)
    finally:
        # Close independently: a camera-stop error must never prevent robot disconnect.
        for name, resource in (("camera", camera), ("robot", robot)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    status = "failed"
                    error = (error or "") + f"; {name} cleanup: {exc}"
                    trial.event("cleanup_error", device=name, message=str(exc))
        trial.finish(status, error)
    print(json.dumps({"run": str(trial.path), "status": status, "error": error}, ensure_ascii=False, indent=2))
    return 0 if status == "completed" else 130 if status == "interrupted" else 1


def replay(path):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    counts = {}
    missing = []
    last_time = -1
    with (path / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            timestamp = event["host_monotonic_ns"]
            if timestamp < last_time:
                raise ValueError("Event host monotonic timestamps are out of order")
            last_time = timestamp
            kind = event["kind"]
            counts[kind] = counts.get(kind, 0) + 1
            camera = event["payload"].get("camera")
            if camera:
                for key in ("color_file", "depth_file"):
                    asset = (path / camera[key]).resolve()
                    if not asset.is_relative_to(path):
                        raise ValueError("Asset reference escapes run directory")
                    if not asset.is_file():
                        missing.append(camera[key])
    result = {"run_id": manifest["run_id"], "backend": manifest["backend"],
              "status": manifest["status"], "task_outcome": manifest["task_outcome"],
              "event_counts": counts, "missing_assets": missing}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if manifest["status"] == "running" or counts.get("trial_finished") != 1 or counts.get("trial_started") != 1:
        raise ValueError("Trial is incomplete or has invalid lifecycle events")
    return 1 if missing or manifest["status"] != "completed" else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Flip Anything read-only hardware diagnostics and recording")
    parser.add_argument("--config", default="configs/platform.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report SDK installation; does not connect to hardware")
    sub.add_parser("cameras", help="Enumerate Orbbec cameras; does not start streaming")
    capture = sub.add_parser("record", help="Record bounded observations; sends no motion commands")
    capture.add_argument("--backend", choices=("mock", "hardware"), default="mock")
    capture.add_argument("--devices", choices=("robot", "camera", "both"), default="both")
    capture.add_argument("--frames", type=int, default=10)
    capture.add_argument("--interval", type=float, default=0.1)
    capture.add_argument("--run-id")
    capture.add_argument("--object-id", default="unspecified")
    playback = sub.add_parser("replay", help="Check lifecycle events and referenced assets")
    playback.add_argument("run")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            versions = software_versions()
            versions["hardware_dependencies_ready"] = all(versions[k] for k in ("xarm-python-sdk", "pyorbbecsdk2", "numpy", "opencv-python"))
            print(json.dumps(versions, ensure_ascii=False, indent=2))
            return 0 if versions["hardware_dependencies_ready"] else 1
        if args.command == "cameras":
            print(json.dumps(OrbbecCamera.enumerate(load_config(args.config)["camera"].get("log_dir")), ensure_ascii=False, indent=2))
            return 0
        if args.command == "replay":
            return replay(args.run)
        if not 1 <= args.frames <= 10000 or not math.isfinite(args.interval) or not 0 <= args.interval <= 10:
            raise ValueError("frames must be 1..10000; interval must be finite and 0..10 seconds")
        return record(load_config(args.config), args)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
