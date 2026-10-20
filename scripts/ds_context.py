"""Read-only, project-scoped DS context. Never imports a hardware module."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def path_in(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError(f"Invalid relative path: {relative!r}")
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Path escapes workspace: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes workspace: {relative}")
    return path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate(root):
    manifest = read_json(root / "context-release.json")
    for item in manifest["files"]:
        path = path_in(root, item["path"])
        if not path.is_file():
            raise ValueError(f"Missing context file: {item['path']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Changed context file: {item['path']}")
    return {item["path"] for item in manifest["files"]}


def load_context(root, project, full=False):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", project):
        raise ValueError("Invalid project id")
    files = validate(root)

    def bound(relative):
        if relative not in files:
            raise ValueError(f"Untracked context reference: {relative}")
        return path_in(root, relative)

    config = read_json(bound(f"projects/{project}/project.json"))
    if config["id"] != project:
        raise ValueError("Project identity mismatch")
    environment = read_json(bound(config["environment"]))
    components = {name: read_json(bound(ref)) for name, ref in environment["components"].items()}
    if not full:
        keep = {
            "hand_eye_profile": ["status", "tool_frame", "tool_from_camera", "intrinsics", "units"],
            "table_plane": ["status", "frame", "normal", "offset_m", "plane_equation", "units", "limitations"],
            "opening_dependent_grasp_model": ["opening_range_m", "per_opening_model_error_bound_m", "fixed_max_position_error_m", "bound_type", "tool_frame", "units"],
        }
        for name, keys in keep.items():
            if name not in components:
                continue
            components[name] = {key: components[name][key] for key in keys if key in components[name]}
            components[name]["full_data_path"] = environment["components"][name]
        tcp = components.get("operational_grasp_tcp", {})
        tcp.pop("publication", None)  # Archived publication state is superseded by the project binding.
        for key in ("load_lift_test_performed", "held_object_rotation_test_performed"):
            tcp.get("validation", {}).pop(key, None)
        environment = {key: value for key, value in environment.items() if key not in ("validation", "use_scope")}
    for ref in config["read_next"]:
        bound(ref)
    experiences = []
    for ref in config["experiences"]:
        card = read_json(bound(ref))
        if card["project_id"] != project or card["use"] != "planning_reference_only":
            raise ValueError("Experience scope mismatch")
        for evidence in card["evidence"]:
            bound(evidence)
        experiences.append(card)
    # Mutable installation release is deliberately read fresh, never copied into an authorization.
    gate_path = root / "devices/calibration_release.json"
    gate = read_json(gate_path) if gate_path.is_file() else {"status": "missing"}
    return {
        "schema_version": 1, "project_id": project,
        "purpose": "planning_context_only", "hardware_authorization": False,
        "read_next": config["read_next"], "environment": environment,
        "calibration": components, "experiences": experiences,
        "current_state": "unknown_until_fresh_observation",
        "pipeline_calibration_release_status": gate.get("status", "unknown"),
        "execution_entrypoints": {
            "pipeline": (root / "scripts/pipeline_cli.py").is_file(),
            "agent": (root / "scripts/agent_cli.py").is_file() and (root / "flip_pipeline/agent_harness.py").is_file(),
        },
        "notice": "Use the existing pipeline checks and authorization. Use the published task-scoped calibration. Archived poses and receipts are not current state.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "context", "check"])
    parser.add_argument("--project")
    parser.add_argument("--full", action="store_true", help="Include complete calibration model and per-view data")
    args = parser.parse_args()
    if args.command == "list":
        tracked = validate(ROOT)
        result = [read_json(path_in(ROOT, p))["id"] for p in sorted(tracked)
                  if p.startswith("projects/") and p.endswith("/project.json")]
    else:
        if not args.project:
            parser.error("context/check require --project; projects never switch implicitly")
        result = load_context(ROOT, args.project, full=args.full)
        if args.command == "check":
            result = {"ok": True, "project_id": args.project,
                      "files_checked": len(validate(ROOT)),
                      "hardware_contacted": False,
                      "pipeline_calibration_release_status": result["pipeline_calibration_release_status"],
                      "execution_entrypoints": result["execution_entrypoints"]}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"Context unavailable: {exc}")
