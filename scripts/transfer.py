"""Offline, evidence-preserving export from a DS workspace. Never controls hardware."""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = {"success", "failure", "incomplete", "unknown"}
FORBIDDEN_DIRS = {".venv", ".git", ".aws", ".ssh", ".codex", ".agents", "__pycache__"}
REQUIRED = {
    "task_id", "object_instance_id", "plan_id", "run_id", "physical_result",
    "autonomy_result", "human_interventions", "failures", "recoveries",
    "unresolved", "evidence", "elapsed_s",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def checked_json(path, guards):
    data = Path(path).read_bytes()
    guards[Path(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data.decode("utf-8-sig"))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def is_link(path):
    info = Path(path).lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def inside(path, base):
    try:
        Path(path).relative_to(base)
        return True
    except ValueError:
        return False


def safe_path(value, base=None, exists=True):
    """Check the lexical path AND every component before resolving links."""
    root = Path(base if base is not None else ROOT).absolute()
    path = Path(value)
    path = Path(os.path.abspath(path if path.is_absolute() else root / path))
    if not inside(path, root):
        raise ValueError(f"Path outside workspace/bundle: {value}")
    if root.exists() and is_link(root):
        raise ValueError(f"Linked workspace/bundle root: {root}")
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            if is_link(current):
                raise ValueError(f"Symlink/junction not allowed: {current}")
    if exists and not path.exists():
        raise ValueError(f"Missing evidence or input: {path}")
    if not inside(path.resolve(), root.resolve()):
        raise ValueError(f"Resolved path outside workspace/bundle: {value}")
    return path


def permitted_file(path):
    parts = {x.lower() for x in path.relative_to(ROOT.absolute()).parts}
    name = path.name.lower()
    if (parts & FORBIDDEN_DIRS or name == ".env" or name.startswith(".env.")
            or "credential" in name or name.startswith(("secret", "id_rsa", "id_ed25519"))
            or path.suffix.lower() in {".pem", ".key", ".pfx", ".p12"}):
        raise ValueError(f"Unrelated/private input is not exportable: {path}")


def tree_files(folder):
    folder = safe_path(folder)
    if not folder.is_dir():
        raise ValueError(f"Expected directory: {folder}")
    found = set()
    for parent, dirs, files in os.walk(folder, followlinks=False):
        for name in dirs + files:
            path = safe_path(Path(parent) / name)
            permitted_file(path)
            if path.is_file():
                found.add(path)
            elif not path.is_dir():
                raise ValueError(f"Non-regular input: {path}")
    return found


def file_ref(value):
    """References are workspace-relative strings or {path, sha256?} objects."""
    if isinstance(value, dict):
        path = value.get("path")
        expected = value.get("sha256")
    else:
        path, expected = value, None
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Evidence reference needs a nonempty path")
    result = safe_path(path)
    permitted_file(result)
    if not result.is_file():
        raise ValueError(f"Evidence must be a regular file: {path}")
    rel = result.relative_to(ROOT.absolute())
    if not (rel.parts[0] in {"logs", "workspace", "devices"}
            or rel.parts[:2] == ("ds_control", "scene")):
        raise ValueError(f"Artifact outside experiment/analysis/device data: {path}")
    if expected is not None and sha256(result) != expected:
        raise ValueError(f"Evidence hash mismatch: {path}")
    return result


def references(node):
    found = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "artifacts":
                if not isinstance(value, list):
                    raise ValueError("artifacts must be an array")
                found.update(file_ref(x) for x in value)
            else:
                found.update(references(value))
    elif isinstance(node, list):
        for value in node:
            found.update(references(value))
    return found


def documents(path):
    if path.suffix.lower() == ".json":
        return [read(path)]
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()]
    return []


def validate_summary(summary, state, episode):
    if not isinstance(summary, dict):
        raise ValueError("summary.json must be an object")
    if not REQUIRED.issubset(summary):
        raise ValueError(f"summary.json missing fields: {sorted(REQUIRED - set(summary))}")
    for key in ("task_id", "object_instance_id", "plan_id", "run_id"):
        if not isinstance(summary[key], str) or not summary[key].strip():
            raise ValueError(f"Summary needs {key}")
        if key in state and state[key] != summary[key]:
            raise ValueError(f"Summary/state identity mismatch: {key}")
    if summary["physical_result"] not in RESULTS or summary["autonomy_result"] not in RESULTS:
        raise ValueError("Result must be success/failure/incomplete/unknown")
    for key in ("human_interventions", "failures", "recoveries", "unresolved"):
        if not isinstance(summary[key], list):
            raise ValueError(f"Summary {key} must be an array")
    if any(item not in summary["human_interventions"] for item in state.get("human_interventions", [])):
        raise ValueError("Summary omits runtime human interventions")
    duration = summary["elapsed_s"]
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
        raise ValueError("elapsed_s must be a finite nonnegative number")
    ev = summary["evidence"]
    if not isinstance(ev, dict) or any(not isinstance(ev.get(k), list)
                                      for k in ("support", "release", "stability")):
        raise ValueError("Summary needs evidence arrays support/release/stability")
    refs = {key: {file_ref(x) for x in ev[key]} for key in ("support", "release", "stability")}
    if summary["physical_result"] == "success":
        outcome_path = episode / "outcome.json"
        if not outcome_path.is_file():
            raise ValueError("Success needs the runtime outcome")
        outcome = read(outcome_path)
        if (state.get("phase") != "completed" or outcome.get("physical_success") is not True
                or outcome.get("backend") != "hardware"):
            raise ValueError("Physical success contradicts the completed hardware outcome")
        if outcome.get("plan_id", summary["plan_id"]) != summary["plan_id"]:
            raise ValueError("Outcome/summary plan mismatch")
        if not all(refs.values()):
            raise ValueError("Success needs support, release and stability evidence")
        snapshot = read(safe_path(episode / "catalog_snapshot.json"))
        plan = snapshot["plan"]
        completion = plan["completion"]
        required = set(completion["requires_completed"]) | {completion["capability"]}
        aliases = snapshot.get("aliases", {})
        completed = {aliases.get(item, item) for item in state.get("completed", [])}
        if plan["id"] != summary["plan_id"] or not required.issubset(completed):
            raise ValueError("Success lacks completion requirements from the selected plan")
        final_capability = snapshot["capabilities"][completion["capability"]]
        required_facts = final_capability["ensures"]
        facts = outcome.get("evidence")
        if (not isinstance(facts, dict) or not facts or not required_facts
                or any(facts.get(name, {}).get("value") is not True for name in required_facts)):
            raise ValueError("Runtime success lacks evidence")
        final_artifacts = set()
        for fact in facts.values():
            if not isinstance(fact, dict) or fact.get("value") is not True or not fact.get("artifacts"):
                raise ValueError("Runtime success has invalid positive evidence")
            final_artifacts.update(file_ref(x) for x in fact["artifacts"])
            reference = safe_path(state.get("last_result", "MISSING_LAST_RESULT"))
            if (not reference.is_file() or fact.get("result_sha256") != sha256(reference)
                    or fact.get("revision") != state.get("revision")):
                raise ValueError("Runtime evidence is not bound to the final receipt")
        if not refs["stability"].intersection(final_artifacts):
            raise ValueError("Stability summary does not cite runtime final evidence")
    if summary["autonomy_result"] == "success":
        outcome = read(episode / "outcome.json") if (episode / "outcome.json").is_file() else {}
        if (summary["physical_result"] != "success" or outcome.get("autonomous_success") is not True
                or summary["human_interventions"] or state.get("human_interventions")
                or outcome.get("human_interventions")):
            raise ValueError("Autonomy success requires unassisted runtime physical success")
    return set().union(*refs.values())


def export_episode(episode, candidates=()):
    episode = safe_path(episode)
    if not inside(episode, safe_path("logs/episodes")) or episode == ROOT.absolute() / "logs/episodes":
        raise ValueError("Episode must be inside logs/episodes")
    state_path, summary_path = episode / "state.json", episode / "summary.json"
    guards = {}
    state = checked_json(safe_path(state_path), guards)
    summary = checked_json(safe_path(summary_path), guards)
    run_id = state.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", run_id):
        raise ValueError("Invalid run ID")
    config_path = safe_path("ds_control/configs/platform.json")
    cfg = checked_json(config_path, guards)
    recording_root = safe_path(config_path.parent / cfg["recording"]["root"])
    if not inside(recording_root, ROOT.absolute() / "logs/experiments"):
        raise ValueError("Configured recording root must be in logs/experiments")
    run = safe_path(recording_root / run_id)
    run_manifest = safe_path(run / "manifest.json")
    manifest = checked_json(run_manifest, guards)
    status = manifest.get("status")
    if not isinstance(status, str) or not status or status.lower() in {"running", "starting", "stopping"}:
        raise ValueError("Cannot export an active or status-less hardware run")
    for key in ("run_id", "run"):
        if key in manifest and manifest[key] != run_id:
            raise ValueError("Raw run/episode ID mismatch")
    files = tree_files(episode) | tree_files(run) | {config_path}
    for name in ("events.jsonl", "catalog_snapshot.json", "outcome.json", "device_profile.json"):
        critical = safe_path(episode / name, exists=False)
        if critical.is_file():
            guards[critical] = sha256(critical)
    if state.get("last_result"):
        reference = safe_path(state["last_result"])
        if not inside(reference, run) or not reference.is_file():
            raise ValueError("Last receipt does not belong to the bound raw run")
        guards[reference] = sha256(reference)
    files.update(validate_summary(summary, state, episode))
    events = safe_path(episode / "events.jsonl")
    files.add(events)
    parse_issues = []
    strict_names = {"state.json", "summary.json", "catalog_snapshot.json", "outcome.json",
                    "device_profile.json", "events.jsonl"}
    for path in list(files):
        if inside(path, episode):
            try:
                nodes = documents(path)
            except (json.JSONDecodeError, UnicodeError) as exc:
                if path.parent == episode and path.name in strict_names:
                    raise ValueError(f"Invalid required episode record: {path}") from exc
                parse_issues.append({"path": path.relative_to(ROOT.absolute()).as_posix(),
                                     "error": str(exc), "handling": "preserved_as_opaque_original"})
                nodes = []
            for node in nodes:
                files.update(references(node))
    selected = []
    for value in candidates:
        candidate = safe_path(value)
        if not inside(candidate, ROOT.absolute() / "logs/candidates") or not candidate.is_file():
            raise ValueError("Candidate must be a file in logs/candidates")
        item = checked_json(candidate, guards)
        if item.get("status") != "candidate":
            raise ValueError("Only candidate status is allowed")
        lesson = item.get("lesson")
        if (not isinstance(lesson, dict) or any(not isinstance(lesson.get(key), str) or not lesson[key].strip()
                for key in ("trigger", "failure", "correction", "verification", "scope"))):
            raise ValueError("Candidate lesson needs trigger/failure/correction/verification/scope")
        if safe_path(item.get("source_episode", "")) != episode:
            raise ValueError("Candidate belongs to another episode")
        if item.get("source_hash") != sha256(events):
            raise ValueError("Candidate source events changed after extraction")
        files.add(candidate)
        files.update(references(item))
        selected.append(item.get("id", candidate.stem))
    profile = state.get("device_profile")
    if profile:
        profile_path = Path(profile)
        files.add(safe_path(profile_path if profile_path.is_absolute() else episode / profile_path))
    release = {}
    snapshot_release = state.get("source_release")
    if snapshot_release:
        release_path = safe_path(episode / snapshot_release)
        if not inside(release_path, episode):raise ValueError("Release snapshot outside episode")
        release = read(release_path)
        files.add(release_path)
    for name in ("source-release.json", "config/source-release.json", "workspace/source-release.json"):
        path = safe_path(name, exists=False)
        if path.is_file():
            files.add(path)
            if not release:
                release = checked_json(path, guards)
    # Freeze the complete input set before copying. Changes during validation/copy abort publication.
    hashes = {path: (sha256(path), path.stat().st_size) for path in sorted(files)}
    if any(hashes[path][0] != expected for path, expected in guards.items()):
        raise ValueError("Input changed during validation")
    outbox = safe_path("transfers/outbox", exists=False)
    outbox.mkdir(parents=True, exist_ok=True)
    folder = safe_path(outbox / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:12]),
                       exists=False)
    folder.mkdir()
    marker = folder / "INCOMPLETE.json"
    write(marker, {"status": "incomplete", "episode": str(episode)})
    try:
        entries = []
        for source, (expected, size) in hashes.items():
            permitted_file(source)
            destination = folder / "data" / source.relative_to(ROOT.absolute())
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if sha256(destination) != expected or destination.stat().st_size != size or sha256(source) != expected:
                raise ValueError(f"Input changed while copying: {source}")
            entries.append({"source_absolute": str(source),
                            "bundle_path": destination.relative_to(folder).as_posix(),
                            "sha256": expected, "size": size})
        if tree_files(episode) | tree_files(run) != {p for p in files if inside(p, episode) or inside(p, run)}:
            raise ValueError("Episode/raw run file set changed during export")
        if any(sha256(path) != h or path.stat().st_size != size for path, (h, size) in hashes.items()):
            raise ValueError("Input changed before publication")
        result = {
            "format": "flip-anything-transfer", "schema_version": 1,
            "status": "candidate_for_review", "created_at": datetime.now(timezone.utc).isoformat(),
            "source_workspace": str(ROOT.absolute()), "release_id": release.get("release_id", release.get("release", release.get("id"))),
            "episode": episode.relative_to(ROOT.absolute()).as_posix(), "run_id": run_id,
            "task_id": summary["task_id"], "object_instance_id": summary["object_instance_id"],
            "plan_id": summary["plan_id"], "physical_result": summary["physical_result"],
            "autonomy_result": summary["autonomy_result"], "raw_status": status,
            "candidate_ids": selected, "parse_issues": parse_issues, "files": entries,
        }
        # Removing the marker is safe: only manifest.json marks a complete transfer.
        marker.unlink()
        write(folder / "manifest.json", result)
        return folder
    except Exception as exc:
        if (folder / "manifest.json").exists():
            (folder / "manifest.json").unlink()
        write(marker, {"status": "incomplete", "error": str(exc), "episode": str(episode)})
        raise


def verify_bundle(value):
    folder = safe_path(value)
    if not folder.is_dir():
        raise ValueError("Expected transfer directory")
    if (folder / "INCOMPLETE.json").exists():
        raise ValueError("Transfer is explicitly incomplete")
    manifest = read(safe_path(folder / "manifest.json"))
    if (manifest.get("format") != "flip-anything-transfer" or manifest.get("schema_version") != 1
            or manifest.get("status") != "candidate_for_review"):
        raise ValueError("Unsupported/incomplete transfer manifest")
    source_root = Path(manifest.get("source_workspace", ""))
    if not source_root.is_absolute():
        raise ValueError("Manifest needs original workspace path")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest has no files")
    listed = set()
    for entry in entries:
        relative = entry.get("bundle_path")
        if not isinstance(relative, str) or not relative.startswith("data/"):
            raise ValueError("Invalid bundle file path")
        if Path(relative).is_absolute() or ".." in Path(relative).parts or "\\" in relative:
            raise ValueError("Bundle path traversal")
        path = safe_path(relative, base=folder)
        if path in listed or not path.is_file():
            raise ValueError("Duplicate/non-file manifest entry")
        listed.add(path)
        if type(entry.get("size")) is not int or entry["size"] < 0:
            raise ValueError("Invalid manifest size")
        if path.stat().st_size != entry["size"] or sha256(path) != entry.get("sha256"):
            raise ValueError(f"Transfer integrity mismatch: {relative}")
        if not isinstance(entry.get("source_absolute"), str) or not Path(entry["source_absolute"]).is_absolute():
            raise ValueError("Missing original absolute path mapping")
        source = Path(entry["source_absolute"])
        if not inside(source, source_root) or source.relative_to(source_root).as_posix() != relative[5:]:
            raise ValueError("Original path mapping does not match bundle path")
    actual = set()
    for parent, dirs, files in os.walk(folder, followlinks=False):
        for name in dirs + files:
            path = safe_path(Path(parent) / name, base=folder)
            if path.is_file() and path != folder / "manifest.json":
                actual.add(path)
    if actual != listed:
        raise ValueError("Manifest does not describe the complete bundle")
    episode_rel = manifest.get("episode")
    if (not isinstance(episode_rel, str) or not episode_rel.startswith("logs/episodes/")
            or ".." in Path(episode_rel).parts or "\\" in episode_rel):
        raise ValueError("Manifest episode path is invalid")
    summary = read(safe_path("data/" + episode_rel + "/summary.json", base=folder))
    state = read(safe_path("data/" + episode_rel + "/state.json", base=folder))
    for key in ("task_id", "object_instance_id", "plan_id", "run_id", "physical_result", "autonomy_result"):
        if manifest.get(key) != summary.get(key):
            raise ValueError(f"Manifest/summary metadata mismatch: {key}")
        if key in state and state[key] != summary.get(key):
            raise ValueError(f"Archived summary/state mismatch: {key}")
    return {"ok": True, "bundle": str(folder), "files": len(entries),
            "status": manifest["status"], "physical_result": manifest.get("physical_result")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--episode", required=True)
    export.add_argument("--candidate", action="append", default=[])
    verify = commands.add_parser("verify")
    verify.add_argument("bundle")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            folder = export_episode(args.episode, args.candidate)
            print(json.dumps({"bundle": str(folder), "verification": verify_bundle(folder)}, ensure_ascii=False))
        else:
            print(json.dumps(verify_bundle(args.bundle), ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
