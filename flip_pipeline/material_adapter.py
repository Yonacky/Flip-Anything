"""Read-only adapter from real ds_control logs to reviewable experience events.

Commands and worker completion are not task success. Physical action claims need
hardware provenance, a paired command/result and a matching controller sample.
No phase/contact/teacher identity is inferred from filenames or whole-run labels.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from . import experience as ex
from .controller_boundary import controller_pose_to_matrix
from .storage import inside, identity


def _json(data):
    return json.loads(data.decode("utf-8-sig"))


def _timestamp(path):
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _path_name(root, path):
    return inside(root, path).relative_to(root).as_posix()


def _files(root, run):
    files, pending = [], [run]
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            if path.is_symlink() or (getattr(path.lstat(), "st_file_attributes", 0) & 0x400):
                raise ValueError("raw run contains a link/reparse point; use a real snapshot")
            inside(root, path)
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                files.append(path)
    return sorted(files)


def import_run(root, run_path, episode_id, calibration_id, annotations=None):
    """Return JSON-only {events, manifest, warnings, ...}; never edit source files.

    annotations: list of per-event records, or a workspace JSON path containing
    that list. Target is 'events.jsonl#L12' or 'result_...json'. Teacher attribution
    needs actor, signed_by and real evidence_refs for each target. Optional phase,
    contact, intervention, hypothesis and task_result create separate annotations.
    """
    root = Path(root).resolve()
    run = inside(root, run_path)
    identity(episode_id)
    identity(calibration_id)
    if not run.is_dir() or not (run / "events.jsonl").is_file():
        raise ValueError("run must contain events.jsonl")
    files = _files(root, run)
    initial = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
    cached, manifest_items = {}, {}

    def capture(path):
        path = inside(root, path)
        if not path.is_file():
            raise ValueError("missing evidence: " + str(path))
        name = _path_name(root, path)
        data = path.read_bytes()
        item = ex.evidence_manifest({name: data})["files"][0]
        if name in manifest_items and manifest_items[name] != item:
            raise ValueError("source changed during import")
        manifest_items[name] = item
        return name, data

    for path in files:
        name, data = capture(path)
        if path.suffix in {".json", ".jsonl"}:
            cached[name] = data
    annotations_path = None
    if isinstance(annotations, (str, Path)):
        annotations_path, data = capture(annotations)
        annotations = _json(data)
    annotations = [] if annotations is None else annotations
    if not isinstance(annotations, list):
        raise ValueError("annotations must be a list; whole-run actor replacement is forbidden")
    ex._reject_private(annotations)
    warnings, raw_events = [], []
    events_path = _path_name(root, run / "events.jsonl")
    for index, line in enumerate(cached[events_path].decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            ex._time(raw["utc"])
            if not isinstance(raw.get("kind"), str) or not isinstance(raw.get("payload"), dict):
                raise ValueError("missing payload")
            raw_events.append((index, raw))
        except (ValueError, KeyError, TypeError) as exc:
            warnings.append(f"events.jsonl#L{index}: unreadable event retained in raw evidence ({type(exc).__name__})")
    if not raw_events:
        raise ValueError("no readable timestamped source events")
    run_manifest = {}
    if (run / "manifest.json").is_file():
        run_manifest = _json(cached[_path_name(root, run / "manifest.json")])
    hardware = run_manifest.get("backend") == "hardware"
    if not hardware:
        warnings.append("hardware provenance missing; all robot motion remains unconfirmed")
    if run_manifest.get("status") not in {"completed", "failed", "aborted"}:
        warnings.append("run not marked closed; this is a stable snapshot, not a complete-run claim")
    samples = {}
    for index, raw in raw_events:
        if raw["kind"] == "robot_sample":
            sample = raw["payload"]
            ns = sample.get("sample_end_monotonic_ns")
            if isinstance(ns, int) and not isinstance(ns, bool):
                samples.setdefault(ns, []).append((index, raw))
    normalized, by_target = [], {}
    prefix = run.name

    def base(target, stamp, kind, source="proposal", refs=None, observations=None):
        return {"schema_version": 1, "event_id": prefix + ":" + target,
            "episode_id": episode_id, "attempt_id": prefix, "timestamp": stamp,
            "actor": "student", "source_kind": source, "kind": kind,
            "evidence_refs": list(refs or []), "observations": list(observations or []),
            "source_target": target}

    def camera_refs(payload):
        camera = payload.get("camera", {})
        refs = []
        if isinstance(camera, dict):
            for key in ("color_file", "depth_file"):
                if camera.get(key):
                    path = inside(run, camera[key])
                    name = _path_name(root, path)
                    if name in manifest_items:
                        refs.append(name)
                    else:
                        warnings.append("camera references missing file: " + name)
        return refs

    for index, raw in raw_events:
        target = f"events.jsonl#L{index}"
        kind = "observation" if raw["kind"] in {"observation", "robot_sample"} else "tool_result"
        e = base(target, raw["utc"], kind, refs=[events_path], observations=camera_refs(raw["payload"]))
        e["raw_kind"] = raw["kind"]
        e["payload"] = deepcopy(raw["payload"])
        e["host_monotonic_ns"] = raw.get("host_monotonic_ns")
        # Standalone samples and camera records are observations, not action proof.
        # Their source stays proposal until joined to an explicit hardware receipt.
        normalized.append(e)
        by_target[target] = e

    for path in sorted(run.glob("result_*.json")):
        target = path.name
        result_name = _path_name(root, path)
        try:
            result = _json(cached[result_name])
        except ValueError:
            warnings.append(target + ": invalid JSON retained as raw evidence")
            continue
        if not isinstance(result, dict):
            warnings.append(target + ": result is not an object")
            continue
        command_path = run / "commands" / target[len("result_"):]
        command_name = _path_name(root, command_path)
        try:
            request = _json(cached[command_name]) if command_name in cached else None
        except ValueError:
            request = None
            warnings.append(target + ": paired command is invalid JSON; execution unconfirmed")
        refs = [result_name, events_path] + ([command_name] if request is not None else [])
        state = result.get("state", {})
        if not isinstance(state, dict):
            state = {}
        sample_ns = state.get("sample_end_monotonic_ns")
        sample_candidates = samples.get(sample_ns, []) if isinstance(sample_ns, int) and not isinstance(sample_ns, bool) else []
        matches = [raw for _, raw in sample_candidates
                   if raw["payload"].get("pose") == state.get("pose")]
        match = matches[0] if matches else None
        stamp = match["utc"] if match else _timestamp(path)
        action = request.get("action") if isinstance(request, dict) else None
        e = base(target, stamp, "tool_result", refs=refs, observations=camera_refs(result))
        e.update(receipt=result, request=request, timestamp_source="controller_sample_event" if match else "file_mtime_not_execution_time")
        actual_pose = None
        if hardware and match and isinstance(request, dict):
            try:
                matrix = controller_pose_to_matrix(state["pose"]).tolist()
                actual_pose = {"matrix": matrix, "frame_id": "base", "units": "m", "source": "controller",
                    "timestamp": stamp, "calibration_id": calibration_id}
            except (ValueError, KeyError, TypeError) as exc:
                warnings.append(target + ": controller pose could not be normalized: " + str(exc))
        if actual_pose is not None:
            e["source_kind"] = "physical_execution"
            e["actual_pose"] = actual_pose
            e["kind"] = "action" if action in {"move", "grip"} else "observation"
            e["command_outcome"] = "success" if result.get("ok") is True else "failure" if result.get("ok") is False else "unknown"
            if result.get("ok") is not True and action in {"move", "grip"}:
                # A failed receipt may follow partial movement; do not claim full execution.
                e["kind"] = "observation"
                e["execution_extent"] = "unknown_or_partial"
        else:
            e["command_outcome"] = "unknown"
            e["execution_extent"] = "unconfirmed"
            if action in {"move", "grip"}:
                warnings.append(target + ": requested action lacks matching hardware pose/receipt evidence")
        normalized.append(e)
        by_target[target] = e
        if result.get("ok") is False:
            failure = base(target + ":failure", stamp, "result", e["source_kind"], refs=refs)
            failure.update(result="failure", result_scope="command", derived_from=target, text=str(result.get("error", "command failed")))
            normalized.append(failure)

    seen_targets, task_results = set(), []
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise ValueError("annotation must be an object")
        target = annotation.get("target")
        if target not in by_target or target in seen_targets:
            raise ValueError("annotation target must exist and be unique")
        seen_targets.add(target)
        original = by_target[target]
        actor = annotation.get("actor", "student")
        if actor not in ex.ACTORS:
            raise ValueError("invalid annotation actor")
        refs = annotation.get("evidence_refs", [])
        ex._refs(refs, "annotation evidence_refs")
        canonical_refs = []
        for ref in refs:
            name, _ = capture(ref)
            canonical_refs.append(name)
        if actor in {"human", "astra"}:
            ex._text(annotation.get("signed_by"), "teacher signed_by")
            if not canonical_refs:
                raise ValueError("each teacher attribution needs signed evidence")
            original["actor"] = actor
            original["attribution"] = {"signed_by": annotation["signed_by"], "evidence_refs": canonical_refs,
                "verification": "recorded attribution, not cryptographic identity verification"}
            original["evidence_refs"] = list(dict.fromkeys(original["evidence_refs"] + canonical_refs))
            for derived in normalized:
                if derived.get("derived_from") == target:
                    derived["actor"] = actor
                    derived["attribution"] = deepcopy(original["attribution"])
                    derived["evidence_refs"] = list(dict.fromkeys(derived["evidence_refs"] + canonical_refs))
        if "object_pose" in annotation:
            if original["source_kind"] != "physical_execution" or "actual_pose" not in original:
                raise ValueError("object-relative annotation needs an actual physical sample")
            if not canonical_refs:
                raise ValueError("object pose annotation needs measured evidence")
            ex.validate_pose(annotation["object_pose"])
            original["object_pose"] = deepcopy(annotation["object_pose"])
        for field, kind in (("phase", "phase"), ("contact", "contact"), ("intervention", "intervention"), ("hypothesis", "hypothesis"), ("task_result", "result")):
            if field not in annotation:
                continue
            if not canonical_refs:
                raise ValueError("annotations need actual evidence references")
            source = "proposal" if kind == "hypothesis" else original["source_kind"]
            note = base(f"annotation-{index}-{field}", original["timestamp"], kind, source,
                        refs=list(dict.fromkeys(original["evidence_refs"] + canonical_refs)), observations=original["observations"])
            note["actor"] = actor
            if actor != "student":
                note["attribution"] = deepcopy(original["attribution"])
            if field == "task_result":
                if source != "physical_execution":
                    raise ValueError("task result requires actual physical evidence, not a proposal")
                note.update(result=annotation[field], result_scope="object_task")
                task_results.append((ex._time(note["timestamp"]), annotation[field]))
            elif kind in {"phase", "contact"}:
                note[field] = annotation[field]
            else:
                note["text"] = annotation[field]
            # Phase/contact applies to this sample; sort before its action at equal time.
            note["annotation_target"] = original["event_id"]
            normalized.append(note)
    normalized.sort(key=lambda e: (ex._time(e["timestamp"]), 0 if e["kind"] in {"phase", "contact"} else 1,
                                   2 if e["kind"] == "result" else 0, e["event_id"]))
    normalized = ex.validate_trace(normalized)
    current_files = _files(root, run)
    final = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in current_files}
    if initial != final:
        raise ValueError("run changed during import; close the run or import a stable snapshot")
    # Verify extra annotation evidence has not changed while processing.
    for name, item in manifest_items.items():
        current = ex.evidence_manifest({name: inside(root, name).read_bytes()})["files"][0]
        if current != item:
            raise ValueError("evidence changed during import: " + name)
    items = [manifest_items[name] for name in sorted(manifest_items)]
    manifest = {"schema_version": 1, "files": items, "manifest_sha256": ex.digest(items)}
    return {"schema_version": 1, "run_id": run.name, "episode_id": episode_id,
        "events": normalized, "manifest": manifest, "source_files": [item["path"] for item in items],
        "warnings": warnings, "annotations_sha256": ex.digest(annotations), "annotations_path": annotations_path,
        "task_outcome": sorted(task_results, key=lambda x: x[0])[-1][1] if task_results else "unknown",
        "limitations": ["worker completed is not object-task success", "unobserved motion is not reconstructed",
                        "teacher attribution requires maintainer review", "snapshot hashes do not establish physical truth"]}
