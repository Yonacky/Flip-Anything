"""Evidence preparation for maintainers, not autonomous skill promotion.

All functions are offline and deterministic. A hash detects edits, not truth.
Teacher segmentation is an annotation/contact baseline, not learned segmentation.
Pose matrices map local coordinates to frame_id in metres. No replay is executed.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import PurePosixPath
import re

ACTORS = {"human", "astra", "student"}
SOURCES = {"physical_execution", "proposal"}
KINDS = {"action", "observation", "phase", "contact", "result", "intervention",
         "hypothesis", "public_transcript", "code", "tool_result"}
RESULTS = {"success", "failure", "incomplete", "unknown"}
CONTACTS = {"free", "grasped", "supported", "released", "unknown"}
PRIVATE_KEYS = {"chain_of_thought", "hidden_reasoning", "reasoning_content", "private_reasoning"}


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _time(value):
    _text(value, "timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be ISO 8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp needs an explicit timezone")
    return parsed


def _number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _reject_private(value):
    if isinstance(value, dict):
        if PRIVATE_KEYS.intersection(value):
            raise ValueError("record public transcript and decision summaries, not hidden reasoning")
        for child in value.values():
            _reject_private(child)
    elif isinstance(value, list):
        for child in value:
            _reject_private(child)


def _refs(value, name):
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
        raise ValueError(f"{name} must be a list of artifact names")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicate references")


def validate_pose(pose):
    if not isinstance(pose, dict):
        raise ValueError("pose must be an object")
    for key in ("frame_id", "calibration_id"):
        _text(pose.get(key), key)
    _time(pose.get("timestamp"))
    if pose.get("units") != "m":
        raise ValueError("pose units must be m")
    m = pose.get("matrix")
    if not isinstance(m, list) or len(m) != 4 or any(not isinstance(r, list) or len(r) != 4 for r in m):
        raise ValueError("pose matrix must be 4x4")
    if any(not _number(x) for row in m for x in row):
        raise ValueError("pose must be finite numeric values")
    if any(abs(m[3][i] - (1 if i == 3 else 0)) > 1e-6 for i in range(4)):
        raise ValueError("invalid homogeneous last row")
    r = [row[:3] for row in m[:3]]
    for i in range(3):
        for j in range(3):
            if abs(sum(r[k][i] * r[k][j] for k in range(3)) - (i == j)) > 1e-5:
                raise ValueError("rotation must be orthonormal")
    det = (r[0][0] * (r[1][1]*r[2][2]-r[1][2]*r[2][1])
           -r[0][1] * (r[1][0]*r[2][2]-r[1][2]*r[2][0])
           +r[0][2] * (r[1][0]*r[2][1]-r[1][1]*r[2][0]))
    if abs(det - 1) > 1e-5:
        raise ValueError("rotation determinant must be +1")
    return deepcopy(pose)


def validate_event(event):
    if not isinstance(event, dict):
        raise ValueError("event must be an object")
    _reject_private(event)
    canonical_bytes(event)
    if event.get("schema_version") != 1:
        raise ValueError("unsupported event schema")
    for key in ("event_id", "episode_id", "attempt_id"):
        _text(event.get(key), key)
    _time(event.get("timestamp"))
    if event.get("actor") not in ACTORS or event.get("source_kind") not in SOURCES:
        raise ValueError("unknown actor or source_kind")
    if event.get("kind") not in KINDS:
        raise ValueError("unknown event kind")
    _refs(event.get("observations"), "observations")
    _refs(event.get("evidence_refs"), "evidence_refs")
    if "actual_pose" in event:
        validate_pose(event["actual_pose"])
    if "object_pose" in event:
        validate_pose(event["object_pose"])
    if event["source_kind"] == "proposal" and "actual_pose" in event:
        raise ValueError("proposal cannot claim an actual execution pose")
    if event["kind"] == "action" and event["source_kind"] == "physical_execution":
        if "actual_pose" not in event or not event["evidence_refs"]:
            raise ValueError("physical action needs actual pose and execution evidence")
        if event["actual_pose"].get("source") != "controller":
            raise ValueError("actual robot pose must come from controller")
    if event["kind"] == "result":
        if event.get("result_scope", "object_task") not in {"object_task", "command"}:
            raise ValueError("invalid result scope")
        if event.get("result") not in RESULTS:
            raise ValueError("invalid result")
        if event["source_kind"] == "physical_execution" and event["result"] in {"success", "failure"} and not event["evidence_refs"]:
            raise ValueError("physical result needs evidence")
    if event["kind"] == "contact":
        if event.get("contact") not in CONTACTS:
            raise ValueError("invalid contact state")
        if event["source_kind"] == "physical_execution" and not event["evidence_refs"]:
            raise ValueError("physical contact annotation needs evidence")
    if event["kind"] == "phase":
        _text(event.get("phase"), "phase")
    if event["kind"] in {"hypothesis", "public_transcript", "intervention"}:
        _text(event.get("text"), "text")
    return deepcopy(event)


def validate_trace(events):
    if not isinstance(events, list) or not events:
        raise ValueError("nonempty ordered event list required")
    clean = [validate_event(x) for x in events]
    ids = [x["event_id"] for x in clean]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate event_id")
    if len({x["episode_id"] for x in clean}) != 1:
        raise ValueError("one episode per trace; combine episodes through lineage")
    times = [_time(x["timestamp"]) for x in clean]
    if any(a > b for a, b in zip(times, times[1:])):
        raise ValueError("events must be in timestamp order")
    return clean


def _artifact_name(name):
    _text(name, "artifact name")
    path = PurePosixPath(name)
    if name == "." or "\\" in name or ":" in name or path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError("artifact names must be normalized relative POSIX paths")
    return name


def evidence_manifest(blobs):
    """Snapshot bytes supplied by collector. Caller must seal storage separately."""
    if not isinstance(blobs, dict) or not blobs:
        raise ValueError("nonempty evidence mapping required")
    if len({name.casefold() for name in blobs}) != len(blobs):
        raise ValueError("case-insensitive artifact path collision")
    files = []
    for name, data in sorted(blobs.items()):
        _artifact_name(name)
        if not isinstance(data, bytes):
            raise ValueError("evidence values must be bytes")
        files.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return {"schema_version": 1, "files": files, "manifest_sha256": digest(files)}


def _manifest_index(manifest):
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list) or not manifest["files"]:
        raise ValueError("invalid evidence manifest")
    if manifest.get("manifest_sha256") != digest(manifest["files"]):
        raise ValueError("manifest was changed")
    result = {}
    for item in manifest["files"]:
        name = _artifact_name(item["path"])
        if name in result or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
            raise ValueError("invalid or duplicate manifest item")
        if not isinstance(item.get("size"), int) or isinstance(item["size"], bool) or item["size"] < 0:
            raise ValueError("invalid artifact size")
        result[name] = item
    return result


def verify_manifest(manifest, blobs):
    _manifest_index(manifest)
    if evidence_manifest(blobs) != manifest:
        raise ValueError("evidence changed, missing, or unexpectedly added")
    return True


def _relative_waypoint(event, max_sync_seconds=0.2):
    a, o = event.get("actual_pose"), event.get("object_pose")
    if not a or not o or o.get("metric") is not True or o.get("quality") != "validated":
        return None, "missing validated metric object pose"
    if a["frame_id"] != o["frame_id"] or a["calibration_id"] != o["calibration_id"]:
        return None, "different coordinate frame or calibration"
    if any(abs((_time(a["timestamp"])-_time(t)).total_seconds()) > max_sync_seconds
           for t in (o["timestamp"], event["timestamp"])):
        return None, "pose samples not synchronized"
    am, om = a["matrix"], o["matrix"]
    # inv(T_world_object) @ T_world_tcp; proper SE(3) validated on ingestion.
    rel = [[0.0] * 4 for _ in range(4)]
    for i in range(3):
        for j in range(3):
            rel[i][j] = sum(om[k][i] * am[k][j] for k in range(3))
        rel[i][3] = sum(om[k][i] * (am[k][3] - om[k][3]) for k in range(3))
    rel[3][3] = 1.0
    return {"event_id": event["event_id"], "T_object_tcp": rel, "units": "m",
            "calibration_id": a["calibration_id"], "evidence_refs": event["evidence_refs"]}, None


def _base_draft(events, manifest, kind, scope):
    clean = validate_trace(events)
    index = _manifest_index(manifest)
    for event in clean:
        if any(x not in index for x in event["evidence_refs"] + event["observations"]):
            raise ValueError("event references missing evidence")
    _validate_scope(scope)
    return clean, {"schema_version": 1, "status": "draft", "kind": kind,
        "record_id": digest({"trace": clean, "manifest": manifest, "kind": kind})[:24],
        "scope": deepcopy(scope), "lineage": {"episode_id": clean[0]["episode_id"],
            "trace_sha256": digest(clean), "source_event_ids": [e["event_id"] for e in clean]},
        "evidence": deepcopy(manifest), "raw_events": clean, "review": None,
        "replay_allowed": False, "teacher_infallible": False,
        "interpretation": "deterministic preparation; requires maintainer review"}


def process_teacher(events, manifest, *, scope):
    """Segment explicit phase/contact annotations; never infer successful contact."""
    clean, draft = _base_draft(events, manifest, "teacher_demonstration", scope)
    physical = [e for e in clean if e["actor"] in {"human", "astra"} and e["source_kind"] == "physical_execution"]
    draft["teacher_proposals"] = [e["event_id"] for e in clean if e["actor"] in {"human", "astra"} and e["source_kind"] == "proposal"]
    draft["demonstration_available"] = any(e["kind"] == "action" for e in physical)
    segments, state, current = [], {}, None
    for event in physical:
        key = (event["attempt_id"], event["actor"])
        phase, contact = state.get(key, ("unannotated", "unknown"))
        if event["kind"] == "phase":
            phase = event["phase"]
        if event["kind"] == "contact":
            contact = event["contact"]
        state[key] = (phase, contact)
        signature = (key, phase, contact)
        if current is None or current["signature"] != signature or event["kind"] == "intervention":
            current = {"signature": signature, "actor": key[1], "attempt_id": key[0],
                "phase": phase, "contact": contact, "event_ids": [], "waypoints": [],
                "nontransferable_samples": [], "reported_result": "unknown"}
            segments.append(current)
        current["event_ids"].append(event["event_id"])
        if event["kind"] == "action":
            waypoint, reason = _relative_waypoint(event)
            if waypoint:
                current["waypoints"].append(waypoint)
            else:
                current["nontransferable_samples"].append({"event_id": event["event_id"], "reason": reason})
        if event["kind"] == "result":
            current["reported_result"] = event["result"]
    for segment in segments:
        segment.pop("signature")
    draft["segments"] = segments
    draft["segmentation_method"] = "explicit phase/contact transitions; no learned motion inference"
    draft["transfer_requirements"] = ["new-scene localization", "current calibration", "collision and reachability checks", "fresh support and grasp checks"]
    return draft


def process_student(events, manifest, *, scope):
    clean, draft = _base_draft(events, manifest, "student_trial", scope)
    attempts = {}
    for event in clean:
        entry = attempts.setdefault(event["attempt_id"], {"attempt_id": event["attempt_id"],
            "event_ids": [], "failure_prefixes": [], "interventions": [], "reported_result": "unknown"})
        entry["event_ids"].append(event["event_id"])
        if event["kind"] == "intervention":
            entry["interventions"].append(event["event_id"])
        if event["kind"] == "result" and event["source_kind"] == "physical_execution":
            if event.get("result_scope", "object_task") == "object_task":
                entry["reported_result"] = event["result"]
            if event["result"] == "failure":
                entry["failure_prefixes"].append(list(entry["event_ids"]))
    draft["attempts"] = list(attempts.values())
    draft["retry_count"] = max(len(attempts) - 1, 0)
    draft["causal_hypotheses"] = [{"event_id": e["event_id"], "text": e["text"],
        "status": "unverified", "evidence_refs": e["evidence_refs"]}
        for e in clean if e["kind"] == "hypothesis"]
    draft["autonomous"] = not any(e["actor"] != "student" or e["kind"] == "intervention" for e in clean)
    draft["all_attempts_retained"] = True
    return draft


def _validate_scope(scope):
    if not isinstance(scope, dict) or not {"goal", "object_family", "tool_revision"}.issubset(scope):
        raise ValueError("scope requires goal, object_family, tool_revision")
    for key, values in scope.items():
        _text(key, "scope key")
        if not isinstance(values, list) or not values or any(not isinstance(x, (str, bool, int, float)) or x in ("", "unknown", "*") for x in values):
            raise ValueError("scope values must be explicit nonempty allowed lists")
    canonical_bytes(scope)


def teacher_request(*, request_id, teacher, missing_information, question,
                    requested_segment, state_snapshot, evidence_refs, timestamp,
                    adapter_connected=False):
    if teacher not in {"human", "astra"}:
        raise ValueError("teacher must be human or astra")
    _text(request_id, "request_id")
    _text(question, "question")
    _text(requested_segment, "requested_segment")
    _refs(missing_information, "missing_information")
    if not missing_information or not isinstance(state_snapshot, dict) or not state_snapshot:
        raise ValueError("missing information and current state are required")
    _refs(evidence_refs, "evidence_refs")
    _time(timestamp)
    _reject_private(state_snapshot)
    result = {"schema_version": 1, "request_id": request_id, "teacher": teacher,
        "missing_information": list(missing_information), "question": question,
        "requested_segment": requested_segment, "state_snapshot": deepcopy(state_snapshot),
        "evidence_refs": list(evidence_refs), "timestamp": timestamp, "status": "pending",
        "adapter_connected": bool(adapter_connected), "sent": False,
        "next_action": "dispatch_via_configured_adapter" if adapter_connected else "manual_handoff_required",
        "instruction": "A textual suggestion is a proposal; physical demonstration needs recorded actions and observations."}
    canonical_bytes(result)
    return result


def _maintainer(name, role, authorized_maintainers):
    _text(name, "maintainer")
    if (role != "maintainer" or name.casefold() in {"student", "ds", "dsh"}
            or not isinstance(authorized_maintainers, (list, tuple, set))
            or name not in authorized_maintainers):
        raise ValueError("explicit registered maintainer required")


def review_record(draft, *, reviewer, reviewer_role, authorized_maintainers, evidence_refs, decision, blobs, notes):
    """Maintainer API. Caller must enforce authorization; this is not authentication."""
    _maintainer(reviewer, reviewer_role, authorized_maintainers)
    if draft.get("status") != "draft":
        raise ValueError("only a maintainer reviews a draft")
    _text(reviewer, "reviewer")
    _text(notes, "notes")
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject")
    verify_manifest(draft["evidence"], blobs)
    _refs(evidence_refs, "evidence_refs")
    if not evidence_refs or not set(evidence_refs).issubset(_manifest_index(draft["evidence"])):
        raise ValueError("review needs actual evidence references")
    if digest(draft["raw_events"]) != draft["lineage"]["trace_sha256"]:
        raise ValueError("source trace changed")
    _validate_scope(draft["scope"])
    result = deepcopy(draft)
    result["status"] = "reviewed" if decision == "approve" else "rejected"
    result["review"] = {"reviewer": reviewer, "role": reviewer_role, "decision": decision,
        "evidence_refs": list(evidence_refs), "notes": notes, "draft_sha256": digest(draft)}
    result["reviewed_sha256"] = digest(result)
    return result


def _verified_review(record):
    expected = record.get("reviewed_sha256")
    data = deepcopy(record)
    for key in ("reviewed_sha256", "release", "revocation"):
        data.pop(key, None)
    data["status"] = "reviewed"
    if expected != digest(data) or data.get("review", {}).get("decision") != "approve":
        raise ValueError("reviewed content changed or approval missing")


def release_record(reviewed, *, maintainer, maintainer_role, authorized_maintainers, blobs):
    _maintainer(maintainer, maintainer_role, authorized_maintainers)
    if reviewed.get("status") != "reviewed":
        raise ValueError("explicit maintainer release of reviewed record required")
    _text(maintainer, "maintainer")
    _verified_review(reviewed)
    verify_manifest(reviewed["evidence"], blobs)
    result = deepcopy(reviewed)
    result["status"] = "released"
    result["release"] = {"maintainer": maintainer, "reviewed_sha256": reviewed["reviewed_sha256"]}
    return result


def revoke_record(record, *, maintainer, maintainer_role, authorized_maintainers, reason):
    _maintainer(maintainer, maintainer_role, authorized_maintainers)
    if record.get("status") not in {"released", "reviewed"}:
        raise ValueError("maintainer may revoke a reviewed or released record")
    _text(maintainer, "maintainer")
    _text(reason, "reason")
    result = deepcopy(record)
    result["status"] = "revoked"
    result["revocation"] = {"maintainer": maintainer, "reason": reason}
    return result


def retrieve(records, context, query_tokens=(), limit=5):
    """Scope is a hard filter before token overlap; result is reference, never authorization."""
    if not isinstance(context, dict) or not isinstance(limit, int) or limit < 1:
        raise ValueError("context and positive limit required")
    matches, tokens = [], {str(x).lower() for x in query_tokens}
    revoked_ids = {r.get("record_id") for r in records if r.get("status") == "revoked"}
    for record in records:
        if record.get("status") != "released" or record.get("record_id") in revoked_ids:
            continue
        try:
            _verified_review(record)
            _validate_scope(record["scope"])
        except (ValueError, KeyError, TypeError):
            continue
        if any(key not in context or context[key] not in allowed for key, allowed in record["scope"].items()):
            continue
        words = set(re.findall(r"\w+", str(record.get("description", "")).lower()))
        matches.append((len(tokens & words), record))
    matches.sort(key=lambda pair: (-pair[0], pair[1]["record_id"]))
    return [{"record": deepcopy(record), "similarity_score": score, "use": "planning_reference_only"}
            for score, record in matches[:limit]]
