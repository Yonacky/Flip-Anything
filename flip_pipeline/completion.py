"""Measured completion contract, independent of prose claims of success.

Checks provenance/order/numbers, not the correctness of an image recognizer.
Raw observations remain reviewable; failure never authorizes physical recovery.
"""
import math
from pathlib import Path
import numpy as np

from .calibration import validate_transform
from .controller_boundary import controller_pose_to_matrix
from .storage import read, digest, inside, verify_evidence


DEFAULT_LIMITS = {
    "max_inversion_error_rad": math.radians(10),
    "max_stable_translation_m": .005,
    "max_stable_rotation_rad": math.radians(5),
    "min_stability_interval_s": 1.0,
    "max_position_uncertainty_m": .002,
    "max_orientation_uncertainty_rad": .03,
    "max_gripper_register_error": 5,
    "max_support_to_release_s": 30,
}


def _number(value, name, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def _ns(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("sample/issue time must be positive integer monotonic nanoseconds")
    return value


def _angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def validate_completion(root, folder, state, data, refs):
    root, folder = Path(root).resolve(), inside(root, folder)
    if state.get("mode") != "experiment" or state.get("status") != "active" or state.get("inflight"):
        raise ValueError("Completion requires an active experiment with no unresolved action")
    run_id = data.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("Completion run_id required")
    verify_evidence(root, refs)
    indexed = {item["path"]: item for item in refs}

    def covered(path):
        resolved = inside(root, path)
        relative = resolved.relative_to(root).as_posix()
        if relative not in indexed:
            raise ValueError("Completion evidence must hash every source: " + relative)
        return resolved

    def covered_raw(items):
        verify_evidence(root, items)
        for item in items:
            covered(item["path"])
            if indexed[item["path"]] != item:
                raise ValueError("Nested evidence hash differs from completion evidence")

    plan_hash = state.get("artifacts", {}).get("plan")
    if not plan_hash:
        raise ValueError("No accepted plan")
    artifact = read(folder / "artifacts" / (plan_hash + ".json"))
    if digest(artifact) != plan_hash:
        raise ValueError("Plan artifact changed")
    plan = artifact["data"]
    segments = plan.get("segments", [])
    events = [item for item in state.get("history", []) if item.get("plan_hash") == plan_hash and item.get("event") in {"action_completed", "action_failed"}]
    if not segments or [item.get("segment_id") for item in events] != [item.get("id") for item in segments] or any(item.get("event") != "action_completed" for item in events):
        raise ValueError("Cannot claim completion without every planned action completed in order")
    if any(item.get("run_id") != run_id for item in events):
        raise ValueError("Completed actions belong to another run")
    limits = dict(DEFAULT_LIMITS)
    policy = read(inside(root, "devices/pipeline_policy.json"))
    overrides = policy.get("completion_limits", {})
    if not isinstance(overrides, dict) or set(overrides) - set(limits):
        raise ValueError("Unknown completion limits in maintenance policy")
    limits.update(overrides)
    for name, value in limits.items():
        if _number(value, name) <= 0:
            raise ValueError("Completion limits must be positive")

    tickets, completed_times, release_indices, ids = [], [], [], set()
    previous_end = 0
    for index, (segment, event) in enumerate(zip(segments, events)):
        ident = event.get("ticket_id")
        if not isinstance(ident, str) or not ident or ident in ids or Path(ident).name != ident:
            raise ValueError("Invalid/reused action ticket id")
        ids.add(ident)
        ticket = read(covered(folder / "tickets" / (ident + ".json")))
        if ticket.get("ticket_id") != ident or ticket.get("state") != "consumed" or ticket.get("plan_hash") != plan_hash or ticket.get("run_id") != run_id or ticket.get("segment_id") != segment["id"]:
            raise ValueError("Action ticket binding mismatch")
        if inside(root, ticket.get("task", "")) != folder:
            raise ValueError("Action ticket belongs to another task")
        reference = ticket.get("reference", [])
        if len(reference) != 1:
            raise ValueError("Action must bind one pre-action controller receipt")
        covered_raw(reference)
        reference_path = inside(root, reference[0]["path"])
        if reference_path.parent.name != run_id or not reference_path.name.startswith("result_"):
            raise ValueError("Action reference is not from this run")
        before = read(reference_path)
        if before.get("ok") is not True:
            raise ValueError("Action reference failed")
        before_state = before.get("state", {})
        controller_pose_to_matrix(before_state.get("pose"))
        request = dict(segment["request"], expected_pose=before_state["pose"])
        if ticket.get("command_hash") != digest(request):
            raise ValueError("Consumed command differs from plan/reference")
        result = event.get("result", {})
        if result.get("ok") is not True:
            raise ValueError("Completed action lacks successful physical receipt")
        result_state = result.get("state", {})
        controller_pose_to_matrix(result_state.get("pose"))
        end = _ns(result_state.get("sample_end_monotonic_ns"))
        start = _ns(before_state.get("sample_end_monotonic_ns"))
        issued = _ns(ticket.get("issued_monotonic_ns"))
        if not previous_end <= start <= issued <= end:
            raise ValueError("Action receipt times are out of order")
        previous_end = end
        completed_times.append(end)
        tickets.append(ticket)
        pre_grip = before_state.get("gripper") or {}
        opening = request.get("action") == "grip" and isinstance(pre_grip.get("gPO"), int) and request.get("pos", 256) < pre_grip["gPO"]
        if segment.get("role") == "release" or (opening and (pre_grip.get("gOBJ") in (1, 2) or request.get("grip_required"))):
            if request.get("action") != "grip" or not opening:
                raise ValueError("Release segment did not open the gripper")
            support = ticket.get("support_evidence")
            if not support or support.get("run_id") != run_id or support.get("method") not in {"observed_contact", "calibrated_contact_measurement"}:
                raise ValueError("Release ticket missing accepted support evidence")
            observation = read(covered(support["path"]))
            if digest(observation) != support["sha256"] or observation.get("supported") is not True:
                raise ValueError("Support observation changed or did not confirm support")
            if observation.get("run_id") != run_id or observation.get("object_id") != state["object_id"]:
                raise ValueError("Support belongs to another object/run")
            if observation.get("reference_receipt_sha256") != digest(before) or observation.get("completed_action_count") != index:
                raise ValueError("Support not tied to immediately preceding state")
            supported_at = _ns(observation.get("sample_monotonic_ns"))
            prior_end = completed_times[index - 1] if index else 0
            if not prior_end <= supported_at <= issued or end - supported_at > limits["max_support_to_release_s"] * 1e9:
                raise ValueError("Support was stale or acquired after release")
            covered_raw(observation.get("raw_evidence", []))
            after_grip = result_state.get("gripper") or {}
            actual = after_grip.get("gPO")
            if after_grip.get("gOBJ") != 3 or after_grip.get("gFLT") != 0 or not isinstance(actual, int) or isinstance(actual, bool) or abs(actual - request["pos"]) > limits["max_gripper_register_error"]:
                raise ValueError("Release did not reach commanded open position without obstruction")
            release_indices.append(index)
    if not release_indices:
        raise ValueError("No verified release in this plan")
    retreats = [index for index, segment in enumerate(segments) if index > release_indices[-1] and segment.get("role") == "retreat" and segment["request"].get("action") == "move"]
    if not retreats:
        raise ValueError("No completed retreat after final release")

    final_path = covered(data.get("final_receipt_path", ""))
    if final_path.parent.name != run_id or not final_path.name.startswith("result_"):
        raise ValueError("Final receipt not from this run")
    final = read(final_path)
    if final.get("ok") is not True or _ns(final.get("state", {}).get("sample_end_monotonic_ns")) < completed_times[-1]:
        raise ValueError("Final receipt failed or predates last action")
    final_pose = controller_pose_to_matrix(final["state"].get("pose"))
    end_pose = controller_pose_to_matrix(events[-1]["result"]["state"]["pose"])
    if np.linalg.norm(final_pose[:3, 3] - end_pose[:3, 3]) > .002 or _angle(end_pose[:3, :3].T @ final_pose[:3, :3]) > math.radians(2):
        raise ValueError("Robot moved after the recorded final action")
    final_grip = final["state"].get("gripper") or {}
    release_position = segments[release_indices[-1]]["request"]["pos"]
    actual_position = final_grip.get("gPO")
    if final_grip.get("gOBJ") != 3 or final_grip.get("gFLT") != 0 or not isinstance(actual_position, int) or isinstance(actual_position, bool) or abs(actual_position - release_position) > limits["max_gripper_register_error"]:
        raise ValueError("Final gripper feedback does not confirm continued release")
    observed = read(covered(data.get("completion_observations_path", "")))
    for key in ("object_id", "installation_id"):
        if observed.get(key) != state[key]:
            raise ValueError("Completion observation identity mismatch: " + key)
    if observed.get("run_id") != run_id or observed.get("calibration_artifact_sha256") != state["artifacts"].get("calibration"):
        raise ValueError("Completion observations use another run/calibration")
    if observed.get("frame") != "base" or observed.get("units") != "m_rad":
        raise ValueError("Completion poses must be base-frame SI geometry")
    axis = np.asarray(observed.get("axis_object"), dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis) - 1) > 1e-6 or not isinstance(observed.get("axis_label"), str) or not observed["axis_label"].strip():
        raise ValueError("Need a labeled directed object axis, not an unsigned cylinder axis")
    if not isinstance(observed.get("before"), dict) or not isinstance(observed.get("post_retreat"), list) or any(not isinstance(x, dict) for x in observed["post_retreat"]):
        raise ValueError("Before and post-retreat observations must be structured measurement records")
    measurements = [observed["before"]] + observed["post_retreat"]
    if len(measurements) < 3:
        raise ValueError("At least two independent post-retreat captures required")
    captures, times, poses, uncertainties, raw_paths = set(), [], [], [], set()
    for measurement in measurements:
        capture = measurement.get("capture_id")
        if not isinstance(capture, str) or not capture or capture in captures:
            raise ValueError("Completion captures must be distinct")
        captures.add(capture)
        if measurement.get("method") not in {"rgbd_model_fit", "tracked_keypoints", "teacher_measured"}:
            raise ValueError("Unsupported measured object-pose source")
        paths = measurement.get("raw_evidence", [])
        covered_raw(paths)
        current_paths = {item["path"] for item in paths}
        if raw_paths & current_paths:
            raise ValueError("Reusing one raw capture does not demonstrate stability")
        raw_paths.update(current_paths)
        times.append(_ns(measurement.get("sample_monotonic_ns")))
        poses.append(validate_transform(measurement.get("base_from_object")))
        pu = _number(measurement.get("position_uncertainty_m"), "position uncertainty", True)
        ru = _number(measurement.get("orientation_uncertainty_rad"), "orientation uncertainty", True)
        if pu > limits["max_position_uncertainty_m"] or ru > limits["max_orientation_uncertainty_rad"]:
            raise ValueError("Object pose uncertainty too large for completion decision")
        uncertainties.append((pu, ru))
    if times[0] > tickets[0]["issued_monotonic_ns"] or any(t < completed_times[-1] for t in times[1:]):
        raise ValueError("Before/after observations do not bracket the executed task")
    if any(a >= b for a, b in zip(times[1:], times[2:])) or times[-1] - times[1] < limits["min_stability_interval_s"] * 1e9:
        raise ValueError("Post-retreat captures need increasing times and a stability interval")
    initial_axis = poses[0][:3, :3] @ axis
    inversion_errors, translation_drift, rotation_drift = [], 0.0, 0.0
    for index in range(1, len(poses)):
        current_axis = poses[index][:3, :3] @ axis
        error = float(math.acos(float(np.clip(-initial_axis @ current_axis, -1, 1))))
        bounded = error + uncertainties[0][1] + uncertainties[index][1]
        if bounded > limits["max_inversion_error_rad"]:
            raise ValueError("Measured directed object axis did not invert within uncertainty")
        inversion_errors.append(bounded)
        for other in range(1, index):
            distance = float(np.linalg.norm(poses[index][:3, 3] - poses[other][:3, 3])) + uncertainties[index][0] + uncertainties[other][0]
            angle = _angle(poses[other][:3, :3].T @ poses[index][:3, :3]) + uncertainties[index][1] + uncertainties[other][1]
            translation_drift, rotation_drift = max(translation_drift, distance), max(rotation_drift, angle)
    if translation_drift > limits["max_stable_translation_m"] or rotation_drift > limits["max_stable_rotation_rad"]:
        raise ValueError("Object not independently stable within measured uncertainty")
    return {"status": "measured_completion_checked", "run_id": run_id, "completed_segments": len(events),
            "release_segment_ids": [segments[i]["id"] for i in release_indices],
            "max_inversion_error_bound_rad": max(inversion_errors), "translation_drift_bound_m": translation_drift,
            "rotation_drift_bound_rad": rotation_drift, "stability_interval_s": (times[-1] - times[1]) / 1e9,
            "quality_limits": limits, "scope": "numeric/provenance contract; not independent image-recognizer certification"}
