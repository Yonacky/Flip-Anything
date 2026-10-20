"""Offline, metre/radian calibration utilities. No hardware connection or motion.

Transforms use column vectors: ``a_from_b @ point_b == point_a``.
OpenCV's eye-in-hand inputs are base_from_tool and camera_from_target;
the output is tool_from_camera. A solver result is always a *draft*.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from itertools import combinations, product
import math
import re

import cv2
import numpy as np


DEFAULT_LIMITS = {
    "max_pose_image_skew_s": 0.03,
    "max_reprojection_error_px": 1.0,
    "min_train_samples": 6,
    "min_validation_samples": 3,
    "min_rotation_span_rad": 0.15,
    "min_axis_singular_ratio": 0.05,
    "max_translation_condition": 100.0,
    "max_sample_translation_m": 5.0,
    "max_handeye_translation_m": 1.0,
    "max_train_translation_error_m": 0.003,
    "max_train_rotation_error_rad": 0.02,
    "max_validation_translation_error_m": 0.005,
    "max_validation_rotation_error_rad": 0.03,
}
FRAME_CONVENTION = {"robot_pose": "base_from_tool", "target_pose": "camera_from_target"}
UNITS = {"translation": "m", "rotation": "rad"}
INSTALLATION_KEYS = ("installation_id", "arm_serial", "camera_serial", "tool_id", "mount_revision", "tcp_config_sha256")


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"{name} outside allowed range")
    return float(value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _array(value, shape, name):
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def validate_transform(matrix, name="transform"):
    """Return a checked 4x4 rigid transform; reject reflections/scales/NaN."""
    T = _array(matrix, (4, 4), name)
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError(f"{name}: invalid homogeneous bottom row")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(R), 1, atol=1e-6):
        raise ValueError(f"{name}: rotation must be in SO(3)")
    return T.copy()


def inverse_transform(matrix):
    T = validate_transform(matrix)
    result = np.eye(4)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ T[:3, 3]
    return result


def _rotation_angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _limits(request):
    supplied = request.get("quality_limits", {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_LIMITS):
        raise ValueError("quality_limits has unknown entries")
    result = dict(DEFAULT_LIMITS, **supplied)
    for key, value in result.items():
        _number(value, key, positive=True)
    for key, minimum in (("min_train_samples", 3), ("min_validation_samples", 3)):
        if int(result[key]) != result[key] or result[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if result["min_axis_singular_ratio"] >= 1:
        raise ValueError("min_axis_singular_ratio must be < 1")
    return result


def _load_samples(request, limits):
    if request.get("units") != UNITS or request.get("frame_convention") != FRAME_CONVENTION:
        raise ValueError("explicit metre/radian units and base_from_tool/camera_from_target frame convention required")
    if request.get("tool_frame") not in {"controller_tcp", "flange"}:
        raise ValueError("tool_frame must explicitly identify controller_tcp or flange")
    _text(request.get("installation_id"), "installation_id")
    _text(request.get("target_id"), "target_id")
    if not re.fullmatch(r"[0-9a-f]{64}", str(request.get("camera_intrinsics_sha256", ""))):
        raise ValueError("camera_intrinsics_sha256 must bind the intrinsics used for target poses")
    _text(request.get("target_geometry_evidence_ref"), "target_geometry_evidence_ref")
    if request.get("target_fixed_in_base") is not True:
        raise ValueError("calibration target must remain fixed in the base frame")
    if request.get("split_policy") != "predeclared":
        raise ValueError("training/validation split must be predeclared, before fitting")
    samples = request.get("samples")
    if not isinstance(samples, list):
        raise ValueError("samples must be a list")
    seen = {key: set() for key in ("sample_id", "capture_id", "evidence_ref")}
    records = []
    for item in samples:
        if not isinstance(item, dict):
            raise ValueError("sample must be an object")
        for key in seen:
            identifier = _text(item.get(key), key)
            if identifier in seen[key]:
                raise ValueError(f"duplicate {key}: validation cannot reuse a capture/evidence")
            seen[key].add(identifier)
        if item.get("split") not in {"train", "validation"}:
            raise ValueError("each sample must have split=train or validation")
        if item.get("robot_stationary") is not True:
            raise ValueError("hand-eye samples require stationary robot and matched exposure/pose")
        if item.get("timestamp_clock") != "shared_monotonic":
            raise ValueError("timestamps must be mapped into the same monotonic clock")
        image_t = _number(item.get("image_timestamp_s"), "image_timestamp_s", nonnegative=True)
        pose_t = _number(item.get("pose_timestamp_s"), "pose_timestamp_s", nonnegative=True)
        if abs(image_t - pose_t) > limits["max_pose_image_skew_s"]:
            raise ValueError("pose/image timestamp skew exceeds limit")
        error = _number(item.get("reprojection_error_px"), "reprojection_error_px", nonnegative=True)
        if error > limits["max_reprojection_error_px"]:
            raise ValueError("target reprojection error exceeds limit")
        G = validate_transform(item.get("base_from_tool"), "base_from_tool")
        C = validate_transform(item.get("camera_from_target"), "camera_from_target")
        if max(np.linalg.norm(G[:3, 3]), np.linalg.norm(C[:3, 3])) > limits["max_sample_translation_m"]:
            raise ValueError("sample translation exceeds metre scale guard; check units")
        records.append((item, G, C))
    train = [x for x in records if x[0]["split"] == "train"]
    validation = [x for x in records if x[0]["split"] == "validation"]
    if len(train) < limits["min_train_samples"] or len(validation) < limits["min_validation_samples"]:
        raise ValueError("insufficient separate training/validation samples")
    for (_, G, C), (_, H, D) in combinations(records, 2):
        if np.allclose(G, H, atol=1e-8, rtol=0) and np.allclose(C, D, atol=1e-8, rtol=0):
            raise ValueError("duplicate pose pair: renaming a sample does not make independent evidence")
    return train, validation


def _diversity(samples, limits):
    vectors, system = [], []
    for (_, G, _), (_, H, _) in combinations(samples, 2):
        relative = inverse_transform(H) @ G
        vectors.append(cv2.Rodrigues(relative[:3, :3])[0].reshape(3))
        system.append(relative[:3, :3] - np.eye(3))
    vectors = np.asarray(vectors)
    singular = np.linalg.svd(vectors, compute_uv=False)
    if np.linalg.norm(vectors, axis=1).max() < limits["min_rotation_span_rad"]:
        raise ValueError("insufficient rotation span for hand-eye calibration")
    if len(singular) < 2 or singular[1] / singular[0] < limits["min_axis_singular_ratio"]:
        raise ValueError("degenerate rotation axes: need nonparallel rotations")
    s = np.linalg.svd(np.vstack(system), compute_uv=False)
    condition = float(s[0] / s[-1]) if s[-1] > 1e-12 else math.inf
    if condition > limits["max_translation_condition"]:
        raise ValueError("translation system is rank deficient or ill conditioned")
    return {"rotation_axis_singular_values": singular.tolist(), "translation_condition": condition,
            "max_relative_rotation_rad": float(np.linalg.norm(vectors, axis=1).max())}


def _mean_transform(transforms):
    U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in transforms))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(U @ Vt)
    mean = np.eye(4)
    mean[:3, :3] = U @ correction @ Vt
    mean[:3, 3] = np.mean([T[:3, 3] for T in transforms], axis=0)
    return mean


def _residuals(samples, X, target, limits, split):
    rows = []
    for item, G, C in samples:
        reconstructed = G @ X @ C
        translation_error = float(np.linalg.norm(reconstructed[:3, 3] - target[:3, 3]))
        rotation_error = _rotation_angle(target[:3, :3].T @ reconstructed[:3, :3])
        rows.append({"sample_id": item["sample_id"], "translation_error_m": translation_error,
                     "rotation_error_rad": rotation_error})
    maximum_t = max(x["translation_error_m"] for x in rows)
    maximum_r = max(x["rotation_error_rad"] for x in rows)
    if maximum_t > limits[f"max_{split}_translation_error_m"] or maximum_r > limits[f"max_{split}_rotation_error_rad"]:
        raise ValueError(f"{split} fixed-target consistency failed: {maximum_t:.6g} m, {maximum_r:.6g} rad")
    return {"max_translation_error_m": maximum_t, "max_rotation_error_rad": maximum_r, "samples": rows}


def solve_hand_eye(request):
    """Solve eye-in-hand with OpenCV Park; validate against *unfitted* poses.

    The stationary target's base pose is estimated using training data only.
    Validation poses never enter calibrateHandEye or the reference target mean.
    Passing this check is necessary, not a physical safety certificate.
    """
    limits = _limits(request)
    train, validation = _load_samples(request, limits)
    diversity = _diversity(train, limits)
    validation_diversity = _diversity(validation, limits)
    try:
        R, t = cv2.calibrateHandEye([G[:3, :3] for _, G, _ in train], [G[:3, 3] for _, G, _ in train],
                                   [C[:3, :3] for _, _, C in train], [C[:3, 3] for _, _, C in train],
                                   method=cv2.CALIB_HAND_EYE_PARK)
    except cv2.error as exc:
        raise ValueError(f"OpenCV hand-eye solve failed: {exc}") from exc
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, t.reshape(3)
    X = validate_transform(X, "solved tool_from_camera")
    if np.linalg.norm(X[:3, 3]) > limits["max_handeye_translation_m"]:
        raise ValueError("solved tool_from_camera exceeds tool translation bound")
    base_from_target = _mean_transform([G @ X @ C for _, G, C in train])
    training_result = _residuals(train, X, base_from_target, limits, "train")
    validation_result = _residuals(validation, X, base_from_target, limits, "validation")
    return {"schema_version": 1, "status": "draft", "method": "opencv.calibrateHandEye.PARK",
            "opencv_version": cv2.__version__, "units": dict(UNITS), "frame_convention": dict(FRAME_CONVENTION),
            "tool_frame": request["tool_frame"],
            "installation_id": request["installation_id"], "target_id": request["target_id"],
            "camera_intrinsics_sha256": request["camera_intrinsics_sha256"],
            "tool_from_camera": X.tolist(), "base_from_target_training_estimate": base_from_target.tolist(),
            "training": training_result, "validation": validation_result,
            "diversity": {"train": diversity, "validation": validation_diversity},
            "train_sample_ids": [x[0]["sample_id"] for x in train],
            "validation_sample_ids": [x[0]["sample_id"] for x in validation],
            "evidence_refs": [x[0]["evidence_ref"] for x in train + validation],
            "quality_limits": limits, "input_sha256": _fingerprint(request),
            "validation_scope": "held-out fixed-target consistency; not independent absolute metrology"}


def _camera_parameters(profile):
    if profile.get("distortion_model") != "opencv_brown_conrady":
        raise ValueError("unsupported distortion model; convert through verified SDK mapping")
    for key in ("width", "height"):
        n = profile.get(key)
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError(f"invalid stream {key}")
    K = _array(profile.get("camera_matrix"), (3, 3), "camera_matrix")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError("invalid camera_matrix")
    if abs(K[0, 1]) > 1e-9 or abs(K[1, 0]) > 1e-9:
        raise ValueError("unsupported skew in camera_matrix")
    if not (0 <= K[0, 2] < profile["width"] and 0 <= K[1, 2] < profile["height"]):
        raise ValueError("principal point outside stream")
    distortion = np.asarray(profile.get("distortion_coefficients"), dtype=float)
    if distortion.ndim != 1 or len(distortion) not in {4, 5, 8, 12, 14} or not np.isfinite(distortion).all():
        raise ValueError("invalid Brown-Conrady distortion coefficients")
    return K, distortion


def estimate_target_pose(request):
    """Known target corners -> camera_from_target using mature OpenCV solvers.

    Board coordinates must come from a measured rigid target specification.
    Planar IPPE alternatives are returned in diagnostics and ambiguous results
    are rejected; a small training residual alone cannot resolve pose ambiguity.
    """
    if request.get("units") != UNITS:
        raise ValueError("target geometry must use metres/radians")
    _text(request.get("target_geometry_evidence_ref"), "target_geometry_evidence_ref")
    _text(request.get("image_evidence_ref"), "image_evidence_ref")
    intrinsics = request.get("intrinsics", {})
    K, distortion = _camera_parameters(intrinsics)
    points = np.asarray(request.get("object_points_m"), dtype=float)
    image = np.asarray(request.get("image_points_px"), dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 6 or not np.isfinite(points).all():
        raise ValueError("object_points_m must contain at least six finite 3D corners")
    if image.shape != (len(points), 2) or not np.isfinite(image).all():
        raise ValueError("image_points_px must match target corners")
    if np.any(image < 0) or np.any(image[:, 0] >= intrinsics["width"]) or np.any(image[:, 1] >= intrinsics["height"]):
        raise ValueError("target corners outside image")
    if len(np.unique(points, axis=0)) != len(points) or len(np.unique(image, axis=0)) != len(image):
        raise ValueError("duplicate target/image corners")
    if np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-9) < 2:
        raise ValueError("target corners are collinear")
    limits = {"max_reprojection_error_px": 1.0, "min_image_span_px": 30.0,
              "min_planar_alternative_gap_px": 0.25}
    supplied = request.get("quality_limits", {})
    if not isinstance(supplied, dict) or set(supplied) - set(limits):
        raise ValueError("unknown target pose quality limits")
    limits.update(supplied)
    for key, value in limits.items():
        _number(value, key, positive=True)
    if np.min(np.ptp(image, axis=0)) < limits["min_image_span_px"]:
        raise ValueError("target occupies too few image pixels")
    planar = np.max(abs(points[:, 2])) < 1e-9
    if not planar and np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-9) == 2:
        raise ValueError("planar target coordinates must define their plane as z=0")
    method = cv2.SOLVEPNP_IPPE if planar else cv2.SOLVEPNP_SQPNP
    try:
        solved = cv2.solvePnPGeneric(points, image, K, distortion, flags=method)
    except cv2.error as exc:
        raise ValueError(f"OpenCV target pose solve failed: {exc}") from exc
    candidates = []
    if solved[0]:
        for rvec, tvec in zip(solved[1], solved[2]):
            R = cv2.Rodrigues(rvec)[0]
            camera_points = points @ R.T + tvec.reshape(3)
            if not np.isfinite(camera_points).all() or np.min(camera_points[:, 2]) <= 0:
                continue
            projected = cv2.projectPoints(points, rvec, tvec, K, distortion)[0].reshape(-1, 2)
            rms = float(np.sqrt(np.mean(np.sum((image - projected) ** 2, axis=1))))
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, tvec.reshape(3)
            candidates.append((rms, validate_transform(T)))
    if not candidates:
        raise ValueError("no finite target pose with positive depth")
    candidates.sort(key=lambda item: item[0])
    if candidates[0][0] > limits["max_reprojection_error_px"]:
        raise ValueError("target reprojection error exceeds limit")
    if planar and len(candidates) > 1 and candidates[1][0] - candidates[0][0] < limits["min_planar_alternative_gap_px"]:
        raise ValueError("planar target pose is ambiguous; change observation angle")
    return {"camera_from_target": candidates[0][1].tolist(), "reprojection_error_px": candidates[0][0],
            "method": "opencv.solvePnP.IPPE" if planar else "opencv.solvePnP.SQPNP",
            "alternative_reprojection_errors_px": [item[0] for item in candidates[1:]],
            "quality_limits": limits, "units": dict(UNITS), "input_sha256": _fingerprint(request),
            "intrinsics_sha256": _fingerprint(intrinsics),
            "evidence_refs": [request["target_geometry_evidence_ref"], request["image_evidence_ref"]]}


def validate_factory_intrinsics(profile, observed_stream, verification):
    """Check SDK intrinsics binding and independent depth-scale check records.

    This checks configuration and measurement records, not the existence or
    authenticity of referenced artifacts; the project evidence store does that.
    It does not certify optical distortion over the full image.
    """
    for key in ("camera_serial", "stream_id", "alignment", "distortion_model"):
        if _text(profile.get(key), key) != _text(observed_stream.get(key), f"observed {key}"):
            raise ValueError(f"camera/stream identity mismatch: {key}")
    for key in ("width", "height"):
        n = profile.get(key)
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0 or n != observed_stream.get(key):
            raise ValueError(f"stream {key} mismatch")
    K, distortion = _camera_parameters(profile)
    current_K, current_distortion = _camera_parameters(observed_stream)
    if not np.allclose(K, current_K, rtol=1e-9, atol=1e-9) or distortion.shape != current_distortion.shape or not np.allclose(distortion, current_distortion, rtol=1e-9, atol=1e-9):
        raise ValueError("intrinsics differ from current SDK stream")
    scale = _number(profile.get("depth_scale_m_per_unit"), "depth_scale_m_per_unit", positive=True)
    if not np.isclose(scale, _number(observed_stream.get("depth_scale_m_per_unit"), "observed depth scale", positive=True), rtol=1e-9, atol=0):
        raise ValueError("depth scale differs from current SDK stream")
    _text(profile.get("sdk_evidence_ref"), "sdk_evidence_ref")
    _text(verification.get("evidence_ref"), "verification evidence_ref")
    if verification.get("independent_reference") is not True:
        raise ValueError("depth scale needs an independent measured reference")
    checks = verification.get("depth_checks", [])
    if not isinstance(checks, list) or len(checks) < 2:
        raise ValueError("at least two independent depth checks required")
    maximum_error = _number(verification.get("max_depth_error_m"), "max_depth_error_m", positive=True)
    reference_depths, errors = [], []
    for check in checks:
        reference = _number(check.get("reference_camera_z_m"), "reference_camera_z_m", positive=True)
        predicted = _number(check.get("raw_depth_units"), "raw_depth_units", positive=True) * scale
        reference_depths.append(reference)
        errors.append(abs(predicted - reference))
    if max(reference_depths) - min(reference_depths) <= 2 * maximum_error:
        raise ValueError("depth checks need distinct reference distances")
    if max(errors) > maximum_error:
        raise ValueError("depth scale check failed")
    return {"status": "configuration_checked", "camera_serial": profile["camera_serial"],
            "stream_id": profile["stream_id"], "max_depth_error_m": max(errors),
            "evidence_refs": [profile["sdk_evidence_ref"], verification["evidence_ref"]],
            "scope": "SDK binding and reported depth-scale checks; not optical intrinsic recalibration"}


def validate_tool_envelopes(boxes):
    """Validate measured rigid boxes. Every box is expressed in tool frame.

    A gripper envelope must state the opening interval it conservatively covers.
    Later geometry consumers must use only boxes applicable to current opening.
    """
    if not isinstance(boxes, list) or not boxes:
        raise ValueError("nonempty tool_envelopes required")
    result, names = [], set()
    for box in boxes:
        name = _text(box.get("id"), "box id")
        if name in names:
            raise ValueError("duplicate box id")
        names.add(name)
        if box.get("frame") != "tool" or box.get("units") != "m":
            raise ValueError("tool envelope must use tool frame and metres")
        T = validate_transform(box.get("tool_from_box"), "tool_from_box")
        half_extents = _array(box.get("half_extents_m"), (3,), "half_extents_m")
        if np.any(half_extents <= 0):
            raise ValueError("box half extents must be positive")
        _text(box.get("measurement_evidence_ref"), "measurement_evidence_ref")
        margin = _number(box.get("uncertainty_margin_m"), "uncertainty_margin_m", nonnegative=True)
        if box.get("component") not in {"camera", "mount", "gripper", "other"}:
            raise ValueError("unknown tool component")
        if box["component"] == "gripper":
            interval = _array(box.get("opening_range_m"), (2,), "opening_range_m")
            if interval[0] < 0 or interval[1] < interval[0]:
                raise ValueError("invalid gripper opening range")
        corners = np.asarray(list(product((-1, 1), repeat=3))) * (half_extents + margin)
        tool_corners = corners @ T[:3, :3].T + T[:3, 3]
        checked = deepcopy(box)
        checked["corners_tool_m"] = tool_corners.tolist()
        result.append(checked)
    return result


def profile_content_sha256(profile):
    """Digest geometry/configuration excluding mutable review metadata."""
    return _fingerprint({key: value for key, value in profile.items() if key not in {"status", "review"}})


def validate_device_profile(profile, current_installation, require_verified=True, task_profile=None):
    """Verify identity, geometry and teacher review binding; never promote itself.

    A signature/authenticated reviewer is an application responsibility. This is
    a data contract, not an authorization boundary against malicious code.
    """
    if profile.get("status") not in {"draft", "verified"} or profile.get("units") != UNITS:
        raise ValueError("profile needs draft/verified status and explicit metre/radian units")
    installation = profile.get("installation", {})
    for key in INSTALLATION_KEYS:
        if _text(installation.get(key), key) != _text(current_installation.get(key), f"current {key}"):
            raise ValueError(f"installation identity changed: {key}")
    for key in ("tool_from_camera", "tool_from_grasp_center"):
        validate_transform(profile.get(key), key)
    envelopes = validate_tool_envelopes(profile.get("tool_envelopes"))
    if not {"camera", "mount", "gripper"}.issubset({box["component"] for box in envelopes}):
        raise ValueError("profile envelopes must cover camera, mount and gripper")
    grasp = profile.get("grasp_center_calibration", {})
    for key in ("evidence_ref", "method"):
        _text(grasp.get(key), f"grasp_center_calibration {key}")
    interval = _array(grasp.get("opening_range_m"), (2,), "grasp center opening range")
    if interval[0] < 0 or interval[1] < interval[0]:
        raise ValueError("invalid grasp center opening range")
    _number(grasp.get("max_position_error_m"), "grasp center max_position_error_m", positive=True)
    intrinsics = profile.get("intrinsics", {})
    if intrinsics.get("camera_serial") != installation["camera_serial"]:
        raise ValueError("intrinsics camera differs from installation")
    validate_factory_intrinsics(intrinsics, profile.get("observed_stream", {}), profile.get("intrinsics_verification", {}))
    report = profile.get("hand_eye_result", {})
    if report.get("installation_id") != installation["installation_id"] or report.get("units") != UNITS:
        raise ValueError("hand-eye evidence belongs to another installation or units")
    if not np.allclose(validate_transform(report.get("tool_from_camera")), validate_transform(profile["tool_from_camera"]), atol=1e-9, rtol=0):
        raise ValueError("hand-eye evidence does not match tool_from_camera")
    request = profile.get("hand_eye_request")
    if not isinstance(request, dict):
        raise ValueError("hand_eye_request required to recompute calibration evidence")
    recomputed = solve_hand_eye(request)
    if request["installation_id"] != installation["installation_id"] or profile.get("tool_frame") != request["tool_frame"] or report.get("tool_frame") != request["tool_frame"]:
        raise ValueError("tool frame or installation mismatch in calibration request")
    if request["camera_intrinsics_sha256"] != _fingerprint(intrinsics):
        raise ValueError("hand-eye target poses used different camera intrinsics")
    if report.get("input_sha256") != recomputed["input_sha256"] or not np.allclose(recomputed["tool_from_camera"], profile["tool_from_camera"], atol=1e-9, rtol=0):
        raise ValueError("calibration report not bound to recomputed input")
    if profile.get("status") == "verified":
        review = profile.get("review", {})
        if review.get("reviewer_role") not in {"maintainer", "human"}:
            raise ValueError("student cannot verify a device profile")
        for key in ("reviewer_id", "reviewed_at", "evidence_ref"):
            _text(review.get(key), key)
        if review.get("profile_sha256") != profile_content_sha256(profile):
            raise ValueError("review hash does not match current profile")
        if review.get("basis") == "operator_accepted_measured_geometry":
            if review.get("task_profiles") != ["supervised_rigid_flip"] or task_profile != "supervised_rigid_flip":
                raise ValueError("Task-scoped calibration does not cover this task profile")
            if review.get("operator_accepted") is not True:
                raise ValueError("Task-scoped calibration needs operator acceptance")
            _text(review.get("operator_acceptance_ref"), "operator_acceptance_ref")
        else:
            if review.get("independent_physical_check_passed") is not True:
                raise ValueError("verified profile requires independent physical check")
            _text(review.get("physical_check_evidence_ref"), "physical_check_evidence_ref")
    elif require_verified:
        raise ValueError("draft calibration cannot authorize execution")
    return {"status": profile["status"], "installation_id": installation["installation_id"],
            "profile_sha256": profile_content_sha256(profile), "geometry_valid": True}
