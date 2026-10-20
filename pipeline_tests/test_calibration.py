"""Numerical calibration tests: no SDK, network, camera or robot access."""
from copy import deepcopy
import hashlib
import json
import unittest

import cv2
import numpy as np

from flip_pipeline.calibration import (
    FRAME_CONVENTION, UNITS, estimate_target_pose, inverse_transform, profile_content_sha256,
    solve_hand_eye, validate_device_profile, validate_factory_intrinsics,
    validate_tool_envelopes, validate_transform,
)


def pose(rvec, translation):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.array(rvec, dtype=float))[0]
    T[:3, 3] = translation
    return T


def synthetic_request(single_axis=False):
    rng = np.random.default_rng(841)
    X = pose([0.31, -0.26, 0.17], [-0.073, 0.018, 0.092])
    Y = pose([-0.1, 0.2, -0.05], [0.45, 0.02, 1.1])
    request = {"units": dict(UNITS), "frame_convention": dict(FRAME_CONVENTION),
               "installation_id": "synthetic-installation", "tool_frame": "controller_tcp",
               "target_id": "synthetic-fixed-board", "target_fixed_in_base": True,
               "split_policy": "predeclared", "target_geometry_evidence_ref": "synthetic/board.json",
               "camera_intrinsics_sha256": hashlib.sha256(json.dumps(intrinsics_fixture()[0], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
               "samples": []}
    for i in range(14):
        rvec = [0, 0, 0.1 * i] if single_axis else rng.uniform(-0.65, 0.65, 3)
        G = pose(rvec, [0.3 + 0.08 * rng.normal(), 0.07 * rng.normal(), 0.4 + 0.05 * rng.normal()])
        C = inverse_transform(X) @ inverse_transform(G) @ Y
        request["samples"].append({"sample_id": f"sample-{i}", "capture_id": f"capture-{i}",
                                   "evidence_ref": f"samples/{i}.json", "split": "train" if i < 10 else "validation",
                                   "base_from_tool": G.tolist(), "camera_from_target": C.tolist(),
                                   "image_timestamp_s": float(i + 1), "pose_timestamp_s": float(i + 1.001),
                                   "timestamp_clock": "shared_monotonic", "robot_stationary": True,
                                   "reprojection_error_px": 0.15})
    return request, X, Y


def intrinsics_fixture():
    profile = {"camera_serial": "SYNTHETIC_CAMERA", "stream_id": "color-aligned-depth-640x480",
               "alignment": "depth_to_color", "distortion_model": "opencv_brown_conrady",
               "width": 640, "height": 480,
               "camera_matrix": [[610, 0, 320], [0, 611, 240], [0, 0, 1]],
               "distortion_coefficients": [0.01, -0.01, 0, 0, 0],
               "depth_scale_m_per_unit": 0.001, "sdk_evidence_ref": "synthetic/sdk.json"}
    stream = {k: deepcopy(profile[k]) for k in ("camera_serial", "stream_id", "alignment", "distortion_model", "width", "height", "depth_scale_m_per_unit", "camera_matrix", "distortion_coefficients")}
    verification = {"evidence_ref": "synthetic/depth-reference.json", "independent_reference": True,
                    "max_depth_error_m": 0.003,
                    "depth_checks": [{"reference_camera_z_m": 0.3, "raw_depth_units": 300},
                                     {"reference_camera_z_m": 0.6, "raw_depth_units": 600}]}
    return profile, stream, verification


def envelope_fixture():
    return [{"id": "synthetic-camera", "component": "camera", "frame": "tool", "units": "m",
             "tool_from_box": pose([0.1, 0.2, 0.3], [0.04, 0, 0.02]).tolist(),
             "half_extents_m": [0.04, 0.02, 0.025], "uncertainty_margin_m": 0.002,
             "measurement_evidence_ref": "synthetic/tool-measurement.json"}]


def profile_fixture():
    request, X, _ = synthetic_request()
    intrinsics, stream, verification = intrinsics_fixture()
    installation = {"installation_id": request["installation_id"], "arm_serial": "SYNTHETIC_ARM",
                    "camera_serial": intrinsics["camera_serial"], "tool_id": "synthetic-tool",
                    "mount_revision": "synthetic-mount-v1", "tcp_config_sha256": "a" * 64}
    profile = {"status": "draft", "units": dict(UNITS), "installation": installation,
               "tool_frame": "controller_tcp", "tool_from_camera": X.tolist(),
               "tool_from_grasp_center": pose([0, 0, 0], [0, 0, 0.1]).tolist(),
               "grasp_center_calibration": {"evidence_ref": "synthetic/grasp-center.json", "method": "synthetic-reference",
                                            "opening_range_m": [0.02, 0.07], "max_position_error_m": 0.002},
               "tool_envelopes": envelope_fixture(), "intrinsics": intrinsics, "observed_stream": stream,
               "intrinsics_verification": verification, "hand_eye_request": request,
               "hand_eye_result": solve_hand_eye(request)}
    mount = deepcopy(profile["tool_envelopes"][0])
    mount.update(id="synthetic-mount", component="mount")
    gripper = deepcopy(profile["tool_envelopes"][0])
    gripper.update(id="synthetic-gripper", component="gripper", opening_range_m=[0.02, 0.07])
    profile["tool_envelopes"].extend([mount, gripper])
    return profile, installation


def approve(profile):
    profile["status"] = "verified"
    profile["review"] = {"reviewer_role": "maintainer", "reviewer_id": "synthetic-maintainer",
                         "reviewed_at": "2026-10-19T00:00:00+08:00", "evidence_ref": "synthetic/review.json",
                         "profile_sha256": profile_content_sha256(profile),
                         "independent_physical_check_passed": True,
                         "physical_check_evidence_ref": "synthetic/independent-check.json"}


class HandEyeTests(unittest.TestCase):
    def test_known_nontrivial_transform_recovered(self):
        request, X, Y = synthetic_request()
        result = solve_hand_eye(request)
        np.testing.assert_allclose(result["tool_from_camera"], X, atol=1e-8)
        np.testing.assert_allclose(result["base_from_target_training_estimate"], Y, atol=1e-8)
        self.assertEqual(result["status"], "draft")
        self.assertLess(result["validation"]["max_translation_error_m"], 1e-8)
        self.assertFalse(set(result["train_sample_ids"]) & set(result["validation_sample_ids"]))

    def test_validation_not_used_to_fit(self):
        request, _, _ = synthetic_request()
        request["samples"][-1]["camera_from_target"][0][3] += 0.04
        with self.assertRaisesRegex(ValueError, "validation fixed-target consistency"):
            solve_hand_eye(request)

    def test_validation_evidence_reuse_rejected(self):
        request, _, _ = synthetic_request()
        request["samples"][-1]["evidence_ref"] = request["samples"][0]["evidence_ref"]
        with self.assertRaisesRegex(ValueError, "duplicate evidence_ref"):
            solve_hand_eye(request)

    def test_renamed_pose_pair_is_not_independent(self):
        request, _, _ = synthetic_request()
        for key in ("base_from_tool", "camera_from_target"):
            request["samples"][-1][key] = deepcopy(request["samples"][0][key])
        with self.assertRaisesRegex(ValueError, "duplicate pose pair"):
            solve_hand_eye(request)

    def test_degenerate_single_axis_motion_rejected(self):
        request, _, _ = synthetic_request(single_axis=True)
        with self.assertRaisesRegex(ValueError, "degenerate rotation axes"):
            solve_hand_eye(request)

    def test_translation_only_motion_rejected(self):
        request, X, Y = synthetic_request()
        for item in request["samples"]:
            G = np.array(item["base_from_tool"])
            G[:3, :3] = np.eye(3)
            item["base_from_tool"] = G.tolist()
            item["camera_from_target"] = (inverse_transform(X) @ inverse_transform(G) @ Y).tolist()
        with self.assertRaisesRegex(ValueError, "rotation span"):
            solve_hand_eye(request)

    def test_missing_or_wrong_unit_rejected(self):
        request, _, _ = synthetic_request()
        request["units"]["translation"] = "mm"
        with self.assertRaisesRegex(ValueError, "metre/radian"):
            solve_hand_eye(request)

    def test_mislabeled_mm_scale_rejected(self):
        request, _, _ = synthetic_request()
        for item in request["samples"]:
            for key in ("base_from_tool", "camera_from_target"):
                for index in range(3):
                    item[key][index][3] *= 1000
        with self.assertRaisesRegex(ValueError, "metre scale guard"):
            solve_hand_eye(request)

    def test_wrong_frame_label_rejected(self):
        request, _, _ = synthetic_request()
        request["frame_convention"]["target_pose"] = "target_from_camera"
        with self.assertRaises(ValueError):
            solve_hand_eye(request)

    def test_inverted_robot_transform_rejected(self):
        request, _, _ = synthetic_request()
        for item in request["samples"]:
            item["base_from_tool"] = inverse_transform(item["base_from_tool"]).tolist()
        with self.assertRaises(ValueError):
            solve_hand_eye(request)

    def test_time_skew_and_unmapped_clock_rejected(self):
        request, _, _ = synthetic_request()
        request["samples"][0]["pose_timestamp_s"] += 0.2
        with self.assertRaisesRegex(ValueError, "timestamp skew"):
            solve_hand_eye(request)
        request["samples"][0]["timestamp_clock"] = "camera_device_clock"
        with self.assertRaisesRegex(ValueError, "same monotonic clock"):
            solve_hand_eye(request)

    def test_bad_reprojection_rejected(self):
        request, _, _ = synthetic_request()
        request["samples"][0]["reprojection_error_px"] = 5
        with self.assertRaisesRegex(ValueError, "reprojection"):
            solve_hand_eye(request)

    def test_no_holdout_or_unknown_limits_rejected(self):
        request, _, _ = synthetic_request()
        request["samples"] = request["samples"][:10]
        with self.assertRaisesRegex(ValueError, "insufficient"):
            solve_hand_eye(request)
        request["quality_limits"] = {"ignore_errors": True}
        with self.assertRaisesRegex(ValueError, "unknown entries"):
            solve_hand_eye(request)

    def test_unidentified_tcp_frame_rejected(self):
        request, _, _ = synthetic_request()
        del request["tool_frame"]
        with self.assertRaisesRegex(ValueError, "tool_frame"):
            solve_hand_eye(request)


class ConfigurationTests(unittest.TestCase):
    def test_transform_valid_and_invalid(self):
        T = pose([0.1, 0.2, -0.4], [0.2, 0.3, 0.4])
        np.testing.assert_allclose(T @ inverse_transform(T), np.eye(4), atol=1e-9)
        for bad in (np.diag([1, 1, -1, 1]), np.diag([2, 1, 1, 1]), np.full((4, 4), np.nan)):
            with self.assertRaises(ValueError):
                validate_transform(bad)

    def test_factory_binding_and_depth_check(self):
        profile, stream, verification = intrinsics_fixture()
        result = validate_factory_intrinsics(profile, stream, verification)
        self.assertEqual(result["status"], "configuration_checked")
        stream["width"] = 1280
        with self.assertRaisesRegex(ValueError, "width mismatch"):
            validate_factory_intrinsics(profile, stream, verification)

    def test_factory_scale_and_reference_errors(self):
        profile, stream, verification = intrinsics_fixture()
        verification["depth_checks"][0]["raw_depth_units"] = 3000
        with self.assertRaisesRegex(ValueError, "depth scale check"):
            validate_factory_intrinsics(profile, stream, verification)
        verification["independent_reference"] = False
        with self.assertRaisesRegex(ValueError, "independent measured"):
            validate_factory_intrinsics(profile, stream, verification)

    def test_factory_intrinsics_mismatch_rejected(self):
        profile, stream, verification = intrinsics_fixture()
        stream["camera_matrix"][0][0] += 10
        with self.assertRaisesRegex(ValueError, "intrinsics differ"):
            validate_factory_intrinsics(profile, stream, verification)

    def test_envelope_rotated_corners_and_margin(self):
        boxes = envelope_fixture()
        result = validate_tool_envelopes(boxes)
        self.assertEqual(len(result[0]["corners_tool_m"]), 8)
        corners = np.array(result[0]["corners_tool_m"])
        T = np.array(boxes[0]["tool_from_box"])
        local = (corners - T[:3, 3]) @ T[:3, :3]
        np.testing.assert_allclose(np.max(abs(local), axis=0), [0.042, 0.022, 0.027], atol=1e-9)

    def test_gripper_envelope_requires_opening_range(self):
        boxes = envelope_fixture()
        boxes[0]["component"] = "gripper"
        with self.assertRaises(ValueError):
            validate_tool_envelopes(boxes)

    def test_draft_cannot_authorize_execution(self):
        profile, installation = profile_fixture()
        self.assertEqual(validate_device_profile(profile, installation, require_verified=False)["status"], "draft")
        with self.assertRaisesRegex(ValueError, "draft calibration"):
            validate_device_profile(profile, installation)

    def test_review_binds_geometry_and_installation(self):
        profile, installation = profile_fixture()
        approve(profile)
        self.assertEqual(validate_device_profile(profile, installation)["status"], "verified")
        profile["tool_from_grasp_center"][2][3] += 0.001
        with self.assertRaisesRegex(ValueError, "review hash"):
            validate_device_profile(profile, installation)

    def test_installation_change_requires_new_verification(self):
        profile, installation = profile_fixture()
        approve(profile)
        changed = dict(installation, mount_revision="new-mount")
        with self.assertRaisesRegex(ValueError, "installation identity changed"):
            validate_device_profile(profile, changed)

    def test_student_cannot_verify_profile(self):
        profile, installation = profile_fixture()
        approve(profile)
        profile["review"]["reviewer_role"] = "student"
        with self.assertRaisesRegex(ValueError, "student cannot verify"):
            validate_device_profile(profile, installation)

    def test_fake_calibration_report_rejected(self):
        profile, installation = profile_fixture()
        profile["hand_eye_request"]["samples"][-1]["camera_from_target"][0][3] += 0.05
        approve(profile)
        with self.assertRaisesRegex(ValueError, "validation fixed-target consistency"):
            validate_device_profile(profile, installation)

    def test_independent_physical_check_required(self):
        profile, installation = profile_fixture()
        approve(profile)
        profile["review"]["independent_physical_check_passed"] = False
        with self.assertRaisesRegex(ValueError, "physical check"):
            validate_device_profile(profile, installation)

    def test_incomplete_tool_envelope_rejected(self):
        profile, installation = profile_fixture()
        profile["tool_envelopes"] = profile["tool_envelopes"][:1]
        with self.assertRaisesRegex(ValueError, "camera, mount and gripper"):
            validate_device_profile(profile, installation, require_verified=False)


class TargetPoseTests(unittest.TestCase):
    def fixture(self, front_on=False, nonplanar=False):
        intrinsics, _, _ = intrinsics_fixture()
        points = np.array([[x, y, 0] for x in [-0.06, 0, 0.06] for y in [-0.04, 0.04]], dtype=float)
        if nonplanar:
            points[-1, 2] = 0.02
        rvec = np.array([0, 0, 0] if front_on else [0.45, -0.35, 0.1], dtype=float)
        tvec = np.array([0, 0, 0.5], dtype=float)
        image = cv2.projectPoints(points, rvec, tvec, np.array(intrinsics["camera_matrix"], dtype=float),
                                  np.array(intrinsics["distortion_coefficients"]))[0].reshape(-1, 2)
        return {"units": dict(UNITS), "intrinsics": intrinsics, "object_points_m": points.tolist(),
                "image_points_px": image.tolist(), "target_geometry_evidence_ref": "synthetic/board.json",
                "image_evidence_ref": "synthetic/board.png"}, pose(rvec, tvec)

    def test_known_planar_target_pose(self):
        request, true_pose = self.fixture()
        result = estimate_target_pose(request)
        np.testing.assert_allclose(result["camera_from_target"], true_pose, atol=1e-8)
        self.assertEqual(result["method"], "opencv.solvePnP.IPPE")

    def test_known_nonplanar_target_pose(self):
        request, true_pose = self.fixture(nonplanar=True)
        result = estimate_target_pose(request)
        np.testing.assert_allclose(result["camera_from_target"], true_pose, atol=1e-8)
        self.assertEqual(result["method"], "opencv.solvePnP.SQPNP")

    def test_ambiguous_planar_pose_rejected(self):
        request, _ = self.fixture(front_on=True)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            estimate_target_pose(request)

    def test_bad_corner_geometry_rejected(self):
        request, _ = self.fixture()
        request["object_points_m"][0] = request["object_points_m"][1]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            estimate_target_pose(request)


if __name__ == "__main__":
    unittest.main()
