import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from flip_pipeline.controller_boundary import (
    controller_pose_to_matrix, matrix_to_controller_move,
    validate_controller_request, validate_motion_against_start,
)


class ControllerBoundaryTests(unittest.TestCase):
    def move(self, rpy=(20, -30, 40), motion="joint"):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
        T[:3, 3] = [0.32, -0.08, 0.41]
        return T, matrix_to_controller_move(T, grip_required=True, motion_space=motion, speed=4)

    def test_m_to_mm_and_exact_fixed_axis_order(self):
        T, command = self.move()
        np.testing.assert_allclose(command["pose"][:3], [320, -80, 410])
        r, p, y = map(math.radians, command["pose"][3:])
        Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
        Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
        Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        np.testing.assert_allclose(Rz @ Ry @ Rx, T[:3, :3], atol=1e-12)
        self.assertEqual(command["speed_units"], "deg_s")
        self.assertEqual(command["orientation_convention"], "extrinsic_xyz_rpy")

    def test_random_pose_roundtrips(self):
        rng = np.random.default_rng(184)
        for _ in range(80):
            T = np.eye(4)
            T[:3, :3] = Rotation.random(random_state=rng).as_matrix()
            T[:3, 3] = rng.uniform(-0.6, 0.6, 3)
            command = matrix_to_controller_move(T, grip_required=False, motion_space="joint", speed=3)
            np.testing.assert_allclose(controller_pose_to_matrix(command["pose"]), T, atol=1e-12)

    def test_gimbal_and_near_gimbal_equivalent_transform(self):
        for pitch in [90, -90, 89.99999999, -89.99999999, 90.00000001]:
            with self.subTest(pitch=pitch):
                T, command = self.move([71, pitch, -23])
                self.assertTrue(command["near_euler_singularity"])
                np.testing.assert_allclose(controller_pose_to_matrix(command["pose"]), T, rtol=0, atol=3e-7)

    def test_legacy_without_unit_marker_rejected(self):
        legacy = {"action": "move", "pose": [320, 0, 400, 0, 0, 0], "grip_required": False}
        with self.assertRaisesRegex(ValueError, "mm_deg"):
            validate_controller_request(legacy)

    def test_wrong_units_frame_or_rotation_convention_rejected(self):
        _, valid = self.move()
        for name, value in [("units", "m_rad"), ("frame", "camera"), ("orientation_convention", "intrinsic_xyz")]:
            bad = dict(valid, **{name: value})
            with self.assertRaises(ValueError):
                validate_controller_request(bad)

    def test_explicit_speed_and_motion_units(self):
        _, command = self.move()
        bad = dict(command, speed_units="mm_s")
        with self.assertRaisesRegex(ValueError, "speed_units"):
            validate_controller_request(bad)
        for speed in [0, -1, float("nan"), float("inf"), True, "4"]:
            with self.assertRaises(ValueError):
                validate_controller_request(dict(command, speed=speed))
        with self.assertRaises(TypeError):
            matrix_to_controller_move(np.eye(4), grip_required=False, speed=4)

    def test_fresh_start_determines_legacy_branch(self):
        _, command = self.move([0, 0, 10], motion="joint")
        result = validate_motion_against_start(command, [0, 0, 0, 0, 0, 0])
        self.assertEqual(result["motion_space"], "joint")
        with self.assertRaisesRegex(ValueError, "differs from worker branch"):
            validate_motion_against_start(command, [0, 0, 0, 0, 0, 10])

    def test_translation_branch_and_threshold(self):
        _, command = self.move([0, 0, 0.4], motion="cartesian_translation")
        result = validate_motion_against_start(command, [0, 0, 0, 0, 0, 0])
        self.assertEqual(result["speed_units"], "mm_s")
        _, at_threshold = self.move([0, 0, 0.5], motion="joint")
        with self.assertRaisesRegex(ValueError, "too close"):
            validate_motion_against_start(at_threshold, [0, 0, 0, 0, 0, 0])

    def test_gripper_registers_not_continuous_lengths(self):
        valid = {"action": "grip", "pos": 110, "speed": 40, "force": 10, "grip_required": False}
        self.assertEqual(validate_controller_request(valid), valid)
        for key, value in [("pos", 0.05), ("speed", -1), ("force", 256), ("pos", True)]:
            with self.assertRaises(ValueError):
                validate_controller_request(dict(valid, **{key: value}))

    def test_bad_pose_and_expected_pose_rejected(self):
        _, command = self.move()
        for pose in [[0] * 5, [0, 0, float("nan"), 0, 0, 0], [True, 0, 0, 0, 0, 0], ["320", 0, 0, 0, 0, 0]]:
            with self.assertRaises(ValueError):
                validate_controller_request(dict(command, pose=pose))
            with self.assertRaises(ValueError):
                validate_controller_request(dict(command, expected_pose=pose))

    def test_cannot_override_converted_pose(self):
        with self.assertRaisesRegex(ValueError, "cannot override"):
            matrix_to_controller_move(np.eye(4), grip_required=False, motion_space="joint", speed=3, pose=[0] * 6)


if __name__ == "__main__":
    unittest.main()
