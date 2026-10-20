import copy
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from flip_pipeline.simulation import simulate_placement, _check_dynamics


def spec_fixture():
    T = np.eye(4)
    T[2, 3] = .0501
    return {"kind": "box", "contents": "known_rigid", "mass_source": "hypothesis",
            "mass_kg": 1, "com_m": [0, 0, 0], "inertia_kg_m2": [.0017] * 3,
            "inertia_frame": "object_com_principal_axes", "friction": [.5, .005, .0001],
            "size_m": [.05] * 3, "base_from_object": T.tolist(), "table_from_base": np.eye(4).tolist(),
            "world_frame": "table_z0", "units": "m_kg_s", "gravity_base_m_s2": [0, 0, -9.81],
            "evidence": ["synthetic_fixture"], "duration_s": .2}


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import mujoco
        except ImportError:
            raise unittest.SkipTest("MuJoCo optional backend not installed")

    def test_nonzero_table_height_converted(self):
        spec = spec_fixture()
        spec["base_from_object"][2][3] += .37
        spec["table_from_base"][2][3] = -.37
        result = simulate_placement(spec)
        self.assertAlmostEqual(result["initial_table_from_object"][2][3], .0501)
        self.assertGreater(result["base_from_object_final"][2][3], .4)
        self.assertGreater(result["contacts"], 0)
        self.assertNotIn("final_velocity_norm", result)
        self.assertLess(result["final_linear_speed_m_s"], .01)
        self.assertLess(result["final_angular_speed_rad_s"], .01)

    def test_tilted_table_transforms_gravity(self):
        spec = spec_fixture()
        table = np.eye(4)
        table[:3, :3] = Rotation.from_euler("x", 10, degrees=True).as_matrix()
        object_in_table = np.array(spec["base_from_object"])
        spec["table_from_base"] = table.tolist()
        spec["base_from_object"] = (np.linalg.inv(table) @ object_in_table).tolist()
        result = simulate_placement(spec)
        expected = table[:3, :3] @ np.array(spec["gravity_base_m_s2"])
        np.testing.assert_allclose(result["gravity_table_m_s2"], expected)
        self.assertGreater(abs(result["gravity_table_m_s2"][1]), 1)

    def test_legacy_or_missing_table_frame_rejected(self):
        spec = spec_fixture()
        spec["world_from_object"] = spec["base_from_object"]
        with self.assertRaisesRegex(ValueError, "ambiguous legacy"):
            simulate_placement(spec)
        del spec["world_from_object"]
        del spec["world_frame"]
        with self.assertRaisesRegex(ValueError, "table_z0"):
            simulate_placement(spec)

    def test_com_must_lie_in_primitive(self):
        for kind, size, com in [("box", [.05] * 3, [.051, 0, 0]),
                                ("cylinder", [.05, .1], [.04, .04, 0]),
                                ("cylinder", [.05, .1], [0, 0, .11])]:
            spec = dict(spec_fixture(), kind=kind, size_m=size, com_m=com)
            with self.assertRaisesRegex(ValueError, "COM outside"):
                simulate_placement(spec)

    def test_severe_initial_penetration_rejected(self):
        spec = spec_fixture()
        spec["base_from_object"][2][3] = .01
        with self.assertRaisesRegex(ValueError, "penetrates table"):
            simulate_placement(spec)

    def test_rotated_primitive_penetration_rejected(self):
        spec = spec_fixture()
        T = np.array(spec["base_from_object"])
        T[:3, :3] = Rotation.from_euler("x", 45, degrees=True).as_matrix()
        spec["base_from_object"] = T.tolist()
        with self.assertRaisesRegex(ValueError, "penetrates table"):
            simulate_placement(spec)

    def test_inertia_requires_com_frame_and_physical_eigenvalues(self):
        spec = spec_fixture()
        del spec["inertia_frame"]
        with self.assertRaisesRegex(ValueError, "explicit COM frame"):
            simulate_placement(spec)
        spec["inertia_frame"] = "object_com_principal_axes"
        spec["inertia_kg_m2"] = [.001, .001, .004]
        with self.assertRaisesRegex(ValueError, "triangle"):
            simulate_placement(spec)

    def test_full_tensor_off_diagonal_preserved(self):
        spec = spec_fixture()
        R = Rotation.from_euler("xyz", [20, 30, 40], degrees=True).as_matrix()
        I = R @ np.diag([.0015, .0017, .002]) @ R.T
        spec["inertia_frame"] = "object_com_axes"
        spec["inertia_kg_m2"] = I.tolist()
        result = simulate_placement(spec)
        attributes = ET.fromstring(result["model_xml"]).find("./worldbody/body/inertial").attrib
        self.assertIn("fullinertia", attributes)
        self.assertNotIn("diaginertia", attributes)
        parsed = np.array([float(x) for x in attributes["fullinertia"].split()])
        np.testing.assert_allclose(parsed[3:], [I[0, 1], I[0, 2], I[1, 2]])

    def test_finite_but_warning_reset_rejected(self):
        data = SimpleNamespace(qpos=np.zeros(7), qvel=np.zeros(6), qacc=np.zeros(6),
                               warning=[SimpleNamespace(number=0), SimpleNamespace(number=1)])
        with self.assertRaisesRegex(RuntimeError, "warnings"):
            _check_dynamics(data, np.array([0, 0]))

    def test_nonfinite_velocity_rejected(self):
        data = SimpleNamespace(qpos=np.zeros(7), qvel=np.array([np.nan]), qacc=np.zeros(6),
                               warning=[SimpleNamespace(number=0)])
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            _check_dynamics(data, np.array([0]))


if __name__ == "__main__":
    unittest.main()
