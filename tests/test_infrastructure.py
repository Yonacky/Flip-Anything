import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from flip_anything.cli import record, replay
from flip_anything.config import load_config
from flip_anything.geometry import camera_point_to_base, validate_transform, load_hand_eye
from flip_anything.robot import MockRobot, XArmObserver
from flip_anything.camera import MockCamera
from flip_anything.trial import Trial
from flip_anything.gripper import require_existing_grasp

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]


class InfrastructureTests(unittest.TestCase):
    def test_grasp_guard_never_writes_or_reactivates_fresh_client(self):
        class FreshClient:
            robotiq_is_activated=False
            robotiq_status={"gACT":1,"gSTA":3,"gFLT":0,"gGTO":1,"gOBJ":2,"gPO":103}
            def robotiq_get_status(self, **kwargs):
                self.assert_kwargs=kwargs
                self.robotiq_is_activated=True
                return 0,[9,3,6]
            def robotiq_set_position(self, *args, **kwargs):
                raise AssertionError("Grasp validation must not send position command")
            def robotiq_reset(self, *args, **kwargs):
                raise AssertionError("Must not reset held gripper")
            def robotiq_set_activate(self, *args, **kwargs):
                raise AssertionError("Must not activate held gripper")
        client=FreshClient()
        self.assertEqual(require_existing_grasp(client)["gPO"],103)
        self.assertEqual(client.assert_kwargs,{"check_baud":False})

    def test_grasp_guard_does_not_recover_fault_or_empty_grip(self):
        class Client:
            robotiq_status={"gACT":1,"gSTA":3,"gFLT":9,"gGTO":1,"gOBJ":2}
            def robotiq_get_status(self, **kwargs):return 0,[]
        client=Client()
        with self.assertRaises(RuntimeError):require_existing_grasp(client)
        client.robotiq_status.update(gFLT=0,gOBJ=3)
        with self.assertRaises(RuntimeError):require_existing_grasp(client)

    def cfg(self, directory):
        cfg = load_config(ROOT / "configs/platform.json")
        cfg["recording"]["root"] = directory
        return cfg

    def args(self, run_id="test"):
        return argparse.Namespace(backend="mock", devices="both", run_id=run_id,
                                  object_id="box", frames=3, interval=0)

    def test_record_and_replay(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(record(self.cfg(directory), self.args()), 0)
            self.assertEqual(replay(Path(directory) / "test"), 0)
            manifest = json.loads((Path(directory) / "test/manifest.json").read_text())
            self.assertEqual(manifest["backend"], "mock")
            self.assertEqual(manifest["task_outcome"], "not_evaluated")
            (Path(directory) / "test/frame_000000.ppm").unlink()
            self.assertEqual(replay(Path(directory) / "test"), 1)

    def test_duplicate_and_traversal_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                Trial(directory, {}, "mock", "../escape")
            trial = Trial(directory, {}, "mock", "same")
            trial.finish("completed")
            with self.assertRaises(FileExistsError):
                Trial(directory, {}, "mock", "same")

    def test_timeout_closes_both_resources_and_records_failure(self):
        closed = []
        class Robot(MockRobot):
            def close(self):
                closed.append("robot")
        class Camera(MockCamera):
            def capture(self, *args):
                raise TimeoutError("camera missing")
            def close(self):
                closed.append("camera")
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(record(self.cfg(directory), self.args(), Robot, Camera), 1)
            self.assertEqual(closed, ["camera", "robot"])
            manifest = json.loads((Path(directory) / "test/manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")

    def test_camera_cleanup_failure_still_disconnects_robot(self):
        closed = []
        class Robot(MockRobot):
            def close(self):
                closed.append("robot")
        class Camera(MockCamera):
            def close(self):
                raise RuntimeError("stop failed")
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(record(self.cfg(directory), self.args(), Robot, Camera), 1)
            self.assertEqual(closed, ["robot"])

    def test_read_only_robot_calls_and_gripper_no_baud_write(self):
        calls = []
        class FakeAPI:
            connected, axis, version = True, 6, "fake"
            tcp_offset, tcp_load, robotiq_status = [0]*6, [0, [0]*3], {"gOBJ": 2}
            def __init__(self, ip, **kwargs):
                calls.append(("init", kwargs))
            def connect(self): calls.append(("connect", {}))
            def disconnect(self): calls.append(("disconnect", {}))
            def get_position(self, **kwargs): return 0, [0]*6
            def get_servo_angle(self, **kwargs): return 0, [0]*6
            def get_state(self): return 0, 0
            def get_err_warn_code(self): return 0, [0, 0]
            def robotiq_get_status(self, **kwargs):
                calls.append(("gripper", kwargs))
                return 0, [9, 3, 6]
        robot = XArmObserver({"ip": "192.168.1.77", "read_gripper": True}, FakeAPI)
        robot.open()
        self.assertIsNone(robot.read()["wrench"])
        robot.close()
        self.assertEqual(calls[0][1], {"is_radian": False, "do_not_open": True})
        self.assertEqual(calls[2], ("gripper", {"check_baud": False}))

    def test_transform_order_and_degrees(self):
        hand_eye = [row[:] for row in IDENTITY]
        hand_eye[0][3] = 10
        point = camera_point_to_base([1, 0, 0], [100, 0, 0, 0, 0, 90], hand_eye)
        for actual, expected in zip(point, [100, 11, 0]):
            self.assertAlmostEqual(actual, expected)
        reflected = [row[:] for row in IDENTITY]
        reflected[0][0] = -1
        with self.assertRaises(ValueError): validate_transform(reflected)
        invalid = [row[:] for row in IDENTITY]
        invalid[0][3] = float("nan")
        with self.assertRaises(ValueError): validate_transform(invalid)

    def test_unverified_calibration_rejected(self):
        with self.assertRaises(ValueError):
            load_hand_eye(ROOT / "calibration/template.json", "test")

    def test_config_relative_paths_and_unconfirmed_gripper(self):
        cfg = load_config(ROOT / "configs/platform.json")
        self.assertEqual(Path(cfg["recording"]["root"]), ROOT / "logs" / "experiments" / "legacy")
        with tempfile.TemporaryDirectory() as directory:
            cfg["robot"]["read_gripper"] = True
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError): load_config(path)


if __name__ == "__main__":
    unittest.main()
