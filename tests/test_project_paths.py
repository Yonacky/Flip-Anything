"""Offline regressions for the relocated project; never connect to hardware."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]


def load_source(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, PROJECT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NORMALIZER = load_source("migration_normalizer", "scripts/maintenance/normalize_paths.py")


class ProjectPathTests(unittest.TestCase):
    def test_both_configs_resolve_log_roots_from_config_location(self):
        for workspace, suffix in (("", "legacy"), ("ds_control", "ds_control")):
            with self.subTest(workspace=workspace):
                prefix = Path(workspace)
                config = load_source("path_config_" + suffix, prefix / "src/flip_anything/config.py")
                cfg = config.load_config(PROJECT / prefix / "configs/platform.json")
                self.assertEqual(Path(cfg["recording"]["root"]), PROJECT / "logs/experiments" / suffix)
                sdk_name = "root" if not workspace else "ds_control"
                self.assertEqual(Path(cfg["camera"]["log_dir"]), PROJECT / "logs/sdk" / sdk_name)
                self.assertTrue(Path(cfg["calibration"]["file"]).is_absolute())

    def test_camera_log_setup_uses_fake_sdk_and_requested_directory(self):
        # Importing these modules is passive; SDK import is intercepted before setup.
        for workspace in ("", "ds_control"):
            with self.subTest(workspace=workspace), tempfile.TemporaryDirectory() as directory:
                camera = load_source("path_camera_" + (workspace or "root"),
                                     Path(workspace) / "src/flip_anything/camera.py")
                context = mock.Mock()
                sdk = types.ModuleType("pyorbbecsdk")
                sdk.Context = context
                sdk.OBLogLevel = types.SimpleNamespace(WARNING=object())
                destination = Path(directory) / "nested/sdk"
                with mock.patch.dict(sys.modules, {"pyorbbecsdk": sdk}):
                    camera._configure_sdk_logging(destination)
                context.set_logger_to_file.assert_called_once_with(
                    sdk.OBLogLevel.WARNING, str(destination.resolve()))
                context.assert_not_called()
                self.assertTrue(destination.is_dir())

    def test_camera_log_defaults_belong_to_central_log_directory(self):
        for workspace, sdk_name in (("", "root"), ("ds_control", "ds_control")):
            with self.subTest(workspace=workspace), tempfile.TemporaryDirectory() as directory:
                camera = load_source("default_log_" + sdk_name,
                                     Path(workspace) / "src/flip_anything/camera.py")
                synthetic_root = Path(directory)
                synthetic_file = synthetic_root / workspace / "src/flip_anything/camera.py"
                sdk = types.ModuleType("pyorbbecsdk")
                sdk.Context = mock.Mock()
                sdk.OBLogLevel = types.SimpleNamespace(WARNING=object())
                with mock.patch.object(camera, "__file__", str(synthetic_file)), \
                        mock.patch.dict(sys.modules, {"pyorbbecsdk": sdk}):
                    camera._configure_sdk_logging()
                expected = synthetic_root / "logs/sdk" / sdk_name
                sdk.Context.set_logger_to_file.assert_called_once_with(
                    sdk.OBLogLevel.WARNING, str(expected.resolve()))
                sdk.Context.assert_not_called()

    def test_normalizer_respects_directory_component_boundaries(self):
        examples = {
            "F:/2610/Flip Anything/ds_control/scripts/control.py":
                "G:/Flip Anything/ds_control/scripts/control.py",
            "G:/Flip Anything/ds/camera.json":
                "G:/Flip Anything/archive/camera_diagnostics/camera.json",
            "G:/Flip Anything/ds_control/runs/DS_001/result.json":
                "G:/Flip Anything/logs/experiments/ds_control/DS_001/result.json",
            "G:/Flip Anything/session.lock": "G:/Flip Anything/session.lock",
            "G:/Flip Anything/runs_backup/file.txt": "G:/Flip Anything/runs_backup/file.txt",
            "G:/Flip Anything/ds_control_extra/file.txt": "G:/Flip Anything/ds_control_extra/file.txt",
        }
        for original, expected in examples.items():
            with self.subTest(original=original):
                self.assertEqual(NORMALIZER.normalize(original, "notes.md"), expected)

    def test_normalizer_preserves_windows_and_json_path_escaping(self):
        for separator in ("/", "\\", "\\\\"):
            with self.subTest(separator=separator):
                original = separator.join(("F:", "2610", "Flip Anything", "ds_control", "runs", "DS_001"))
                expected = separator.join(("G:", "Flip Anything", "logs", "experiments", "ds_control", "DS_001"))
                self.assertEqual(NORMALIZER.normalize(original, "notes.md"), expected)
        value = json.dumps({"path": r"F:\2610\Flip Anything\ds_control\runs\DS_001"})
        converted = json.loads(NORMALIZER.normalize(value, "record.json"))
        self.assertEqual(converted["path"], r"G:\Flip Anything\logs\experiments\ds_control\DS_001")
        self.assertEqual(
            NORMALIZER.normalize("file:///F:/2610/Flip%20Anything/ds_control", "direct_url.json"),
            "file:///G:/Flip%20Anything/ds_control")

    def test_normalizer_is_idempotent_for_paths_and_python_expressions(self):
        examples = [
            ("F:/2610/Flip Anything/ds_control/runs/DS_001", "notes.md"),
            (r"F:\2610\Flip Anything\ds\camera.json", "notes.md"),
            ("runs/DS_001/result.json", "logs/experiments/ds_control/a.md"),
            ("runs/RUN_001/result.json", "notes.md"),
            ("list(Path('runs').glob('DS_*'))", "ds_control/scripts/report.py"),
            ("ROOT / 'runs' / run_id", "ds_control/scripts/report.py"),
            ("skill_harness/episodes/demo", "README.md"),
        ]
        for original, relative in examples:
            with self.subTest(original=original):
                once = NORMALIZER.normalize(original, relative)
                self.assertEqual(NORMALIZER.normalize(once, relative), once)

    def test_normalized_path_expression_preserves_glob_receiver(self):
        source = "list(Path('runs').glob('DS_*'))"
        rewritten = NORMALIZER.normalize(source, "ds_control/scripts/report.py")
        ast.parse(rewritten, mode="eval")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "logs/experiments/ds_control"
            (destination / "DS_001").mkdir(parents=True)
            (destination / "ignore").mkdir()
            script = root / "ds_control/scripts/report.py"
            result = eval(compile(rewritten, "<normalized-path>", "eval"),
                          {"Path": Path, "__file__": str(script)})
            self.assertEqual(result, [destination / "DS_001"])

    def test_existing_control_reads_run_from_config_without_real_dispatch(self):
        # Patch subprocess before dispatch; all files are synthetic and local.
        with mock.patch.object(sys, "path", [str(PROJECT)] + sys.path):
            from skill_harness import runtime
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            workspace = project / "ds_control"
            (workspace / "configs").mkdir(parents=True)
            config = workspace / "configs/platform.json"
            config.write_text(json.dumps({"recording": {"root": "../../logs/experiments/ds_control"}}))
            (workspace / "active.json").write_text(json.dumps({"run": "DS_test"}))
            run = project / "logs/experiments/ds_control/DS_test"
            run.mkdir(parents=True)
            (run / "manifest.json").write_text(json.dumps({"status": "running"}))
            receipt = {"result_file": str(run / "receipt.json"), "result": {"ok": True}}
            process = types.SimpleNamespace(stdout=json.dumps(receipt), stderr="", returncode=0)
            with mock.patch.object(runtime, "PROJECT", project), \
                    mock.patch.object(runtime.subprocess, "run", return_value=process) as dispatch:
                adapter = runtime.ExistingControl("DS_test")
                self.assertEqual(adapter.dispatch({"action": "observe"}, None), receipt)
                dispatch.assert_called_once()
                self.assertEqual(dispatch.call_args.kwargs["cwd"], workspace)
                self.assertEqual(dispatch.call_args.args[0][-1], "observe")
                (run / "STOP").touch()
                with self.assertRaisesRegex(RuntimeError, "Hardware run not running"):
                    adapter.dispatch({"action": "observe"}, None)
                dispatch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
