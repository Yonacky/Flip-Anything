"""Real ds_control-shaped fixtures, no motion or hardware imports."""
from pathlib import Path
import json
import tempfile
import unittest

from flip_pipeline.material_adapter import import_run
from flip_pipeline.experience import process_student, process_teacher

SCOPE = {"goal": ["invert"], "object_family": ["container"], "tool_revision": ["mount1"]}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run = self.root / "logs/DS_run"
        (self.run / "commands").mkdir(parents=True)
        self.pose = [400, 0, 200, 0, 0, 90]
        self.state = {"pose": self.pose, "sample_end_monotonic_ns": 3000,
                      "sample_start_monotonic_ns": 2900, "state": 4}
        self.write("manifest.json", {"backend": "hardware", "status": "completed", "task_outcome": "not_evaluated"})
        self.write("commands/001.json", {"action": "move", "pose": self.pose})
        self.write("result_001.json", {"ok": True, "state": self.state,
                   "camera": {"color_file": "frame.png", "depth_file": "depth.npy"}})
        (self.run / "frame.png").write_bytes(b"raw image")
        (self.run / "depth.npy").write_bytes(b"raw depth")
        self.rows = [self.row("trial_started", 1000, {"backend": "hardware"}),
                     self.row("command", 2000, {"request": {"action": "move", "pose": self.pose}}),
                     self.row("robot_sample", 3000, self.state),
                     self.row("trial_finished", 4000, {"status": "completed"})]
        self.write_events()
        (self.root / "teacher.txt").write_text("Operator identified and approved individual sample.", encoding="utf8")

    def tearDown(self):
        self.tmp.cleanup()

    def row(self, kind, ns, payload):
        return {"kind": kind, "utc": f"2026-10-19T10:00:{ns//1000:02d}+08:00",
                "host_monotonic_ns": ns, "payload": payload}

    def write(self, name, value):
        p = self.run / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value), encoding="utf8")

    def write_events(self):
        (self.run / "events.jsonl").write_text("\n".join(json.dumps(r) for r in self.rows)+"\n", encoding="utf8")

    def run_import(self, annotations=None):
        return import_run(self.root, self.run, "episode-1", "cal-1", annotations)

    def test_hardware_receipt_proves_action_not_task_success(self):
        out = self.run_import()
        actions = [e for e in out["events"] if e["kind"] == "action"]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["source_kind"], "physical_execution")
        self.assertAlmostEqual(actions[0]["actual_pose"]["matrix"][0][3], .4)
        self.assertAlmostEqual(actions[0]["actual_pose"]["matrix"][0][1], -1)
        self.assertEqual(out["task_outcome"], "unknown")
        student = process_student(out["events"], out["manifest"], scope=SCOPE)
        self.assertEqual(student["attempts"][0]["reported_result"], "unknown")
        self.assertEqual(student["causal_hypotheses"], [])
        json.dumps(out, allow_nan=False)  # No bytes or numpy objects in CLI output.

    def test_all_raw_files_hashed(self):
        (self.run / "extra.bin").write_bytes(b"unused but must be preserved")
        out = self.run_import()
        names = {e["path"] for e in out["manifest"]["files"]}
        self.assertIn("logs/DS_run/depth.npy", names)
        self.assertIn("logs/DS_run/extra.bin", names)
        self.assertEqual(names, set(out["source_files"]))

    def test_command_without_receipt_stays_proposal(self):
        (self.run / "result_001.json").unlink()
        out = self.run_import()
        commands = [e for e in out["events"] if e.get("raw_kind") == "command"]
        self.assertEqual(commands[0]["source_kind"], "proposal")
        self.assertFalse(any(e["kind"] == "action" for e in out["events"]))

    def test_receipt_without_controller_sample_is_unconfirmed(self):
        self.rows = [r for r in self.rows if r["kind"] != "robot_sample"]
        self.write_events()
        out = self.run_import()
        e = next(e for e in out["events"] if e["source_target"] == "result_001.json")
        self.assertEqual(e["source_kind"], "proposal")
        self.assertEqual(e["execution_extent"], "unconfirmed")
        self.assertNotIn("actual_pose", e)

    def test_mismatched_pose_is_unconfirmed(self):
        self.rows[2]["payload"] = dict(self.state, pose=[500, 0, 200, 0, 0, 90])
        self.write_events()
        self.assertFalse(any(e["kind"] == "action" for e in self.run_import()["events"]))

    def test_simulation_cannot_become_physical(self):
        self.write("manifest.json", {"backend": "simulation", "status": "completed"})
        self.assertTrue(all(e["source_kind"] == "proposal" for e in self.run_import()["events"]))

    def test_teacher_needs_per_event_signed_attribution(self):
        for annotations in ({"actor": "human"}, [{"target": "result_001.json", "actor": "human"}]):
            with self.assertRaises(ValueError):
                self.run_import(annotations)
        annotation = {"target": "result_001.json", "actor": "human", "signed_by": "operator-1",
                      "evidence_refs": ["teacher.txt"], "phase": "lift", "contact": "grasped"}
        out = self.run_import([annotation])
        action = next(e for e in out["events"] if e["kind"] == "action")
        self.assertEqual(action["actor"], "human")
        self.assertTrue(any(e["actor"] == "student" for e in out["events"]))
        teacher = process_teacher(out["events"], out["manifest"], scope=SCOPE)
        self.assertTrue(teacher["demonstration_available"])
        self.assertEqual(teacher["segments"][-1]["contact"], "grasped")

    def test_phase_contact_not_guessed_from_raw_names(self):
        self.rows.insert(2, self.row("observation", 2500, {"phase": "grasp", "camera": {"color_file": "frame.png"}}))
        self.write_events()
        out = self.run_import()
        self.assertFalse(any(e["kind"] in {"phase", "contact"} for e in out["events"]))

    def test_explicit_task_result_required(self):
        annotation = {"target": "result_001.json", "actor": "human", "signed_by": "operator-1",
                      "evidence_refs": ["teacher.txt"], "task_result": "success"}
        out = self.run_import([annotation])
        self.assertEqual(out["task_outcome"], "success")
        self.assertEqual(process_student(out["events"], out["manifest"], scope=SCOPE)["attempts"][0]["reported_result"], "success")

    def test_failed_command_prefix_retained_without_whole_task_verdict(self):
        self.write("result_001.json", {"ok": False, "state": self.state, "error": "interrupted"})
        out = self.run_import()
        self.assertFalse(any(e["kind"] == "action" for e in out["events"]))
        student = process_student(out["events"], out["manifest"], scope=SCOPE)
        self.assertTrue(student["attempts"][0]["failure_prefixes"])
        self.assertEqual(student["attempts"][0]["reported_result"], "unknown")

    def test_failed_teacher_receipt_keeps_teacher_identity(self):
        self.write("result_001.json", {"ok": False, "state": self.state, "error": "interrupted"})
        out = self.run_import([{"target": "result_001.json", "actor": "astra", "signed_by": "teacher-adapter",
                               "evidence_refs": ["teacher.txt"]}])
        failure = next(e for e in out["events"] if e.get("result_scope") == "command")
        self.assertEqual(failure["actor"], "astra")
        self.assertFalse(process_teacher(out["events"], out["manifest"], scope=SCOPE)["demonstration_available"])

    def test_truncated_jsonl_retains_raw_and_warning(self):
        with (self.run / "events.jsonl").open("a", encoding="utf8") as f:
            f.write('{"partial":')
        out = self.run_import()
        self.assertTrue(any("unreadable" in w for w in out["warnings"]))
        self.assertIn("logs/DS_run/events.jsonl", out["source_files"])

    def test_invalid_command_json_not_upgraded(self):
        (self.run / "commands/001.json").write_text("{partial", encoding="utf8")
        out = self.run_import()
        self.assertFalse(any(e["kind"] == "action" for e in out["events"]))
        self.assertTrue(any("invalid JSON" in w for w in out["warnings"]))

    def test_manual_status_sample_is_observation_not_motion(self):
        self.write("commands/001.json", {"action": "status"})
        out = self.run_import([{"target": "result_001.json", "actor": "human", "signed_by": "operator",
                               "evidence_refs": ["teacher.txt"]}])
        e = next(e for e in out["events"] if e["source_target"] == "result_001.json")
        self.assertEqual(e["kind"], "observation")
        self.assertEqual(e["source_kind"], "physical_execution")
        self.assertFalse(process_teacher(out["events"], out["manifest"], scope=SCOPE)["demonstration_available"])

    def test_annotations_file_is_hashed(self):
        p = self.root / "annotations.json"
        p.write_text("[]", encoding="utf8")
        out = self.run_import(p)
        self.assertIn("annotations.json", out["source_files"])


if __name__ == "__main__":
    unittest.main()
