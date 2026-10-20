"""Lifecycle tests using synthetic evidence; no camera or robot access."""
from pathlib import Path
import tempfile
import unittest

from flip_pipeline import workflow as w
from flip_pipeline.storage import read, write
from test_calibration import approve, profile_fixture


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw = self.root / "raw/receipt.json"
        write(self.raw, {"fixture": "synthetic; not a physical validation", "ok": False})
        self.summary = {"text": "Stopped before completing the task.", "interventions": [], "failures": [], "unresolved": []}

    def tearDown(self):
        self.tmp.cleanup()

    def task(self, mode="experiment"):
        return w.new_task(self.root, "object-1", "synthetic-installation", mode)

    def calibration(self, task):
        profile, installation = profile_fixture()
        approve(profile)
        write(self.root / "device/profile.json", profile)
        write(self.root / "device/installation.json", installation)
        refs = ["raw/receipt.json", "device/profile.json", "device/installation.json"]
        checks = {k: {"result": "pass", "method": sorted(w.ALLOWED_METHODS[k])[0],
                      "evidence": refs, "detail": "Synthetic lifecycle fixture."}
                  for k in w.REQUIRED["calibration"]}
        report = {"source": "physical_evidence", "checks": checks,
                  "profile_path": refs[1], "installation_path": refs[2]}
        return w.attach_report(self.root, task, "calibration", report, refs)

    def test_dedicated_calibration_finishes_without_object_trial(self):
        task = self.task("calibration")
        self.calibration(task)
        status = w.status(self.root, task)
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["stage"], "calibration")
        self.assertEqual(status["outcome"], "calibrated")
        self.assertTrue(status["terminal"])
        self.assertEqual(status["required_checks"], [])
        self.assertEqual(status["next_action"], "none")
        with self.assertRaises(ValueError):
            w.attach_report(self.root, task, "model", {}, ["raw/receipt.json"])

    def test_experiment_continues_after_calibration(self):
        task = self.task()
        self.calibration(task)
        status = w.status(self.root, task)
        self.assertEqual(status["stage"], "model")
        self.assertEqual(status["status"], "active")
        self.assertFalse(status["terminal"])

    def test_failed_early_task_can_finalize_without_success_gates(self):
        task = self.task()
        w.halt(self.root, task, "No reliable depth for the object", "run-1")
        out = w.finalize(self.root, task, "failure", ["raw/receipt.json"], self.summary)
        self.assertEqual(out["outcome"], "failure")
        self.assertEqual(out["stopped_at_stage"], "calibration")
        self.assertEqual(out["skill_status"], "draft_material_only")
        status = w.status(self.root, task)
        self.assertEqual(status["status"], "closed")
        self.assertEqual(status["next_action"], "none")
        self.assertNotIn("execution", status["artifacts"])
        self.assertEqual(len(out["history"]), 1)

    def test_early_task_cannot_claim_success(self):
        with self.assertRaises(ValueError):
            w.finalize(self.root, self.task(), "success", ["raw/receipt.json"], self.summary)

    def test_unknown_inflight_retained_and_requires_human(self):
        task = self.task()
        state = read(task / "state.json")
        state["inflight"] = {"ticket_id": "unknown-result", "run_id": "run-1"}
        write(task / "state.json", state)
        w.halt(self.root, task, "Connection lost after command was sent", "run-1")
        out = w.finalize(self.root, task, "unknown", ["raw/receipt.json"], self.summary)
        self.assertEqual(out["inflight"]["ticket_id"], "unknown-result")
        status = w.status(self.root, task)
        self.assertEqual(status["inflight"], out["inflight"])
        self.assertEqual(status["next_action"], "await_human_action_resolution")

    def test_halt_uses_teacher_request_without_claiming_delivery(self):
        task = self.task()
        w.halt(self.root, task, "Unknown contents", "run-1")
        request = read(task / "teacher_request.json")
        self.assertEqual(request["status"], "pending")
        self.assertFalse(request["sent"])
        self.assertFalse(request["adapter_connected"])
        self.assertFalse(request["physical_recovery"])
        self.assertEqual(request["state_snapshot"]["object_id"], "object-1")
        self.assertTrue(request["missing_information"])
        self.assertTrue(request["requested_segment"])

    def test_finalization_is_immutable_and_detects_evidence_edit(self):
        task = self.task()
        w.finalize(self.root, task, "incomplete", ["raw/receipt.json"], self.summary)
        with self.assertRaises(ValueError):
            w.finalize(self.root, task, "failure", ["raw/receipt.json"], self.summary)
        write(self.raw, {"changed": True})
        status = w.status(self.root, task)
        self.assertIsNotNone(status["integrity_error"])
        self.assertEqual(status["next_action"], "inspect_and_call_teacher")

    def test_calibration_success_scope_is_not_flip_success(self):
        task = self.task("calibration")
        self.calibration(task)
        report = w.finalize(self.root, task, "success", ["raw/receipt.json"], self.summary)
        self.assertEqual(report["scope"], "calibration")
        self.assertNotIn("execution", read(task / "state.json")["artifacts"])

    def test_partial_summary_rejected(self):
        with self.assertRaises(ValueError):
            w.finalize(self.root, self.task(), "unknown", ["raw/receipt.json"], {"text": "done"})

    def test_hidden_reasoning_not_accepted_in_summary(self):
        bad = dict(self.summary, chain_of_thought="private")
        with self.assertRaises(ValueError):
            w.finalize(self.root, self.task(), "unknown", ["raw/receipt.json"], bad)

    def test_closed_task_stays_closed(self):
        task = self.task()
        w.finalize(self.root, task, "incomplete", ["raw/receipt.json"], self.summary)
        with self.assertRaises(ValueError):
            w.halt(self.root, task, "try again")


if __name__ == "__main__":
    unittest.main()
