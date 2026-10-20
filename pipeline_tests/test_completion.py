from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
import numpy as np

from flip_pipeline.completion import validate_completion
from flip_pipeline.storage import read, write, digest, evidence


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.folder = self.root / "logs/pipeline/TASK_test"
        self.folder.mkdir(parents=True)
        self.run = "DS_fixture"
        self.paths = []
        write(self.root / "devices/pipeline_policy.json", {})
        self.segments = [
            {"id": "flip", "role": "invert", "request": {"action": "move", "pose": [100, 0, 300, 180, 0, 0], "grip_required": True}},
            {"id": "release", "role": "release", "request": {"action": "grip", "pos": 20, "grip_required": False}},
            {"id": "retreat", "role": "retreat", "request": {"action": "move", "pose": [100, 0, 350, 180, 0, 0], "grip_required": False}},
        ]
        artifact = {"data": {"segments": self.segments}}
        self.plan_hash = digest(artifact)
        write(self.folder / "artifacts" / (self.plan_hash + ".json"), artifact)
        self.state = {"task_id": "TASK_test", "object_id": "obj", "installation_id": "install", "mode": "experiment",
                      "status": "active", "inflight": None, "artifacts": {"calibration": "c" * 64, "plan": self.plan_hash}, "history": []}
        previous_pose = [100, 0, 300, 0, 0, 0]
        for i, segment in enumerate(self.segments):
            grip_before = {"gPO": 130 if i < 2 else 20, "gOBJ": 2 if i < 2 else 3, "gFLT": 0}
            before = {"ok": True, "state": {"pose": previous_pose, "sample_end_monotonic_ns": int((1 + i * 2 + (0.2 if i else 0)) * 1e9), "gripper": grip_before}}
            ref = self.save(f"logs/raw/{self.run}/result_before_{i}.json", before)
            ticket = {"ticket_id": f"ticket{i}", "task": str(self.folder.relative_to(self.root)), "run_id": self.run,
                      "segment_id": segment["id"], "plan_hash": self.plan_hash, "state": "consumed",
                      "reference": evidence(self.root, [ref]), "issued_monotonic_ns": int((2 + i * 2) * 1e9),
                      "command_hash": digest(dict(segment["request"], expected_pose=previous_pose))}
            next_pose = segment["request"].get("pose", previous_pose)
            result = {"ok": True, "state": {"pose": next_pose, "sample_end_monotonic_ns": int((3 + i * 2) * 1e9),
                                            "gripper": {"gPO": 130 if i == 0 else 20, "gOBJ": 2 if i == 0 else 3, "gFLT": 0}}}
            if i == 1:
                raw = self.save("captures/support.json", {"capture": "support-frame"})
                support = {"supported": True, "object_id": "obj", "run_id": self.run, "reference_receipt_sha256": digest(before),
                           "completed_action_count": 1, "sample_monotonic_ns": int(3.5e9), "raw_evidence": evidence(self.root, [raw])}
                support_path = self.save("support.json", support)
                ticket["support_evidence"] = {"path": support_path, "sha256": digest(support), "run_id": self.run, "method": "observed_contact"}
            self.save(f"logs/pipeline/TASK_test/tickets/ticket{i}.json", ticket)
            self.state["history"].append({"event": "action_completed", "segment_id": segment["id"], "plan_hash": self.plan_hash,
                                          "run_id": self.run, "ticket_id": f"ticket{i}", "result": result})
            previous_pose = next_pose
        self.final = deepcopy(self.state["history"][-1]["result"])
        self.final["state"]["sample_end_monotonic_ns"] = int(8e9)
        final_path = self.save(f"logs/raw/{self.run}/result_final.json", self.final)
        self.observations = {"object_id": "obj", "installation_id": "install", "run_id": self.run,
                             "calibration_artifact_sha256": "c" * 64, "frame": "base", "units": "m_rad",
                             "axis_object": [0, 0, 1], "axis_label": "toward labeled opening", "post_retreat": []}
        for i, when in enumerate([1.5, 8.5, 10.5]):
            T = np.eye(4)
            T[2, 3] = .08
            if i:
                T[:3, :3] = np.diag([1, -1, -1])
            raw = self.save(f"captures/pose_{i}.json", {"capture": f"unique-{i}", "time_ns": int(when * 1e9)})
            item = {"capture_id": f"image{i}", "sample_monotonic_ns": int(when * 1e9), "base_from_object": T.tolist(),
                    "position_uncertainty_m": .0005, "orientation_uncertainty_rad": .005,
                    "method": "rgbd_model_fit", "raw_evidence": evidence(self.root, [raw])}
            if i:
                self.observations["post_retreat"].append(item)
            else:
                self.observations["before"] = item
        obs_path = self.save("completion_observations.json", self.observations)
        self.data = {"run_id": self.run, "final_receipt_path": final_path, "completion_observations_path": obs_path}

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, relative, data):
        write(self.root / relative, data)
        if relative not in self.paths:
            self.paths.append(relative)
        return relative

    def check(self):
        self.save("completion_observations.json", self.observations)
        return validate_completion(self.root, self.folder, self.state, self.data, evidence(self.root, self.paths))

    def test_measured_completion_passes(self):
        result = self.check()
        self.assertEqual(result["completed_segments"], 3)
        self.assertEqual(result["release_segment_ids"], ["release"])
        self.assertEqual(result["stability_interval_s"], 2)

    def test_cannot_claim_success_without_actions(self):
        self.state["history"] = []
        with self.assertRaisesRegex(ValueError, "every planned action"):
            self.check()

    def test_unresolved_action_rejected(self):
        self.state["inflight"] = {"ticket_id": "unknown"}
        with self.assertRaisesRegex(ValueError, "unresolved"):
            self.check()

    def test_wrong_run_or_order_rejected(self):
        self.state["history"][1]["run_id"] = "other"
        with self.assertRaisesRegex(ValueError, "another run"):
            self.check()
        self.state["history"][1]["run_id"] = self.run
        self.state["history"].reverse()
        with self.assertRaisesRegex(ValueError, "in order"):
            self.check()

    def test_wrong_calibration_and_axis_rejected(self):
        self.observations["calibration_artifact_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "another run/calibration"):
            self.check()
        self.observations["calibration_artifact_sha256"] = "c" * 64
        self.observations["axis_label"] = ""
        with self.assertRaisesRegex(ValueError, "directed object axis"):
            self.check()

    def test_not_inverted_rejected(self):
        self.observations["post_retreat"][0]["base_from_object"] = self.observations["before"]["base_from_object"]
        with self.assertRaisesRegex(ValueError, "did not invert"):
            self.check()

    def test_drifting_object_rejected(self):
        self.observations["post_retreat"][-1]["base_from_object"][0][3] += .02
        with self.assertRaisesRegex(ValueError, "not independently stable"):
            self.check()

    def test_repeat_capture_or_short_interval_rejected(self):
        post = self.observations["post_retreat"]
        post[1]["capture_id"] = post[0]["capture_id"]
        with self.assertRaisesRegex(ValueError, "captures must be distinct"):
            self.check()
        post[1]["capture_id"] = "new"
        post[1]["sample_monotonic_ns"] = post[0]["sample_monotonic_ns"] + 1000
        with self.assertRaisesRegex(ValueError, "stability interval"):
            self.check()

    def test_raw_data_mutation_or_omission_rejected(self):
        path = self.observations["post_retreat"][0]["raw_evidence"][0]["path"]
        self.save(path, {"altered": True})
        with self.assertRaisesRegex(ValueError, "Evidence changed"):
            self.check()

    def test_support_bound_to_immediate_state(self):
        support = read(self.root / "support.json")
        support["completed_action_count"] = 0
        self.save("support.json", support)
        ticket_path = "logs/pipeline/TASK_test/tickets/ticket1.json"
        ticket = read(self.root / ticket_path)
        ticket["support_evidence"]["sha256"] = digest(support)
        self.save(ticket_path, ticket)
        with self.assertRaisesRegex(ValueError, "immediately preceding"):
            self.check()

    def test_gripper_obstruction_or_missing_final_release_rejected(self):
        release = self.state["history"][1]["result"]["state"]["gripper"]
        release["gOBJ"] = 2
        with self.assertRaisesRegex(ValueError, "without obstruction"):
            self.check()
        release["gOBJ"] = 3
        self.final["state"]["gripper"]["gOBJ"] = 2
        self.save(self.data["final_receipt_path"], self.final)
        with self.assertRaisesRegex(ValueError, "continued release"):
            self.check()

    def test_final_receipt_must_follow_last_motion(self):
        self.final["state"]["sample_end_monotonic_ns"] = int(6e9)
        self.save(self.data["final_receipt_path"], self.final)
        with self.assertRaisesRegex(ValueError, "predates last action"):
            self.check()

    def test_real_synthetic_calibration_is_accepted_and_bound(self):
        # No calibration validator mocks: exercise the full hand-eye/profile gate.
        from pipeline_tests.test_calibration import profile_fixture, approve
        from flip_pipeline import workflow
        profile, installation = profile_fixture()
        approve(profile)
        initial_state = deepcopy(self.state)
        initial_state.update(installation_id=installation["installation_id"], stage="calibration", artifacts={}, history=[])
        write(self.folder / "state.json", initial_state)
        self.save("real_synthetic_profile.json", profile)
        self.save("real_synthetic_installation.json", installation)
        report = {"source": "physical_evidence", "profile_path": "real_synthetic_profile.json", "installation_path": "real_synthetic_installation.json",
                  "checks": {name: {"result": "pass", "method": sorted(workflow.ALLOWED_METHODS[name])[0],
                                    "detail": "synthetic integration fixture only", "evidence": ["real_synthetic_profile.json"]}
                             for name in workflow.REQUIRED["calibration"]}}
        artifact = workflow.attach_report(self.root, self.folder, "calibration", report,
                                          ["real_synthetic_profile.json", "real_synthetic_installation.json"])
        calibration_hash = digest(artifact)
        self.state["installation_id"] = installation["installation_id"]
        self.state["artifacts"]["calibration"] = calibration_hash
        self.observations["installation_id"] = installation["installation_id"]
        self.observations["calibration_artifact_sha256"] = calibration_hash
        self.assertEqual(self.check()["status"], "measured_completion_checked")


if __name__ == "__main__":
    unittest.main()
