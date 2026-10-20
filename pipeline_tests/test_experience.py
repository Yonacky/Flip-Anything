"""Offline evidence tests: actual traces, not robot control or mocked hardware success."""
import copy
import unittest

from flip_pipeline import experience as ex

TIME = "2026-10-19T10:00:00+08:00"
SCOPE = {"goal": ["invert"], "object_family": ["rigid_container"],
         "tool_revision": ["mount-01"], "contents": ["confirmed_empty"]}
BLOBS = {"rgb/start.png": b"test image bytes", "controller/result.json": b'{"code":0}'}
REGISTRY = ["codex-maintainer"]


def pose(x=0, y=0, z=0):
    return {"matrix": [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]],
        "frame_id": "base", "units": "m", "timestamp": TIME,
        "calibration_id": "cal-01", "source": "controller"}


def event(i, kind="action", actor="student", **extra):
    e = {"schema_version": 1, "event_id": f"e{i}", "episode_id": "ep1", "attempt_id": "try1",
        "timestamp": TIME, "actor": actor, "source_kind": "physical_execution", "kind": kind,
        "observations": ["rgb/start.png"], "evidence_refs": ["controller/result.json"]}
    if kind == "action":
        e["actual_pose"] = pose(0.5, 0.1, 0.2)
    e.update(extra)
    return e


def draft():
    return ex.process_student([event(1), event(2, "result", result="failure")],
                              ex.evidence_manifest(BLOBS), scope=SCOPE)


def reviewed(value=None):
    return ex.review_record(value or draft(), reviewer="codex-maintainer", reviewer_role="maintainer",
        authorized_maintainers=REGISTRY, evidence_refs=["controller/result.json"],
        decision="approve", blobs=BLOBS, notes="Reviewed material as a failed example; no success claim.")


def released(value=None):
    return ex.release_record(reviewed(value), maintainer="codex-maintainer", maintainer_role="maintainer",
        authorized_maintainers=REGISTRY, blobs=BLOBS)


class EventTests(unittest.TestCase):
    def test_proposal_is_not_execution(self):
        e = event(1, actor="astra", source_kind="proposal")
        with self.assertRaises(ValueError):
            ex.validate_event(e)
        del e["actual_pose"]
        out = ex.process_teacher([e], ex.evidence_manifest(BLOBS), scope=SCOPE)
        self.assertFalse(out["demonstration_available"])
        self.assertEqual(out["segments"], [])
        self.assertEqual(out["teacher_proposals"], ["e1"])

    def test_actual_pose_required(self):
        e = event(1)
        del e["actual_pose"]
        with self.assertRaises(ValueError):
            ex.validate_event(e)

    def test_controller_not_guessed_pose(self):
        e = event(1)
        e["actual_pose"]["source"] = "vlm_guess"
        with self.assertRaises(ValueError):
            ex.validate_event(e)

    def test_timezone_and_order(self):
        a, b = event(1), event(2)
        b["timestamp"] = "2026-10-19T09:59:00+08:00"
        with self.assertRaises(ValueError):
            ex.validate_trace([a, b])
        a["timestamp"] = "2026-10-19T10:00:00"
        with self.assertRaises(ValueError):
            ex.validate_event(a)

    def test_duplicate_and_cross_episode(self):
        with self.assertRaises(ValueError):
            ex.validate_trace([event(1), event(1)])
        with self.assertRaises(ValueError):
            ex.validate_trace([event(1), event(2, episode_id="ep2")])

    def test_hidden_reasoning_not_harvested(self):
        e = event(1, "public_transcript", text="I will reobserve because the target is occluded.")
        self.assertEqual(ex.validate_event(e)["text"], e["text"])
        e["payload"] = {"hidden_reasoning": "private"}
        with self.assertRaises(ValueError):
            ex.validate_event(e)

    def test_nan_and_improper_rotation(self):
        for value in (float("nan"), True):
            e = event(1)
            e["actual_pose"]["matrix"][0][0] = value
            with self.assertRaises(ValueError):
                ex.validate_event(e)
        e = event(1)
        e["actual_pose"]["matrix"][0][0] = -1
        with self.assertRaises(ValueError):
            ex.validate_event(e)

    def test_success_needs_evidence(self):
        with self.assertRaises(ValueError):
            ex.validate_event(event(1, "result", result="success", evidence_refs=[]))


class EvidenceTests(unittest.TestCase):
    def test_manifest_deterministic_and_detects_changes(self):
        m = ex.evidence_manifest(BLOBS)
        self.assertEqual(m, ex.evidence_manifest(dict(reversed(list(BLOBS.items())))))
        self.assertTrue(ex.verify_manifest(m, BLOBS))
        changed = dict(BLOBS)
        changed["rgb/start.png"] = b"changed"
        with self.assertRaises(ValueError):
            ex.verify_manifest(m, changed)
        with self.assertRaises(ValueError):
            ex.verify_manifest(m, {"rgb/start.png": BLOBS["rgb/start.png"]})

    def test_manifest_rejects_path_ambiguity(self):
        for name in ("../secret", "C:/secret", "/abs", "a/../b", "a\\b", ".", "a//b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                ex.evidence_manifest({name: b"x"})
        with self.assertRaises(ValueError):
            ex.evidence_manifest({"A": b"a", "a": b"b"})

    def test_missing_reference_and_manifest_tampering(self):
        e = event(1, evidence_refs=["missing.json"])
        with self.assertRaises(ValueError):
            ex.process_student([e], ex.evidence_manifest(BLOBS), scope=SCOPE)
        m = ex.evidence_manifest(BLOBS)
        m["files"][0]["size"] += 1
        with self.assertRaises(ValueError):
            ex.process_student([event(1)], m, scope=SCOPE)


class TeacherTests(unittest.TestCase):
    def test_phase_contact_segmentation_and_object_frame(self):
        e = event(3, actor="human")
        o = pose(0.4, 0.1, 0.2)
        o["matrix"] = [[0, -1, 0, .4], [1, 0, 0, .1], [0, 0, 1, .2], [0, 0, 0, 1]]
        o.update(metric=True, quality="validated")
        e["object_pose"] = o
        trace = [event(1, "phase", "human", phase="grasp"),
            event(2, "contact", "human", contact="grasped"), e,
            event(4, "contact", "human", contact="supported"),
            event(5, "result", "human", result="success")]
        out = ex.process_teacher(trace, ex.evidence_manifest(BLOBS), scope=SCOPE)
        self.assertEqual(len(out["segments"]), 3)
        rel = out["segments"][1]["waypoints"][0]["T_object_tcp"]
        self.assertAlmostEqual(rel[0][3], 0)
        self.assertAlmostEqual(rel[1][3], -.1)
        self.assertEqual(rel[0][:3], [0, 1, 0])
        self.assertTrue(out["demonstration_available"])
        self.assertFalse(out["replay_allowed"])
        self.assertFalse(out["teacher_infallible"])

    def test_bad_object_registration_keeps_raw_not_transfer(self):
        for changes in ({"quality": "estimated"}, {"metric": False}, {"frame_id": "camera"},
                        {"calibration_id": "other"}, {"timestamp": "2026-10-19T10:00:02+08:00"}):
            e = event(1, actor="human")
            e["object_pose"] = dict(pose(), metric=True, quality="validated", **changes) if not set(changes) & {"metric", "quality"} else dict(pose(), metric=True, quality="validated")
            e["object_pose"].update(changes)
            out = ex.process_teacher([e], ex.evidence_manifest(BLOBS), scope=SCOPE)
            self.assertEqual(out["segments"][0]["waypoints"], [])
            self.assertEqual(len(out["raw_events"]), 1)
            self.assertEqual(len(out["segments"][0]["nontransferable_samples"]), 1)

    def test_attempt_phase_not_carried_forward(self):
        trace = [event(1, "phase", "human", phase="release"),
                 event(2, actor="human", attempt_id="try2")]
        out = ex.process_teacher(trace, ex.evidence_manifest(BLOBS), scope=SCOPE)
        self.assertEqual(out["segments"][1]["phase"], "unannotated")

    def test_request_is_pending_never_sent(self):
        r = ex.teacher_request(request_id="r1", teacher="astra", missing_information=["contents"],
            question="Is there a noncontact way to confirm the contents?", requested_segment="inspection",
            state_snapshot={"stage": "material_assessment", "holding": False},
            evidence_refs=["rgb/start.png"], timestamp=TIME)
        self.assertEqual(r["status"], "pending")
        self.assertFalse(r["sent"])
        self.assertFalse(r["adapter_connected"])
        self.assertEqual(r["next_action"], "manual_handoff_required")


class StudentAndGovernanceTests(unittest.TestCase):
    def test_failed_prefix_and_retry_preserved(self):
        trace = [event(1), event(2, "result", result="failure"),
            event(3, "hypothesis", text="The object may have slipped.", source_kind="proposal"),
            event(4, "intervention", "human", text="Reset object"),
            event(5, attempt_id="try2"), event(6, "result", result="success", attempt_id="try2")]
        out = ex.process_student(trace, ex.evidence_manifest(BLOBS), scope=SCOPE)
        self.assertEqual(out["retry_count"], 1)
        self.assertEqual(out["attempts"][0]["failure_prefixes"], [["e1", "e2"]])
        self.assertEqual(out["attempts"][0]["reported_result"], "failure")
        self.assertEqual(out["attempts"][1]["reported_result"], "success")
        self.assertEqual(out["causal_hypotheses"][0]["status"], "unverified")
        self.assertFalse(out["autonomous"])
        self.assertEqual(len(out["raw_events"]), 6)

    def test_teacher_advice_not_counted_as_autonomous(self):
        trace = [event(1, "public_transcript", "astra", source_kind="proposal", text="Observe from the side."), event(2)]
        out = ex.process_student(trace, ex.evidence_manifest(BLOBS), scope=SCOPE)
        self.assertFalse(out["autonomous"])

    def test_only_registered_maintainer_can_review(self):
        for name, role, registry in [("student", "maintainer", ["student"]),
                ("unregistered", "maintainer", REGISTRY), ("codex-maintainer", "student", REGISTRY)]:
            with self.assertRaises(ValueError):
                ex.review_record(draft(), reviewer=name, reviewer_role=role,
                    authorized_maintainers=registry, evidence_refs=["controller/result.json"],
                    decision="approve", blobs=BLOBS, notes="checked")

    def test_review_and_release_separate(self):
        d = draft()
        with self.assertRaises(ValueError):
            ex.release_record(d, maintainer="codex-maintainer", maintainer_role="maintainer",
                              authorized_maintainers=REGISTRY, blobs=BLOBS)
        r = reviewed(d)
        self.assertEqual(r["status"], "reviewed")
        self.assertEqual(d["status"], "draft")
        self.assertEqual(released()["status"], "released")

    def test_change_after_review_requires_review_again(self):
        r = reviewed()
        r["scope"]["contents"] = ["any_contents"]
        with self.assertRaises(ValueError):
            ex.release_record(r, maintainer="codex-maintainer", maintainer_role="maintainer",
                              authorized_maintainers=REGISTRY, blobs=BLOBS)

    def test_scope_before_similarity_and_no_unreviewed(self):
        good = released()
        other = draft()
        other["scope"]["contents"] = ["liquid"]
        other["description"] = "perfect red can inversion match"
        other = released(other)
        revoked = ex.revoke_record(good, maintainer="codex-maintainer", maintainer_role="maintainer",
            authorized_maintainers=REGISTRY, reason="Evidence contradicted by later measurement")
        context = {k: v[0] for k, v in SCOPE.items()}
        found = ex.retrieve([draft(), reviewed(), other, good], context, ["red", "can"])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["record"]["record_id"], good["record_id"])
        self.assertEqual(ex.retrieve([good], {"goal": "invert"}), [])
        self.assertEqual(found[0]["use"], "planning_reference_only")
        self.assertEqual(ex.retrieve([good, revoked], context), [])

    def test_tampered_release_not_retrieved(self):
        r = released()
        r["scope"]["tool_revision"] = ["other"]
        context = {k: v[0] for k, v in r["scope"].items()}
        self.assertEqual(ex.retrieve([r], context), [])

    def test_scope_cannot_hide_missing_constraints(self):
        for scope in ({}, {"goal": ["invert"], "object_family": ["*"], "tool_revision": ["v1"]}):
            with self.assertRaises(ValueError):
                ex.process_student([event(1)], ex.evidence_manifest(BLOBS), scope=scope)

    def test_unknown_result_stays_unknown(self):
        out = ex.process_student([event(1)], ex.evidence_manifest(BLOBS), scope=SCOPE)
        self.assertEqual(out["attempts"][0]["reported_result"], "unknown")
        self.assertEqual(out["status"], "draft")


if __name__ == "__main__":
    unittest.main()
