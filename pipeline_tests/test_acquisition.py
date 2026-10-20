"""Acquisition authorization uses temp fixtures; no hardware calls or motion."""
import copy
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from flip_pipeline import supervision as s
from flip_pipeline import workflow as w
from flip_pipeline.storage import digest, evidence, read, write


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name).resolve()
        self.task=w.new_task(self.root,'object_fixture','installation_fixture',mode='calibration')
        self.run='DS_acquisition_fixture'
        write(self.root/'ds_control/active.json',{'run':self.run})
        write(self.root/'devices/pipeline_policy.json',{'authorized_path_reviewers':['teacher_fixture']})
        write(self.root/'review_evidence.json',{'synthetic':True,'note':'review fixture only; not real path certification'})
        self.initial=self.root/'logs'/self.run/'result_initial.json'
        self.current=self.root/'logs'/self.run/'result_current.json'
        receipt={'ok':True,'state':{'pose':[400,0,300,0,0,0],
            'sample_end_monotonic_ns':time.monotonic_ns(),
            'gripper':{'gOBJ':3,'gPO':0,'gSTA':3,'gACT':1,'gFLT':0}}}
        write(self.initial,receipt);write(self.current,receipt)
        self.request={'action':'move','pose':[405,0,300,0,0,0],
            'grip_required':False,'frame':'base','units':'mm_deg',
            'orientation_convention':'extrinsic_xyz_rpy','motion_space':'cartesian_translation',
            'speed':5,'speed_units':'mm_s'}
        state=read(self.task/'state.json')
        self.plan={key:state[key] for key in ('task_id','object_id','installation_id')}
        initial_ref=evidence(self.root,[self.initial])[0]
        self.plan.update({'purpose':'calibration_views','reviewed':True,'reviewer_role':'teacher',
            'reviewer_id':'teacher_fixture','checks':{'path_reviewed':True,'empty_tool':True},
            'initial_result':initial_ref,
            'evidence':evidence(self.root,[self.initial,'review_evidence.json']),
            'segments':[{'id':'observe_1','role':'observe','request':copy.deepcopy(self.request)}]})
        self.plan_path=self.root/'acquisition_plan.json'
        self.save_plan(self.plan)

    def tearDown(self):
        self.temp.cleanup()

    def save_plan(self,plan):
        plan=copy.deepcopy(plan)
        plan.pop('reviewed_plan_sha256',None)
        plan['reviewed_plan_sha256']=digest(plan)
        write(self.plan_path,plan)
        return plan

    def issue(self,request=None):
        return s.issue(self.root,self.task,self.run,'observe_1',request or self.request,
                       self.current,collection_plan=self.plan_path)

    def test_reviewed_single_view_can_issue_consume_complete(self):
        payload=self.issue()
        self.assertEqual(payload['expected_pose'],[400,0,300,0,0,0])
        folder,ticket=s.consume(self.root,self.run,payload)
        self.assertEqual(read(folder/'state.json')['inflight']['ticket_id'],ticket['ticket_id'])
        s.complete(self.root,folder,ticket,{'ok':True,'fixture':True})
        state=read(folder/'state.json')
        self.assertIsNone(state['inflight'])
        self.assertEqual(state['stage'],'calibration')
        self.assertEqual(state['history'][-1]['event'],'action_completed')
        self.assertIn('plan_hash',state['history'][-1])

    def test_missing_teacher_or_unauthorized_reviewer_rejected(self):
        for changes in ({'reviewed':False},{'reviewer_role':'student'},
                        {'reviewer_id':'unconfigured_teacher'}):
            with self.subTest(changes=changes):
                self.save_plan({**self.plan,**changes})
                with self.assertRaisesRegex(ValueError,'teacher review'):
                    self.issue()

    def test_any_plan_edit_without_new_review_digest_rejected(self):
        plan=read(self.plan_path)
        plan['segments'][0]['request']['pose'][0]+=1
        write(self.plan_path,plan)
        with self.assertRaisesRegex(ValueError,'review hash'):
            self.issue()

    def test_wrong_purpose_for_stage_rejected(self):
        for purpose in ('object_views','material_probe','arbitrary_motion'):
            with self.subTest(purpose=purpose):
                self.save_plan({**self.plan,'purpose':purpose})
                with self.assertRaisesRegex(ValueError,'purpose'):
                    self.issue()

    def test_missing_initial_result_rejected(self):
        plan=copy.deepcopy(self.plan);plan.pop('initial_result')
        self.save_plan(plan)
        with self.assertRaises(ValueError):self.issue()

    def test_initial_result_must_be_hashed_review_evidence(self):
        plan=copy.deepcopy(self.plan)
        plan['evidence']=evidence(self.root,['review_evidence.json'])
        self.save_plan(plan)
        with self.assertRaises(ValueError):self.issue()

    def test_initial_receipt_hash_mismatch_rejected(self):
        plan=copy.deepcopy(self.plan)
        plan['initial_result']['sha256']='0'*64
        self.save_plan(plan)
        with self.assertRaises(ValueError):self.issue()

    def test_current_pose_must_match_reviewed_initial_pose(self):
        for index,delta in ((0,3),(3,3)):
            with self.subTest(component=index):
                initial=read(self.current)
                initial['state']['pose'][index]+=delta
                write(self.initial,initial)
                plan=copy.deepcopy(self.plan)
                plan['initial_result']=evidence(self.root,[self.initial])[0]
                plan['evidence']=evidence(self.root,[self.initial,'review_evidence.json'])
                self.save_plan(plan)
                with self.assertRaises(ValueError):self.issue()

    def test_held_or_unknown_gripper_cannot_acquire_views(self):
        cases=[{'gOBJ':1,'gPO':100},{'gOBJ':2,'gPO':120},{'gPO':0},
               {'gOBJ':0,'gPO':0},None]
        for gripper in cases:
            with self.subTest(gripper=gripper):
                current=read(self.current);current['state']['gripper']=gripper
                write(self.current,current)
                with self.assertRaises(ValueError):self.issue()

    def test_acquisition_view_may_not_use_gripper_action(self):
        grip={'action':'grip','pos':100,'speed':40,'force':10,'grip_required':False}
        plan=copy.deepcopy(self.plan);plan['segments'][0]['request']=grip
        self.save_plan(plan)
        with self.assertRaisesRegex(ValueError,'cannot manipulate'):
            self.issue(request=grip)

    def test_acquisition_move_may_not_require_loaded_gripper(self):
        request={**self.request,'grip_required':True}
        plan=copy.deepcopy(self.plan);plan['segments'][0]['request']=request
        self.save_plan(plan)
        with self.assertRaisesRegex(ValueError,'cannot manipulate'):
            self.issue(request=request)

    def test_material_probe_requires_excluded_contact_risk(self):
        # Isolates the acquisition gate; not a claim that upstream artifacts passed.
        state=read(self.task/'state.json');state.update(mode='experiment',stage='properties')
        write(self.task/'state.json',state)
        plan=copy.deepcopy(self.plan);plan['purpose']='material_probe'
        plan['checks']['contact_risks_excluded']=False
        self.save_plan(plan)
        with self.assertRaisesRegex(ValueError,'Contact risks'):
            self.issue()

    def test_duplicate_and_preissued_tickets_are_single_use(self):
        first=self.issue();second=self.issue()
        folder,ticket=s.consume(self.root,self.run,first)
        s.complete(self.root,folder,ticket,{'ok':True})
        for payload in (first,second):
            with self.subTest(ticket=payload['pipeline_ticket']):
                with self.assertRaises(ValueError):s.consume(self.root,self.run,payload)
        with self.assertRaises(ValueError):self.issue()

    def test_plan_changed_after_ticket_issue_rejected_at_consume(self):
        payload=self.issue()
        plan=copy.deepcopy(self.plan);plan['checks']['review_note']='changed after issuing'
        self.save_plan(plan)
        with self.assertRaisesRegex(ValueError,'plan changed'):
            s.consume(self.root,self.run,payload)

    def test_evidence_changed_after_ticket_issue_rejected_at_consume(self):
        payload=self.issue()
        write(self.root/'review_evidence.json',{'changed':True})
        with self.assertRaisesRegex(ValueError,'Evidence changed'):
            s.consume(self.root,self.run,payload)

    def test_expiry_during_evidence_verification_rejects_without_consuming_ticket(self):
        payload=self.issue()
        clock=[time.monotonic_ns()]
        original=s.acquisition_plan
        def slow_verification(*args,**kwargs):
            plan=original(*args,**kwargs)
            clock[0]+=int(90e9)
            return plan
        with patch.object(s,'acquisition_plan',slow_verification), \
             patch.object(s.time,'monotonic_ns',lambda:clock[0]):
            with self.assertRaisesRegex(ValueError,'expired during evidence verification'):
                s.consume(self.root,self.run,payload)
        ticket=read(self.root/payload['pipeline_ticket'])
        self.assertEqual(ticket['state'],'issued')
        self.assertIsNone(read(self.task/'state.json')['inflight'])

    def test_live_acquisition_rechecks_gripper_after_ticket_is_consumed(self):
        payload=self.issue()
        folder,ticket=s.consume(self.root,self.run,payload)
        before=read(self.current)['state']
        s.validate_live_acquisition(self.root,payload,before)
        cases=[{'gOBJ':1,'gPO':100,'gACT':1,'gSTA':3,'gFLT':0},
               {'gOBJ':2,'gPO':120,'gACT':1,'gSTA':3,'gFLT':0},
               None,{}, {'gPO':0},
               {'gOBJ':0,'gPO':0,'gACT':1,'gSTA':3,'gFLT':0}]
        for grip in cases:
            with self.subTest(gripper=grip):
                changed=copy.deepcopy(before);changed['gripper']=grip
                with self.assertRaisesRegex(ValueError,'fresh empty stationary tool feedback'):
                    s.validate_live_acquisition(self.root,payload,changed)
        self.assertEqual(read(folder/'state.json')['inflight']['ticket_id'],ticket['ticket_id'])
    def test_acquisition_failure_halts_and_does_not_reverse(self):
        payload=self.issue()
        folder,ticket=s.consume(self.root,self.run,payload)
        s.complete(self.root,folder,ticket,{'ok':False,'error':'synthetic stop'})
        state=read(folder/'state.json')
        self.assertEqual(state['status'],'halted')
        self.assertIsNone(state['inflight'])
        self.assertTrue((folder/'teacher_request.json').is_file())
        with self.assertRaises(ValueError):self.issue()


if __name__=='__main__':
    unittest.main()

