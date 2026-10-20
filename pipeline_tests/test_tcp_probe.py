"""Bounded TCP contact acquisition, synthetic only, no hardware."""
import copy
from pipeline_tests import test_acquisition as fixture
from flip_pipeline import supervision as s
from flip_pipeline.storage import read,write

class TCPProbeTests(fixture.AcquisitionTests):
    def tcp(self,request):
        plan=copy.deepcopy(self.plan);plan['purpose']='tcp_grip_probe';plan['checks'].update(contact_risks_excluded=True,tcp_reference_reviewed=True);plan['segments'][0]['request']=request;self.save_plan(plan)
    def test_tcp_open_accepts_idle_activation_feedback(self):
        req={'action':'grip','pos':0,'speed':30,'force':5,'grip_required':False};self.tcp(req)
        q=read(self.current);q['state']['gripper'].update(gOBJ=0,gGTO=0);write(self.current,q)
        payload=self.issue(req);s.validate_live_acquisition(self.root,payload,q['state'])
    def test_tcp_contact_close_accepts_reviewed_empty_start(self):
        req={'action':'grip','pos':255,'speed':30,'force':5,'grip_required':False};self.tcp(req);self.issue(req)
    def test_tcp_no_unreviewed_reference(self):
        req={'action':'grip','pos':255,'speed':30,'force':5,'grip_required':False};self.tcp(req);p=read(self.plan_path);p['checks']['tcp_reference_reviewed']=False;self.save_plan(p)
        with self.assertRaisesRegex(ValueError,'reference'):self.issue(req)
    def test_tcp_cannot_move_held_or_unknown_gripper(self):
        req={**self.request,'max_translation_mm':10,'max_rotation_deg':.49,'max_joint_deg':8};self.tcp(req)
        for state in [0,1,2]:
            q=read(self.current);q['state']['gripper']['gOBJ']=state;write(self.current,q)
            with self.assertRaisesRegex(ValueError,'empty'):self.issue(req)
    def test_tcp_rejects_rotation_and_fast_grip(self):
        req={'action':'grip','pos':255,'speed':41,'force':5,'grip_required':False};self.tcp(req)
        with self.assertRaisesRegex(ValueError,'bounds'):self.issue(req)
        req={**self.request,'motion_space':'joint','speed_units':'deg_s','max_translation_mm':10,'max_rotation_deg':.49,'max_joint_deg':8};self.tcp(req)
        with self.assertRaisesRegex(ValueError,'bounds'):self.issue(req)
