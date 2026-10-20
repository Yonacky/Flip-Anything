import copy,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from flip_pipeline.storage import write,read,digest
from flip_pipeline import workflow as w
from flip_pipeline import supervision as s
from flip_pipeline.planning import propose_rigid_flip,box_vertices,support_margin
from flip_pipeline.simulation import simulate_placement

class PipelineTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
  self.task=w.new_task(self.root,"object1","install1")
  write(self.root/"evidence.json",{"measured":"fixture"});write(self.root/"profile.json",{});write(self.root/"installation.json",{})
 def tearDown(self):self.tmp.cleanup()
 def report(self,stage):
  data={"source":"physical_evidence","checks":{k:{"result":"pass","method":sorted(w.ALLOWED_METHODS[k])[0],"evidence":["evidence.json"],"detail":"test fixture: contractual validation, not physical certification"} for k in w.REQUIRED[stage]}}
  if stage=="calibration":data.update(profile_path="profile.json",installation_path="installation.json")
  if stage=="properties":data.update(hazards={k:"excluded" for k in ("spill","breakage","sharp_or_hot")},mass_status="measured")
  if stage=="plan":data.update(execution_path="bounded_trial",path_scope="robot_tool_object",segments=[{"id":"one","role":"transfer","request":{"action":"move","pose":[0,0,0,0,0,0],"grip_required":True,"frame":"base","units":"mm_deg","orientation_convention":"extrinsic_xyz_rpy","motion_space":"cartesian_translation","speed":10,"speed_units":"mm_s"}}])
  return data
 def attach(self,stage,data=None):
  with patch("flip_pipeline.calibration.validate_device_profile",return_value={"installation_id":"install1"}), patch("flip_pipeline.evaluation.validate_numeric_report",return_value={"valid":True}), patch("flip_pipeline.plan_evaluation.validate_plan_report",return_value={"valid":True}):
   return w.attach_report(self.root,self.task,stage,data or self.report(stage),["evidence.json","profile.json","installation.json"])
 def ready(self):
  for stage in w.STAGES[:4]:self.attach(stage)
  self.run="DS_fixture";write(self.root/"ds_control/active.json",{"run":self.run})
  self.ref=self.root/"logs"/self.run/"result_1.json"
  write(self.ref,{"ok":True,"state":{"pose":[0,0,0,0,0,0],"sample_end_monotonic_ns":time.monotonic_ns(),"gripper":{"gPO":130,"gOBJ":2}}})
  self.request=self.report("plan")["segments"][0]["request"]
 def test_cannot_skip_stage(self):
  with self.assertRaises(ValueError):self.attach("plan")
 def test_hazard_unknown_blocks(self):
  self.attach("calibration");self.attach("model");data=self.report("properties");data["hazards"]["spill"]="unknown"
  with self.assertRaises(ValueError):self.attach("properties",data)
 def test_changed_evidence_blocks(self):
  self.attach("calibration");write(self.root/"evidence.json",{"altered":True})
  self.assertIn("Evidence changed",w.status(self.root,self.task)["integrity_error"])
 def test_recomputed_upstream_invalidates_downstream(self):
  self.ready();self.attach("model")
  self.assertEqual(set(read(self.task/"state.json")["artifacts"]),{"calibration","model"})
 def test_single_use_and_duplicate_preissued_ticket(self):
  self.ready();a=s.issue(self.root,self.task,self.run,"one",self.request,self.ref);b=s.issue(self.root,self.task,self.run,"one",self.request,self.ref)
  folder,ticket=s.consume(self.root,self.run,a);s.complete(self.root,folder,ticket,{"ok":True})
  for payload in (a,b):
   with self.assertRaises(ValueError):s.consume(self.root,self.run,payload)
 def test_mutated_command_rejected(self):
  self.ready();a=s.issue(self.root,self.task,self.run,"one",self.request,self.ref);a["pose"][0]=10
  with self.assertRaises(ValueError):s.consume(self.root,self.run,a)
 def test_edited_receipt_rejected(self):
  self.ready();a=s.issue(self.root,self.task,self.run,"one",self.request,self.ref);write(self.ref,{"ok":True})
  with self.assertRaises(ValueError):s.consume(self.root,self.run,a)
 def test_failure_halts_without_reverse_or_release(self):
  self.ready();a=s.issue(self.root,self.task,self.run,"one",self.request,self.ref)
  def fail():raise RuntimeError("external stop")
  with self.assertRaises(RuntimeError):s.execute_guarded(self.root,self.run,a,fail)
  state=read(self.task/"state.json");self.assertEqual(state["status"],"halted");self.assertTrue((self.task/"teacher_request.json").exists());self.assertIsNone(state["inflight"])
 def test_opening_cannot_hide_behind_role(self):
  self.ready();p=self.report("plan");p["segments"][0]["request"]={"action":"grip","pos":0,"grip_required":True};self.attach("plan",p)
  with self.assertRaises(ValueError):s.issue(self.root,self.task,self.run,"one",p["segments"][0]["request"],self.ref)
 def test_stale_receipt(self):
  self.ready();r=read(self.ref);r["state"]["sample_end_monotonic_ns"]-=int(120e9);write(self.ref,r)
  with self.assertRaises(ValueError):s.issue(self.root,self.task,self.run,"one",self.request,self.ref)

class GeometryTests(unittest.TestCase):
 def test_flip_uses_tool_body_in_placement(self):
  B=np.eye(4);B[2,3]=.1;G=np.eye(4)
  obj=box_vertices([0,0,0],[.03,.03,.1]);tool=box_vertices([0,0,0],[.02,.02,.02])
  result=propose_rigid_flip(B,G,obj,tool,[0,0,1,0],.01,[1,0,0])
  self.assertGreaterEqual(result["placement_tool_clearance_m"],.01)
  badtool=box_vertices([0,0,.15],[.02,.02,.05])
  with self.assertRaises(ValueError):propose_rigid_flip(B,G,obj,badtool,[0,0,1,0],.01,[1,0,0])
 def test_static_support_margin(self):
  polygon=[[-1,-1],[1,-1],[1,1],[-1,1]]
  self.assertFalse(support_margin([1.1,0],polygon)["stable_under_static_assumptions"])
  self.assertAlmostEqual(support_margin([0,0],polygon,.1)["minimum_margin_m"],.9)
 def test_real_mujoco_stable_box(self):
  try:import mujoco
  except ImportError:self.skipTest("MuJoCo optional backend not installed")
  T=np.eye(4);T[2,3]=.0501
  result=simulate_placement({"kind":"box","contents":"known_rigid","mass_source":"hypothesis","mass_kg":1,"com_m":[0,0,0],"inertia_kg_m2":[.0017]*3,"inertia_frame":"object_com_principal_axes","friction":[.5,.005,.0001],"size_m":[.05]*3,"base_from_object":T.tolist(),"table_from_base":np.eye(4).tolist(),"units":"m_kg_s","world_frame":"table_z0","gravity_base_m_s2":[0,0,-9.81],"evidence":["synthetic_fixture"],"duration_s":.5})
  self.assertGreater(result["contacts"],0);self.assertLess(result["orientation_change_rad"],.01);self.assertEqual(result["mass_source"],"hypothesis")
 def test_no_unknown_contents_simulation(self):
  try:import mujoco
  except ImportError:self.skipTest("MuJoCo optional backend not installed")
  with self.assertRaises(ValueError):simulate_placement({"kind":"box","contents":"unknown"})
if __name__=="__main__":unittest.main()
