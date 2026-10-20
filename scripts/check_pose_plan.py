"""Read-only FK sweep of a specific IK path against estimated tabletop/tool envelopes."""
import argparse
import json
from itertools import product
from pathlib import Path
import numpy as np
from xarm.wrapper import XArmAPI
from flip_anything.config import load_config
from flip_anything.geometry import pose_matrix
from flip_anything.robot import checked

p=argparse.ArgumentParser();p.add_argument('--pose',nargs=6,type=float,required=True)
p.add_argument('--max-joint-delta',type=float,default=30)
p.add_argument('--carry-can',action='store_true');a=p.parse_args()
cfg=load_config('configs/platform.json');g=json.loads(Path('calibration/geometry_estimate.json').read_text())
arm=XArmAPI(cfg['robot']['ip'],do_not_open=True,is_radian=False)
try:
 arm.connect();start=np.asarray(checked(arm.get_servo_angle(is_radian=False),'joints'))[:6]
 initial=np.asarray(pose_matrix(checked(arm.get_position(is_radian=False),'pose')))
 target=np.asarray(checked(arm.get_inverse_kinematics(a.pose,input_is_radian=False,return_is_radian=False),'IK'))[:6]
 for i in (0,3,5):
  candidates=[target[i]+360*k for k in (-1,0,1) if -360<=target[i]+360*k<=360]
  target[i]=min(candidates,key=lambda v:abs(v-start[i]))
 if checked(arm.is_joint_limit(target.tolist(),is_radian=False),'limits'):raise RuntimeError('Joint limits')
 X=np.asarray(g['T_tcp_camera']);n=np.asarray(g['table_plane_base']['normal']);d=g['table_plane_base']['d_mm']
 camera=np.array([[x,y,z,1] for x,y,z in product((-90,90),(-25,25),(-40,10))])@X.T
 palm=np.array([[x,y,z,1] for x,y,z in product((-60,60),(-60,60),(-174,0))])
 # Include mount between flange and camera with a 20mm expansion.
 bracket=[]
 for center in (X[:3,3],np.array([0,0,-174])):
  bracket.extend([[*(center+np.array(offset)),1] for offset in product((-20,20),repeat=3)])
 tool=np.concatenate((camera,palm,np.asarray(bracket)))
 if a.carry_can:
  axis=np.asarray(g['can']['axis_base']);u=np.array([1.,0,0]);u-=axis*(u@axis);u/=np.linalg.norm(u);v=np.cross(axis,u)
  O=np.eye(4);O[:3,:3]=np.column_stack((u,v,axis));O[:3,3]=g['can']['bottom_center_base_mm']
  grasp=np.asarray(pose_matrix([608.956970,-19.720247,100.681282,180.00002,85.000008,179.999963]))
  Xcan=np.linalg.inv(grasp)@O
  radius=g['can']['diameter_estimate_mm']/2+8;h=g['can']['height_estimate_mm']+10
  obj=np.asarray([[x,y,z,1] for x,y,z in product((-radius,radius),(-radius,radius),(-5,h))])@Xcan.T
  tool=np.concatenate((tool,obj))
 samples=[]
 for fraction in np.linspace(0,1,21):
  q=start+fraction*(target-start)
  pose=checked(arm.get_forward_kinematics(q.tolist(),input_is_radian=False,return_is_radian=False),'FK')
  B=np.asarray(pose_matrix(pose));pts=tool@B.T
  clearance=float(np.min(pts[:,:3]@n+d))
  samples.append({'fraction':float(fraction),'pose':pose,'tool_table_clearance_mm':clearance})
 R=np.asarray(pose_matrix(a.pose))[:3,:3]
 angle=float(np.degrees(np.arccos(np.clip((np.trace(initial[:3,:3].T@R)-1)/2,-1,1))))
 minimum=min(s['tool_table_clearance_mm'] for s in samples)
 result={'target_pose':a.pose,'initial_joint_deg':start.tolist(),'target_joint_deg':target.tolist(),
  'joint_delta_deg':(target-start).tolist(),'actual_rotation_delta_deg':angle,
  'minimum_modeled_tool_table_clearance_mm':minimum,'uncertainty_margin_mm':30,
  'remaining_clearance_after_margin_mm':minimum-30,'fk_samples':samples,
  'accepted':bool(minimum-30>20 and np.max(np.abs(target-start))<a.max_joint_delta),
  'carry_can_in_envelope':a.carry_can,
  'scope':'Specific observation-pose adjustment only; estimated end-tool envelope, not full scene collision certification'}
 Path('calibration/pose_plan_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k!='fk_samples'},ensure_ascii=False,indent=2))
finally:arm.disconnect()
