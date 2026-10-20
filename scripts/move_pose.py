"""Logged, explicitly requested pose step using nearest valid IK joint solution."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from xarm.wrapper import XArmAPI
from flip_anything.camera import OrbbecCamera
from flip_anything.config import load_config
from flip_anything.geometry import pose_matrix
from flip_anything.robot import checked
from flip_anything.gripper import require_existing_grasp
from flip_anything.trial import Trial

p=argparse.ArgumentParser();p.add_argument('--pose',nargs=6,type=float,required=True)
p.add_argument('--reason',required=True);p.add_argument('--grip-required',action='store_true')
p.add_argument('--execute',action='store_true');a=p.parse_args()
if not a.execute or not all(math.isfinite(x) for x in a.pose):p.error('Explicit execution and finite pose required')
cfg=load_config('configs/platform.json')
trial=Trial(cfg['recording']['root'],cfg,'hardware',context={'task':'modeled_can_flip','target_pose':a.pose,
  'decision_summary':a.reason,'grip_required':a.grip_required,
  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
plan_file=Path('calibration/pose_plan_check.json')
if plan_file.exists():
    plan=json.loads(plan_file.read_text())
    if np.allclose(plan['target_pose'],a.pose):
        trial.event('precomputed_fk_envelope_check',check=plan)
arm=XArmAPI(cfg['robot']['ip'],is_radian=False,do_not_open=True);cam=OrbbecCamera(cfg['camera'])
status,error,prepared='failed',None,False
def call(name,**kwargs):
 trial.event('sdk_request',method=name,kwargs=kwargs)
 result=getattr(arm,name)(**kwargs);trial.event('sdk_result',method=name,result=result)
 code=result[0] if isinstance(result,tuple) else result
 if code!=0:raise RuntimeError(f'{name}:{code}')
 return result
def read():
 s=checked(arm.get_state(),'state');faults=checked(arm.get_err_warn_code(),'faults')
 pose=np.asarray(checked(arm.get_position(is_radian=False),'pose'))
 joints=np.asarray(checked(arm.get_servo_angle(is_radian=False),'joints'))[:6]
 trial.event('robot_sample',state=s,error_warn=faults,pose=pose.tolist(),joint_deg=joints.tolist())
 if any(faults):raise RuntimeError(f'Controller fault/warn{faults}')
 return s,pose,joints
def capture(index,phase):
 for _ in range(14):
  if cam.pipeline.wait_for_frames(1500) is None:raise TimeoutError('Missing camera frame')
 trial.event('observation',phase=phase,camera=cam.capture(trial.path,index))
try:
 arm.connect()
 if not arm.connected or arm.axis!=6 or arm.is_simulation_robot:raise RuntimeError('Not real connected xArm6')
 trial.event('camera_started',metadata=cam.open());capture(0,'before')
 s,initial,joints=read()
 if s not in (0,2,4,5):raise RuntimeError(f'Initial state{s}')
 if np.linalg.norm(np.asarray(a.pose[:3])-initial[:3])>180:raise RuntimeError('Translation exceeds180mm pose-step bound')
 target=np.asarray(checked(arm.get_inverse_kinematics(a.pose,input_is_radian=False,return_is_radian=False),'IK'))[:6]
 for i in (0,3,5):
  candidates=[target[i]+360*k for k in (-1,0,1)]
  candidates=[v for v in candidates if -360<=v<=360]
  target[i]=min(candidates,key=lambda v:abs(v-joints[i]))
 if checked(arm.is_joint_limit(target.tolist(),is_radian=False),'joint limit'):raise RuntimeError('Joint limits')
 if np.max(np.abs(target-joints))>110:raise RuntimeError('IK branch jump exceeds110deg')
 trial.event('ik_preflight',initial_joint_deg=joints.tolist(),target_joint_deg=target.tolist())
 prepared=True;call('set_state',state=4);call('motion_enable',enable=True);call('set_mode',mode=0);call('set_state',state=0)
 if a.grip_required:
  grasp_status=require_existing_grasp(arm)
  trial.event('gripper_guard_read_only',decoded=grasp_status)
 call('set_servo_angle',angle=target.tolist(),speed=8,mvacc=20,is_radian=False,wait=False)
 deadline=time.monotonic()+max(20,float(np.max(np.abs(target-joints))/8+10));stable=0
 while time.monotonic()<deadline:
  s,pose,current=read()
  if s in (4,5):raise RuntimeError('User/controller stop; aborting')
  stable=stable+1 if np.max(np.abs(current-target))<0.2 and s!=1 else 0
  if stable>=2:break
  time.sleep(0.1)
 else:raise TimeoutError('Pose target timeout')
 final_matrix=np.asarray(pose_matrix(pose.tolist()));target_matrix=np.asarray(pose_matrix(a.pose))
 position_error=float(np.linalg.norm(pose[:3]-np.asarray(a.pose[:3])))
 angle_error=float(np.degrees(np.arccos(np.clip((np.trace(final_matrix[:3,:3].T@target_matrix[:3,:3])-1)/2,-1,1))))
 if position_error>2 or angle_error>1:raise RuntimeError('Final FK pose disagrees with request')
 capture(1,'after');trial.event('pose_completed',pose=pose.tolist(),position_error_mm=position_error,angle_error_deg=angle_error)
 status='completed'
except BaseException as exc:
 error=f'{type(exc).__name__}:{exc}';trial.event('error',message=error)
finally:
 if prepared and arm.connected:
  try:call('set_state',state=4)
  except Exception as exc:status='failed';error=(error or '')+f';stop:{exc}'
 try:cam.close()
 except Exception as exc:status='failed';error=(error or '')+f';camera:{exc}'
 arm.disconnect();trial.finish(status,error)
print(json.dumps({'run':str(trial.path),'status':status,'error':error},ensure_ascii=False))
raise SystemExit(0 if status=='completed' else 1)
