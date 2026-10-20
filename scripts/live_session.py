"""Single connection, explicit file commands, read-only gripper polling, logged bounded steps."""
import json,time,math,argparse,hashlib
from pathlib import Path
import numpy as np
from xarm.wrapper import XArmAPI
from flip_anything.camera import OrbbecCamera
from flip_anything.config import load_config
from flip_anything.geometry import pose_matrix
from flip_anything.robot import checked
from flip_anything.gripper import require_existing_grasp
from flip_anything.trial import Trial
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
cfg=load_config('configs/platform.json');trial=Trial(cfg['recording']['root'],cfg,'hardware',a.run_id,{'purpose':'development_source_collaborative','object_id':'D_green_can','success_definition':'inverted on table, released, stable >=3s; visual confirmation required','prior_human_strategy_exposure':True,'model_id':'not available via runtime','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
folder=trial.path;inbox=folder/'commands';inbox.mkdir();arm=XArmAPI(cfg['robot']['ip'],do_not_open=True,is_radian=False);cam=OrbbecCamera(cfg['camera']);idx=0;done=set();failed=False

def call(method,**kw):
 trial.event('sdk_request',method=method,kwargs=kw);result=getattr(arm,method)(**kw);trial.event('sdk_result',method=method,result=result)
 code=result[0] if isinstance(result,tuple) else result
 if code!=0:raise RuntimeError(f'{method}: {code}')
 return result

def state(log=False):
 s=checked(arm.get_state(),'state');faults=checked(arm.get_err_warn_code(),'faults');pose=list(checked(arm.get_position(is_radian=False),'pose'));joints=list(checked(arm.get_servo_angle(is_radian=False),'joints'))[:6]
 checked(arm.robotiq_get_status(check_baud=False),'gripper read');grip=dict(arm.robotiq_status)
 result={'state':s,'error_warn':faults,'pose':pose,'joint_deg':joints,'gripper':grip}
 if log:trial.event('robot_sample',**result)
 (folder/'live_state.json').write_text(json.dumps(result,indent=2))
 if any(faults):raise RuntimeError(f'Controller faults {faults}')
 return result

def capture(label):
 global idx
 for _ in range(8):
  if cam.pipeline.wait_for_frames(1500) is None:raise TimeoutError('Camera frame')
  state()
 frame=cam.capture(folder,idx);idx+=1;trial.event('observation',phase=label,camera=frame);return frame

def command(c):
 before=state(True);trial.event('command',request=c)
 if c['action']=='inspect':return {'state':before,'camera':capture('inspect')}
 if c['action']=='shutdown':return {'shutdown':True}
 if c['action'] not in ('move','grip'):raise ValueError('Unknown action')
 if before['state'] not in (0,2,4):raise RuntimeError('Controller not in expected idle/stop state')
 if np.linalg.norm(np.array(before['pose'][:3])-c['expected_pose'][:3])>2:raise RuntimeError('Expected initial position changed')
 R0=np.array(pose_matrix(before['pose']))[:3,:3];Re=np.array(pose_matrix(c['expected_pose']))[:3,:3]
 if np.degrees(np.arccos(np.clip((np.trace(R0.T@Re)-1)/2,-1,1)))>1:raise RuntimeError('Expected initial orientation changed')
 if c.get('grip_required'):require_existing_grasp(arm)
 if c['action']=='move':
  target=np.array(c['pose'],float)
  if not np.isfinite(target).all() or np.linalg.norm(target[:3]-before['pose'][:3])>160:raise RuntimeError('Pose step bound')
  R=np.array(pose_matrix(target.tolist()))[:3,:3];angle=np.degrees(np.arccos(np.clip((np.trace(R0.T@R)-1)/2,-1,1)))
  if angle>95:raise RuntimeError('Rotation step bound')
  q=np.array(checked(arm.get_inverse_kinematics(target.tolist(),input_is_radian=False,return_is_radian=False),'IK'))[:6]
  for i in (0,3,5):q[i]=min([q[i]+360*k for k in (-1,0,1) if -360<=q[i]+360*k<=360],key=lambda v:abs(v-before['joint_deg'][i]))
  if max(abs(q-np.array(before['joint_deg'])))>110 or checked(arm.is_joint_limit(q.tolist(),is_radian=False),'limits'):raise RuntimeError('Joint limits/branch jump')
 elif not 0<=c['pos']<=180:raise RuntimeError('Grip position bound')
 if c['action']=='grip' and (before['gripper']['gACT']!=1 or before['gripper']['gSTA']!=3):raise RuntimeError('No automatic activation')
 call('set_state',state=4);call('motion_enable',enable=True);call('set_mode',mode=0);call('set_state',state=0)
 try:
  if c['action']=='grip':call('robotiq_set_position',pos=c['pos'],speed=40,force=10,wait=False,check_baud=False,auto_enable=False)
  elif angle<.1:call('set_position',x=float(target[0]),y=float(target[1]),z=float(target[2]),roll=float(target[3]),pitch=float(target[4]),yaw=float(target[5]),speed=min(c.get('speed',15),20),mvacc=50,is_radian=False,wait=False)
  else:call('set_servo_angle',angle=q.tolist(),speed=6,mvacc=15,is_radian=False,wait=False)
  deadline=time.monotonic()+35;stable=0
  time.sleep(.2)
  while time.monotonic()<deadline:
   s=state(True)
   if s['state'] in (4,5):raise RuntimeError('External stop; abort')
   if c.get('grip_required') and (s['gripper']['gFLT'] or s['gripper']['gOBJ'] not in (1,2)):raise RuntimeError('Held-object status lost')
   if c['action']=='grip':
    if s['gripper']['gFLT']:raise RuntimeError(f"Gripper fault {s['gripper']['gFLT']}")
    reached=s['gripper']['gOBJ'] in (1,2,3)
   else:reached=np.linalg.norm(np.array(s['pose'][:3])-target[:3])<.8 and max(abs(np.array(s['joint_deg'])-q))<.3 and s['state']!=1
   stable=stable+1 if reached else 0
   if stable>=3:break
   time.sleep(.12)
  else:raise TimeoutError('Step timeout')
 finally:call('set_state',state=4)
 return {'state':state(True),'camera':capture('after_'+c['action'])}

try:
 arm.connect()
 if not arm.connected or arm.axis!=6 or arm.is_simulation_robot:raise RuntimeError('Expected real xArm6')
 trial.event('camera_started',metadata=cam.open());capture('session_start');state(True);print(json.dumps({'ready':str(folder)}),flush=True)
 expiry=time.monotonic()+1800;lastlog=0
 while time.monotonic()<expiry:
  state(time.monotonic()-lastlog>1)
  if time.monotonic()-lastlog>1:lastlog=time.monotonic()
  pending=[f for f in sorted(inbox.glob('*.json')) if f.name not in done]
  if pending:
   f=pending[0];done.add(f.name)
   try:result=command(json.loads(f.read_text(encoding='utf-8-sig')));result['ok']=True
   except Exception as exc:
    result={'ok':False,'error':f'{type(exc).__name__}: {exc}'};trial.event('error',**result);failed=True
    call('set_state',state=4)
   (folder/('result_'+f.name)).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
   if failed or result.get('shutdown'):break
  time.sleep(.2)
except BaseException as exc:
 failed=True;trial.event('error',message=f'{type(exc).__name__}: {exc}');print(str(exc),flush=True)
finally:
 if arm.connected:
  try:call('set_state',state=4)
  except Exception:pass
 cam.close();arm.disconnect();trial.finish('failed' if failed else 'completed',None)
