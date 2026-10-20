"""Collect stationary RGB-D/robot pairs at bounded rotations for local hand-eye estimation."""
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from xarm.wrapper import XArmAPI
from flip_anything.camera import OrbbecCamera
from flip_anything.config import load_config
from flip_anything.geometry import pose_matrix
from flip_anything.robot import checked
from flip_anything.trial import Trial

cfg=load_config('configs/platform.json')
trial=Trial(cfg['recording']['root'],cfg,'hardware',context={'task':'local_rgbd_hand_eye_and_object_geometry',
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'max_axis_rotation_deg':12,
    'decision_summary':'Retreated and open-jaw observation position; infer geometry from static multi-view RGB-D, no web or object-size input.'})
arm=XArmAPI(cfg['robot']['ip'],do_not_open=True,is_radian=False)
camera=OrbbecCamera(cfg['camera'])
samples=[]
status,error,prepared='failed',None,False

def call(name,**kwargs):
    trial.event('sdk_request',method=name,kwargs=kwargs)
    result=getattr(arm,name)(**kwargs)
    trial.event('sdk_result',method=name,result=result)
    code=result[0] if isinstance(result,tuple) else result
    if code!=0:raise RuntimeError(f'{name}: {code}')
    return result

def state():
    state=checked(arm.get_state(),'get_state')
    faults=checked(arm.get_err_warn_code(),'get_err_warn_code')
    if any(faults):raise RuntimeError(f'Controller faults: {faults}')
    return state,np.asarray(checked(arm.get_position(is_radian=False),'get_position'))

def settle_capture(index,label):
    for _ in range(18):
        if camera.pipeline.wait_for_frames(1500) is None:raise TimeoutError('Missing stationary frame')
    s,pose=state()
    if s==1:raise RuntimeError('Arm still moving at capture')
    frame=camera.capture(trial.path,index)
    sample={'index':index,'label':label,'pose_base_tcp_mm_deg':pose.tolist(),
       'T_base_tcp':pose_matrix(pose.tolist()),'camera':frame}
    samples.append(sample)
    trial.event('calibration_observation',**sample)
    (trial.path/'geometry_samples.json').write_text(json.dumps({'camera_metadata':metadata,'samples':samples},indent=2),encoding='utf-8')

try:
    arm.connect()
    if not arm.connected or arm.axis!=6 or arm.is_simulation_robot:raise RuntimeError('Expected real xArm6')
    s,reference_pose=state()
    if s not in (0,2,4,5):raise RuntimeError(f'Unexpected state {s}')
    reference=np.asarray(pose_matrix(reference_pose.tolist()))
    if abs(reference_pose[0]-470.022)>2 or abs(reference_pose[2]-119.379)>2:
        raise RuntimeError('Observation position changed; require fresh scene inspection')
    metadata=camera.open()
    trial.event('camera_started',metadata=metadata)
    # First pose is baseline. Last is a held-out view, followed by return.
    plans=[('reference',[0,0,0]),('yaw_plus',[0,0,12]),('pitch_plus',[0,12,0]),
           ('roll_plus',[12,0,0]),('mixed',[-8,8,-8]),('heldout',[6,-10,6]),('returned',[0,0,0])]
    targets=[]
    for label,rotation in plans:
        delta=cv2.Rodrigues(np.radians(rotation).astype(float))[0]
        target=reference.copy();target[:3,:3]=delta@reference[:3,:3]
        rv=np.degrees(cv2.Rodrigues(target[:3,:3])[0].reshape(3))
        targets.append((label,target,rv))
    prepared=True
    call('set_state',state=4);call('motion_enable',enable=True);call('set_mode',mode=0);call('set_state',state=0)
    for index,(label,target,rv) in enumerate(targets):
        trial.event('planned_calibration_view',label=label,target_matrix=target.tolist())
        if index:
            request=reference_pose[:3].tolist()+rv.tolist()
            call('set_position_aa',axis_angle_pose=request,speed=10,mvacc=40,is_radian=False,wait=False)
            deadline=time.monotonic()+15;stable=0
            while time.monotonic()<deadline:
                s,pose=state()
                if s in (4,5):raise RuntimeError('User/controller stop; no additional poses sent')
                actual=np.asarray(pose_matrix(pose.tolist()))
                angle=math.degrees(math.acos(float(np.clip((np.trace(target[:3,:3].T@actual[:3,:3])-1)/2,-1,1))))
                stable=stable+1 if angle<0.3 and np.linalg.norm(actual[:3,3]-target[:3,3])<1 and s!=1 else 0
                if stable>=2:break
                time.sleep(0.1)
            else:raise TimeoutError('Calibration view target timeout')
        settle_capture(index,label)
        print(json.dumps({'view':label,'index':index},ensure_ascii=False),flush=True)
    status='completed'
except BaseException as exc:
    error=f'{type(exc).__name__}: {exc}';trial.event('error',message=error)
finally:
    if prepared and arm.connected:
        try:call('set_state',state=4)
        except Exception as exc:status='failed';error=(error or '')+f'; stop: {exc}'
    try:camera.close()
    except Exception as exc:status='failed';error=(error or '')+f'; camera: {exc}'
    arm.disconnect();trial.finish(status,error)
print(json.dumps({'run':str(trial.path),'status':status,'error':error},ensure_ascii=False))
raise SystemExit(0 if status=='completed' else 1)
