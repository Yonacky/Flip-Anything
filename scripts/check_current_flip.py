"""Read-only joint path/table-envelope check using this session's grasp transform."""
import json,argparse
from pathlib import Path
from itertools import product
import numpy as np
from xarm.wrapper import XArmAPI
from flip_anything.geometry import pose_matrix
from flip_anything.robot import checked
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--angle',type=int,choices=[90,180],required=True);a=p.parse_args();r=Path(a.run);plan=json.loads((r/'candidate_flip_plan.json').read_text());arm=XArmAPI('192.168.1.209',do_not_open=True,is_radian=False)
try:
 arm.connect();initial=checked(arm.get_position(is_radian=False),'pose');q0=np.array(checked(arm.get_servo_angle(is_radian=False),'q'))[:6];target=[x for x in plan['rotations'] if x['angle']==a.angle][0]['pose'];target[:3]=initial[:3]
 q=np.array(checked(arm.get_inverse_kinematics(target,input_is_radian=False,return_is_radian=False),'IK'))[:6]
 for i in (0,3,5):q[i]=min([q[i]+360*k for k in (-1,0,1) if -360<=q[i]+360*k<=360],key=lambda v:abs(v-q0[i]))
 if checked(arm.is_joint_limit(q.tolist(),is_radian=False),'limit'):raise RuntimeError('Joint limit')
 X=np.array(json.loads(Path('calibration/geometry_estimate.json').read_text())['T_tcp_camera']);cam=np.array([[x,y,z,1] for x,y,z in product((-90,90),(-25,25),(-40,10))])@X.T;palm=np.array([[x,y,z,1] for x,y,z in product((-60,60),(-60,60),(-174,0))]);bracket=np.array([[*(center+np.array(offset)),1] for center in (X[:3,3],np.array([0,0,-174])) for offset in product((-20,20),repeat=3)])
 n=np.array(plan['table']['normal']);d=plan['table']['d_mm'];bottom=np.array(plan['can_bottom']);h=plan['height_mm'];B=np.array(plan['grasp_pose']);obj=np.array([[*(bottom+z*n+[x,y,0]),1] for x,y,z in product((-35,35),(-35,35),(-6,h+6))])@np.linalg.inv(B).T
 pts=np.concatenate((cam,palm,bracket,obj));samples=[]
 for t in np.linspace(0,1,41):
  pose=checked(arm.get_forward_kinematics((q0+t*(q-q0)).tolist(),input_is_radian=False,return_is_radian=False),'FK');V=pts@np.array(pose_matrix(pose)).T;samples.append({'fraction':float(t),'min_table_clearance_mm':float((V[:,:3]@n+d).min()),'pose':pose})
 result={'initial_pose':initial,'target_pose':target,'initial_joint':q0.tolist(),'target_joint':q.tolist(),'joint_delta':(q-q0).tolist(),'min_clearance_mm':min(x['min_table_clearance_mm'] for x in samples),'margin_mm':30,'samples':samples,'limitations':'current grasp rigid assumption, prior hand-eye, conservative envelope; table-only check'};result['accepted']=result['min_clearance_mm']>50 and float(max(abs(q-q0)))<110
 (r/f'flip_check_{a.angle}.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))
finally:arm.disconnect()
