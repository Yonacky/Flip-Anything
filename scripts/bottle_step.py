"""One bounded action and inspection per invocation. No autonomous retry after a stop."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from flip_anything.camera import OrbbecCamera
from flip_anything.config import load_config
from flip_anything.robot import checked
from flip_anything.gripper import require_existing_grasp
from flip_anything.trial import Trial


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--action',choices=('translate','grip','inspect'),required=True)
    parser.add_argument('--dx',type=float,default=0)
    parser.add_argument('--dy',type=float,default=0)
    parser.add_argument('--dz',type=float,default=0)
    parser.add_argument('--grip-pos',type=int,default=0)
    parser.add_argument('--reason',required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--grip-required',action='store_true')
    args=parser.parse_args()
    if args.action!='inspect' and not args.execute:parser.error('Explicit --execute required')
    delta=[args.dx,args.dy,args.dz]
    pure_retreat = args.dy == 0 and args.dz == 0 and -150 <= args.dx < 0
    pure_lift = args.dx == 0 and args.dy == 0 and 0 < args.dz <= 200
    if not all(math.isfinite(v) for v in delta) or (math.dist(delta,[0,0,0])>50 and not (pure_retreat or pure_lift)):
        parser.error('Translation limited to50mm; pure retreat may be150mm, pure lift200mm')
    if not 0<=args.grip_pos<=180:parser.error('Grip pos limited to 0..180; force is fixed at 10/255')
    cfg=load_config('configs/platform.json')
    trial=Trial(cfg['recording']['root'],cfg,'hardware',context={'task':'green_sprite_container_flip',
        'action':args.action,'decision_summary':args.reason,'delta_base_mm':delta,
        'grip_pos':args.grip_pos,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    from xarm.wrapper import XArmAPI
    arm=XArmAPI(cfg['robot']['ip'],do_not_open=True,is_radian=False)
    camera=OrbbecCamera(cfg['camera'])
    status,error,prepared='failed',None,False
    def call(name,**kwargs):
        trial.event('sdk_request',method=name,kwargs=kwargs)
        result=getattr(arm,name)(**kwargs)
        trial.event('sdk_result',method=name,result=result)
        code=result[0] if isinstance(result,tuple) else result
        if code!=0:raise RuntimeError(f'{name} failed: {code}')
        return result
    def state():
        faults=checked(arm.get_err_warn_code(),'get_err_warn_code')
        s=checked(arm.get_state(),'get_state')
        pose=list(checked(arm.get_position(is_radian=False),'get_position'))
        trial.event('robot_sample',state=s,error_warn=faults,pose_base_mm_deg=pose,
                    joint_deg=list(checked(arm.get_servo_angle(is_radian=False),'get_servo_angle'))[:6])
        if faults[0] or faults[1]:raise RuntimeError(f'Controller faults/warnings: {faults}')
        return s,pose
    def capture(index,phase):
        # Discard startup/buffered frames; only inspect while arm is stationary.
        for _ in range(8):
            if camera.pipeline.wait_for_frames(cfg['camera']['timeout_ms']) is None:
                raise TimeoutError('Camera warmup frame missing')
        frame=camera.capture(trial.path,index)
        trial.event('observation',phase=phase,camera=frame)
        return frame
    try:
        arm.connect()
        if not arm.connected or arm.axis!=6 or arm.is_simulation_robot:
            raise RuntimeError('Expected real connected xArm6')
        trial.event('controller_mode',simulation_robot=arm.is_simulation_robot)
        trial.event('camera_started',metadata=camera.open())
        capture(0,'before')
        initial_state,initial=state()
        if initial_state not in (0,2,4,5):raise RuntimeError(f'Unexpected initial state {initial_state}')
        trial.event('gripper_before',raw=call('robotiq_get_status',check_baud=False),decoded=dict(arm.robotiq_status))
        if args.action!='inspect':
            prepared=True
            call('set_state',state=4)
            call('motion_enable',enable=True)
            call('set_mode',mode=0)
            call('set_state',state=0)
            s,_=state()
            if s in (4,5):raise RuntimeError('Controller not ready after setup')
            if args.grip_required:
                grasp_status=require_existing_grasp(arm)
                trial.event('gripper_guard_read_only',decoded=grasp_status)
        if args.action=='translate':
            target=list(initial)
            for i in range(3):target[i]+=delta[i]
            code,angles=arm.get_inverse_kinematics(target,input_is_radian=False,return_is_radian=False)
            trial.event('ik_check',code=code,joint_deg=angles)
            if code!=0:raise RuntimeError('Target IK failed')
            if checked(arm.is_joint_limit(angles,is_radian=False),'is_joint_limit'):
                raise RuntimeError('IK target exceeds joint limits')
            call('set_position',x=target[0],y=target[1],z=target[2],roll=target[3],pitch=target[4],yaw=target[5],
                 speed=20,mvacc=80,is_radian=False,wait=False)
            deadline=time.monotonic()+15
            stable=0
            while time.monotonic()<deadline:
                s,pose=state()
                if s in (4,5):raise RuntimeError('User/controller stop; aborting this step')
                stable=stable+1 if math.dist(pose[:3],target[:3])<0.8 and s!=1 else 0
                if stable>=2:break
                time.sleep(0.1)
            else:raise TimeoutError('Motion target not reached')
        elif args.action=='grip':
            if arm.robotiq_status['gACT']!=1 or arm.robotiq_status['gSTA']!=3:
                raise RuntimeError('Gripper requires explicit unloaded activation; no automatic activation')
            call('robotiq_set_position',pos=args.grip_pos,speed=60,force=10,wait=False,check_baud=False,auto_enable=False)
            time.sleep(0.2)
            deadline=time.monotonic()+6
            while time.monotonic()<deadline:
                s,_=state()
                if s in (4,5):raise RuntimeError('User/controller stop during gripper action')
                call('robotiq_get_status',check_baud=False)
                gripper=dict(arm.robotiq_status)
                trial.event('gripper_sample',decoded=gripper)
                if gripper['gFLT']!=0:raise RuntimeError(f'Gripper fault {gripper["gFLT"]}')
                if gripper['gOBJ'] in (1,2,3):break
                time.sleep(0.1)
            else:raise TimeoutError('Gripper did not finish')
        capture(1,'after')
        s,final=state()
        trial.event('step_completed',initial_pose=initial,final_pose=final,task_outcome='not_evaluated')
        status='completed'
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'
        trial.event('error',message=error)
    finally:
        if prepared and arm.connected:
            try:call('set_state',state=4)
            except Exception as exc:
                status='failed';error=(error or '')+f'; stop failed: {exc}'
        try:camera.close()
        except Exception as exc:
            status='failed';error=(error or '')+f'; camera stop failed: {exc}'
        arm.disconnect()
        trial.finish(status,error)
    print(json.dumps({'run':str(trial.path),'status':status,'error':error},ensure_ascii=False))
    return 0 if status=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
