"""Record available geometry without fabricating missing calibration or object poses."""
import json
from datetime import datetime, timezone
from pathlib import Path

from xarm.wrapper import XArmAPI
from flip_anything.config import load_config
from flip_anything.geometry import pose_matrix
from flip_anything.robot import checked

cfg=load_config('configs/platform.json')
arm=XArmAPI(cfg['robot']['ip'],do_not_open=True,is_radian=False)
try:
    arm.connect()
    pose=list(checked(arm.get_position(is_radian=False),'get_position'))
    angles=list(checked(arm.get_servo_angle(is_radian=False),'get_servo_angle'))[:6]
    gripper_raw=arm.robotiq_get_status(check_baud=False)
    scene={
      'schema_version':1,'updated_utc':datetime.now(timezone.utc).isoformat(),
      'coordinate_convention':{'translation':'mm','angles':'deg','world_frame':'robot_base'},
      'robot':{'model':'xArm6','source':'SDK controller readback','joint_deg':angles,
        'pose_base_tcp_mm_deg':pose,'T_base_tcp':pose_matrix(pose),'state':arm.get_state(),
        'error_warn':arm.get_err_warn_code(),'simulation':arm.is_simulation_robot,
        'tcp_offset_reported_mm_deg':list(arm.tcp_offset),'tcp_definition_verified':False},
      'gripper':{'model':'Robotiq 2F-85','raw':gripper_raw,'decoded':dict(arm.robotiq_status),
        'encoder_to_opening_calibrated':False,'contact_pad_geometry_tcp':None},
      'camera':{'model':'Gemini336','serial':'CP9E1630019L',
        'body_dimensions_whd_mm':[90,25,30.7],
        'body_dimensions_source':'https://store.orbbec.com/products/gemini-336',
        'bracket_height_mm':70,'bracket_height_source':'user approximate measurement',
        'T_tcp_camera':None,'mount_and_body_collision_vertices_tcp':None,
        'cable_in_collision_model':False,'cable_exclusion_source':'user request'},
      'table':{'plane_in_base':None,'source':'unmeasured'},
      'object':{'description':'green Sprite can','initial_axis':'visually vertical',
        'height_mm':None,'diameter_mm':None,'T_base_object':None,
        'T_tcp_object_after_grasp':None,'grasp_verified_by_lift':False},
      'planning':{'collision_check_ready':False,'flip_executed':False,
        'missing':['camera-to-TCP transform and full mounting envelope',
          'table plane in base frame','can dimensions and grasp transform'],
        'prior_approach_method':'bounded image-guided trial; not calibrated 3D planning'}
    }
    output=Path('calibration/scene_current.json')
    output.write_text(json.dumps(scene,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'scene_file':str(output.resolve()),'robot_pose':pose,
       'state':scene['robot']['state'],'gripper':scene['gripper']['decoded'],
       'collision_check_ready':False},ensure_ascii=False))
finally:
    arm.disconnect()
