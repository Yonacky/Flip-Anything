import json
from xarm.wrapper import XArmAPI
arm = XArmAPI('192.168.1.209', do_not_open=True, is_radian=False)
arm.connect()
print('connected', arm.connected, 'axis', arm.axis, 'sim', arm.is_simulation_robot, 'version', arm.version)
print('tcp_offset', arm.tcp_offset, 'tcp_load', arm.tcp_load)
print('state', arm.get_state(), 'err_warn', arm.get_err_warn_code())
print('pose', arm.get_position(is_radian=False))
print('joints', arm.get_servo_angle(is_radian=False))
q = arm.get_servo_angle(is_radian=False)[1][:6]
fk = arm.get_forward_kinematics(q, input_is_radian=False, return_is_radian=False)
print('FK(joints)', fk)
print('warn_code', arm.warn_code, 'error_code', arm.error_code)
print('collision_sensitivity', arm.collision_sensitivity, 'tcp_maxacc', getattr(arm, 'tcp_maxacc', None))
arm.disconnect()
