"""一次性：读取控制器的错误/警告码，清除已授权的历史警告（不做任何运动）。"""
import time
from xarm.wrapper import XArmAPI

arm = XArmAPI("192.168.1.209", is_radian=False, do_not_open=True)
arm.connect()
print("connected:", arm.connected, "state:", arm.state, "error_warn:", arm.get_err_warn_code())
if arm.state != 4:
    arm.set_state(4)
    time.sleep(0.5)
print("clean_warn ->", arm.clean_warn())
time.sleep(0.5)
print("error_warn after:", arm.get_err_warn_code())
if arm.get_err_warn_code()[1]:
    print("clean_error ->", arm.clean_error())
    time.sleep(0.5)
    print("error_warn after clean_error:", arm.get_err_warn_code())
print("final state:", arm.state, "pose:", arm.get_position(is_radian=False))
arm.disconnect()