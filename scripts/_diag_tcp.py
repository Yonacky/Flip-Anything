from xarm.wrapper import XArmAPI
arm = XArmAPI('192.168.1.209', do_not_open=True, is_radian=False)
arm.connect()
print('get_tcp_offset', arm.get_tcp_offset())
print('get_tcp_load', arm.get_tcp_load())
print('get_state', arm.get_state())
arm.disconnect()
