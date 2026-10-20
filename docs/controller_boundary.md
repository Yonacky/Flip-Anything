# 控制器单位和姿态边界

`flip_pipeline/controller_boundary.py` 是项目 SI 几何与现有 xArm 执行器之间的唯一转换位置。内部 `base_from_tool` 使用米和旋转矩阵；控制器六元位姿是 `[x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg]`。

姿态定义为 `Rz(yaw) @ Ry(pitch) @ Rx(roll)`，对应 SciPy **小写** `xyz` 固定轴旋转。不能换成大写内禀 `XYZ`。逆变换在接近俯仰 ±90° 时，欧拉角不唯一；转换器返回等价矩阵的一个表示，并标记 `near_euler_singularity`。标记不代表轨迹通过验证。

从 SI 几何生成移动命令：

```python
from flip_pipeline.controller_boundary import matrix_to_controller_move
request = matrix_to_controller_move(
    base_from_tool,
    grip_required=True,
    motion_space="joint",
    speed=chosen_speed_deg_s,
)
```

`motion_space` 和 `speed` 必须显式给出。生成的 move 包含 `frame="base"`、`units="mm_deg"`、`orientation_convention="extrinsic_xyz_rpy"`，以及与插补空间一致的 `speed_units`。缺少这三个姿态标记的旧 move 请求不能直接进入新版执行器。

现有执行器在姿态变化小于 0.5° 时使用 `set_position`，其他情况先调用控制器 IK 再用 `set_servo_angle`。因此同名 `speed` 在两个分支中有不同单位：`cartesian_translation` 对应 `mm_s`，`joint` 对应 `deg_s`。授权时和运动前都必须调用 `validate_motion_against_start(request, fresh_pose)`，核对声明与实际分支；距分支阈值不到 0.000001° 的请求会拒绝，避免数值舍入使验证与执行选择不同分支。

`controller_pose_to_matrix(receipt_pose)` 将明确来自控制器的原始 mm/deg 位姿转回 SI 矩阵。它不能接收来源不明的六元数组并猜单位。

`validate_controller_request(request)` 校验 move/grip 的字段、有限数值、正速度和单位。夹爪 pos/speed/force 是 0..255 的整数寄存器值，不是长度或角速度；小数值会拒绝，不再经过 `int()` 静默截断。夹爪可显式声明 `units="robotiq_registers"`。

模块没有 SDK 调用，不执行 IK，不提供全机械臂碰撞验证。关节插补的轨迹必须由对应路径检查或有记录的教师复核覆盖，不能把 TCP 两个端点之间的直线当成关节运动轨迹。
