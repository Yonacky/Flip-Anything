# 当前实验环境：xArm6 / Gemini 336 / Robotiq 2F85

此版本收录 2026-10-19 的安装测量与 2026-10-20 的重复接触标定。运行 `context.ps1 context --project flip_anything` 读取经过文件校验的完整数值，按 environment.components 定位各组件。

当前可复用：软件操作抓取参考点、原手眼档案、桌面平面和开口相关名义工具模型。每个组件保留自己的验证状态。完整安装仍是 draft；本包发布表示资料已整理可读取，不表示 pipeline 全局标定门槛通过。

软件抓取参考点相对控制器 TCP 的平移为 [0, 1.8568452814711213, -16.99214606154837] mm，方向沿用控制器工具轴。控制器原偏置 [0,0,174,0,0,0] mm/deg 保持不变。计算使用米/弧度与 parent_from_child 矩阵：

`base_from_controller_tcp = base_from_desired_grasp_reference @ inverse(controller_tcp_from_operational_grasp_reference)`

不要将软件偏移写入控制器；不要重复补偿。抓住后物体相对夹持参考点的变换由本次夹持观察确定。

两次接触验证对应位置寄存器 84/83、TCP 高度 90/100 mm，参考点差 2.73 mm；5 mm 是工程规划余量。验证对象是本次直立刚性罐，估计直径 56.25 mm。未做负载抬升、持物旋转和全开口范围接触验证。新物体重新定位并选择插入深度，不能照搬这些历史位置和寄存器值。

局部相机旋转修正仅用于本次重建，没有替换全局手眼旋转/平移。桌面文件的 measured_scene_draft 状态保留，可见平面不代表整个空间无障碍。用户对工具尺量的认可保存在原始材料中，不能扩展成未做过的验证。

执行前匹配机械臂 XI130511J56A15、相机 CP9E1630019L、工具和安装版本、TCP 配置。相机或安装移动、指垫更换、TCP 改动使相关标定失效。硬件时间戳为零的历史观测不按硬件同步数据使用。

最后归档状态为仍夹住物体、未抬升，之后执行了会话 shutdown。此处仅为历史，不推断现在仍持物或已经空爪；启动新任务先读当前状态。

证据保存在 evidence/；sources.json 给出原路径和 SHA-256。快照内 publication、current_condition 等字段保持原始内容，当前资料绑定以本目录 environment.json 为准。
