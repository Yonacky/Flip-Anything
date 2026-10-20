# 当前标定：已就绪

当前安装标定已发布，可直接用于 supervised_rigid_flip 的几何引导实验。依据是现有手眼与深度测量、操作员确认的工具尺寸、桌面参考和今天的重复接触标定。无需重做标定，也无需先做抬升、持物旋转或重复包络调查来获得标定资格。

读取 context 的 calibration，按当前物体观察计算动作参数。软件抓取参考点仍使用今日测量；控制器 TCP 保持 174 mm。不要重复补偿，不把历史物体坐标当当前坐标。完整开口模型、包络和测量证据按 components 路径读取。

发布状态 verified 表示当前任务所需标定已经验收；验收依据明确记录为 operator_accepted_measured_geometry，不声称未做过的独立测试通过。旧 evidence 内的 draft 和未测试字段是当时记录，当前有效状态以本版本与 devices/calibration_release.json 为准。

新任务调用 pipeline.ps1 reuse-calibration <task>，应直接进入 model 阶段。开始任务只核对安装仍一致、读取当前物体与相关障碍物。发生安装变化时处理变化对应的标定，不从无关实验扩展缺项。
