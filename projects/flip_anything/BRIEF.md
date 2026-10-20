# DS 当前项目：Flip Anything

在工作区根运行 `./context.ps1 context --project flip_anything`。先读其 environment 状态、calibration 数值和两张 experiences；只按需要展开证据。已有测量优先复用，仅在安装不匹配、发生改变或当前动作需要未验证能力时补充相关检查。

任务目标：当前物体翻转，获得桌面支撑，松爪退开后独立稳定。普通空刚性容器用 supervised_rigid_flip；原始范围声明与当前观察一致时，精确质量和重心不是必测项。

先读本轮实时设备状态和当前图像，匹配 environment 身份及控制器 TCP。包中的历史罐坐标、旧回执、持物状态不用于当前控制。桌面和工具模型用于当前候选几何检查，仍需观察新物体、障碍物和支撑。

当前安装标定已就绪，软件抓取参考点与现有几何可以直接复用。创建 supervised_rigid_flip 任务后运行 `pipeline.ps1 reuse-calibration <task>`，进入 model 阶段。抬升、持物旋转和重复工具包络调查不属于标定前置任务，不为这些项目停下实验。只在安装改变、现有数据与当前观察冲突或当前动作确有参数缺口时处理相关项。

执行查 docs/PIPELINE_OPERATIONS.md，标定采集查 docs/ACQUISITION.md。新任务由 pipeline begin / calibration-task 创建；原任务路径和日志继续使用。通过授权的请求才交 robot.ps1 执行，每步读取新回执与图像；异常停止并报告，不重发未知结果。完成判据见 docs/completion_api.md。

当前 agent.ps1 所需脚本缺失。使用现存 pipeline；context 会报告两个入口是否完整，不凭 README 中的旧描述假定 agent 可用。

临时分析输出放 workspace/projects/flip_anything/<task_id>/。原始硬件数据留在执行器 logs，任务封存保留 task、run、project_id、environment_version 及 context-release.json 的哈希。DS 只准备材料，维护者负责环境版本、经验和发布。

输出保持简短：当前事实、下一步及理由、结果证据。只追查影响当前步骤的信息。成功与失败、人工照片/复位/复核都记入结果；提出方案不记成执行，接触成功不记成负载验证。
