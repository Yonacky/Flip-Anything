当前资料入口：先读 DS_START.md，运行 context.ps1 context --project flip_anything，按项目读取环境标定和经验。agent.ps1 后端当前缺失，动作使用现存 pipeline.ps1 / robot.ps1。

# 当前工作区约定

先读 SKILL.md，按 docs/PIPELINE_OPERATIONS.md 使用入口。主仓库由维护者修改和发布；DS 副本只执行实验和准备材料。不同目录共用实机，不能并行控制，不停止别人的进程。

用户要求翻转时，已授权任务所需操作及记录，无需逐动作询问。使用正式权限机制。动作通过 pipeline 授权后的 robot.ps1 执行；不绕过阶段判据、驱动或设备互斥。标定/观测/接触采集使用教师复核的采集片段，不把采集完成当阶段完成。实验范围按任务配置及原始操作员声明确认；只处理与当前动作有关的未知项。观察与范围矛盾时转交教师；发生错误后停止并叫人，不自动恢复。

硬件环境 .venv 与计算环境 .venv-pipeline 分开。临时输入输出放 workspace，原始数据放 logs。技能、发布目录、配置和代码由维护者管理；DS不发布经验。保留失败、人工干预和未知结果。配置值须来自实际测量，不能为了检查通过填写假值。

标定实验按 docs/CALIBRATION_EXPERIMENT.md 推进，未验证不是自动结束理由。DS 可在 workspace/calibration 下编写仅处理已有数据的检测、转换和误差分析脚本；新增实时设备接口由维护者接入同一会话。采集运动经教师复核并使用 authorize-acquisition；DS 准备 draft，维护者审阅发布。不得套用翻转任务的未标定退出规则。
