> 2026-10-19：当前执行架构和阶段判据见项目 docs/PIPELINE_ARCHITECTURE.md 与 docs/PIPELINE_OPERATIONS.md。下文保留原材料、分层及历史接口说明；动作执行以新版单次凭据入口为准。

# 工作区与分工

`G:\Flip Anything` 是 Codex 维护项目。`G:\Flip Anything DS` 是 DS 实验工作区。原项目的 ds_control 保留为可读历史案例的对照环境，两边不能同时控制硬件。

DS 使用现有工具完成实验，只准备维护所需的材料。新入口是 SKILL.md，任务材料由 experiment.ps1 收集，原始数据由 robot.ps1 的执行器保存。DS 不提取候选、不维护经验状态、不修改技能库。

Codex 读取学生 logs/tasks/*/materials.json、摘要和完整原始 run，审查证据，提取或修订经验，维护分层技能和设备资料，再发布带版本的工具与说明。具体规则见 EXPERIMENT_RECORDING_AND_TRANSFER.md 和 DS_MAINTENANCE_ROLES.md。

旧 episode/candidate/export 文件保留供历史审查，不是新任务的必经流程。新发布不会改写过去任务使用的版本。设备几何未知就保留未知，不把历史坐标作为当前物体参数。
