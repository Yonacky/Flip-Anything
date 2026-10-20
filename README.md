# Flip Anything 工作环境

DS 从 [DS_START.md](DS_START.md) 开始，运行 `context.ps1 context --project flip_anything`。只读入口提供当前项目、版本化标定和经验卡片，不连接硬件。

projects/ 管理各项目目标与指导；environments/ 管理可共享的版本化设备与场地资料；knowledge/ 按项目整理有来源的成功经验及失败边界。结构与新项目模板见 [projects/README.md](projects/README.md)。

当前实验环境为 xarm6_lab/20261020-task-ready，包含今天的操作抓取参考点与重复接触证据，以及原手眼、桌面和工具模型。当前安装已按 supervised_rigid_flip 所需范围验收并发布，可直接复用；无需先做额外抬升或旋转标定验证。

运行实现保留在 flip_pipeline/ 与 ds_control/，动作通过 pipeline.ps1 授权后由 robot.ps1 单会话执行。操作见 docs/PIPELINE_OPERATIONS.md。agent.ps1 所需后端文件缺失，当前使用 pipeline，不将旧 ORA 发布记录当成可用性证明。

临时材料放 workspace/projects/<project>/<task>/；原始证据保留 logs 路径。旧 skill_harness/ 和 skills/registry/ 保留原状态，历史案例不自动成为已发布动作技能。

维护主目录 G:\Flip Anything；DS 执行副本 G:\Flip Anything DS。硬件 Python 为 .venv，计算 Python 为 .venv-pipeline。共享设备不可并行控制。资料版本由 context-release.json 校验，运行文件由各工作区 source-release.json 校验；两者都不代表设备标定认证。

从 GitHub 安装、离线检查和版本发布见 [仓库维护说明](docs/REPOSITORY_RELEASE.md)。统一离线验收命令为 `python scripts/check_repository.py`。
