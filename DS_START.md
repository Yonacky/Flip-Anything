# DS 开始工作

先在当前工作区根运行：

```powershell
.\context.ps1 list
.\context.ps1 check --project flip_anything
.\context.ps1 context --project flip_anything
```

这些命令只读本地文件，不连接设备。显式选择本轮项目；随后只读返回的 read_next 和当前阶段接口文档。context 提供标定实际数值、组件验证状态和项目经验。check 的 ok 只表示资料完整、引用匹配，不表示物理标定通过或动作获批。

项目：projects/flip_anything/BRIEF.md。环境：environments/xarm6_lab/20261020-task-ready/。经验：knowledge/flip_anything/。不同项目从 projects/README.md 进入，禁止将上一项目的物体状态当成本项目事实。

设备和动作入口仍为 pipeline.ps1 / robot.ps1，遵守 AGENTS.md 与 SKILL.md。context 是只读资料入口，不替代动作授权；硬件使用 .venv，离线计算使用 .venv-pipeline。当前 agent.ps1 后端缺失，按现存 pipeline 执行，不自行补写硬件驱动。

当前任务的安装标定已发布，可直接复用。新任务用 pipeline.ps1 reuse-calibration <task> 进入 model。抬升、持物旋转和重复包络调查不作为标定前置要求；开始时核对当前安装并观察新物体即可。

默认 context 只展示关键标定变换、身份和经验摘要。完整开口采样、每视角误差、工具包络按 full_data_path/components 路径读取；确需完整 JSON 时加 --full。
