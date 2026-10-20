> 标定实验完整入口：[CALIBRATION_EXPERIMENT.md](CALIBRATION_EXPERIMENT.md)。该实验在 unverified 时继续准备、采集与计算，使用获批采集通道；正式翻转仍需要有效标定。

# 标定、补视角和接触测量怎样采集

采集动作与正式翻转分开。还没有物体模型时，可以在教师复核后移动到几个观察位置；还没有物性结果时，可以做教师批准的接触测量。采集许可只覆盖这些指定动作，不会自动通过标定、模型或物性阶段，也不允许夹着物体直接开始翻转。

## 先确定谁在操作

有两种方式。教师可以直接使用上位机或实体示教器操作，DS 只负责采集前后状态和图像；也可以由教师审核一小段采集动作，再通过项目执行器逐段执行。前一种是人工示教，后一种是教师审核的采集程序。不能把 DS 生成的计划标成“教师已经示范”。

教师建议如果只是文字，保存为 proposal。实际动作必须有控制器采样和动作回执；每个被归为 human/astra 的事件还需要具体署名记录，不能因为教师出现过一次就把整次学生实验改成教师示范。

## 三种用途

| purpose | 能做什么 | 开始前要核实什么 |
|---|---|---|
| calibration_views | 在固定标定板前取得多姿态图像与机器人状态 | 安装身份、标定板固定、指定视角路径经过教师复核，空夹爪 |
| object_views | 为当前物体补观察视角 | 物体身份、相机与环境几何、指定路径经过教师复核，空夹爪 |
| material_probe | 为明确问题做一次接触或称重测量 | 本次接触的风险已排除，测量装置和读数来源明确，起始工具状态已核实 |

这里的接触风险排除仅适用于批准的探针，例如外部轻触检查；它不会顺带证明物体适合翻转。内容物、易碎性或移动内部物体等问题仍由物性阶段分别处理。没有足够证据排除本次接触风险，就不签发探针动作。

## 采集计划结构

以下为计划字段示意，实用文件必须替换占位值。`segments[].request` 使用现有 controller_boundary 契约，既不猜单位，也不猜欧拉角顺序。先由当前几何和教师观察得到实际目标，再计算请求。

```json
{
  "purpose": "object_views",
  "task_id": "TASK_example",
  "object_id": "OBJECT_example",
  "installation_id": "INSTALLATION_example",
  "reviewer_id": "registered-teacher-id",
  "reviewer_role": "human",
  "reviewed": true,
  "initial_result": {
    "path": "logs/experiments/ds_control/RUN_example/result_initial.json",
    "sha256": "SHA256_OF_INITIAL_RECEIPT"
  },
  "segments": [
    {"id": "view_01", "role": "observe_view", "request": {"action": "move"}}
  ],
  "evidence": [
    {"path": "workspace/teacher_path_review.json", "sha256": "SHA256_OF_REVIEW"},
    {"path": "logs/experiments/ds_control/RUN_example/result_initial.json", "sha256": "SHA256_OF_INITIAL_RECEIPT"}
  ],
  "checks": {
    "path_reviewed": true,
    "contact_risks_excluded": true,
    "empty_tool": true
  },
  "reviewed_plan_sha256": "SHA256_OF_ALL_OTHER_FIELDS"
}
```

这个例子特意没有编造实际目标位姿，因此不能直接执行。完整 move request 要有目标 pose、frame=base、units=mm_deg、orientation_convention=extrinsic_xyz_rpy、grip_required、motion_space、speed 与 speed_units。夹爪请求使用 Robotiq 寄存器值，不使用毫米。

`reviewed_plan_sha256` 对除自身以外的整个计划做项目标准 JSON SHA-256，包含审核身份、检查、证据、初始回执和所有实际请求。改一个目标或起点之后必须重新审核，不能保留旧签名。软件中的审核者名单与哈希是流程约束，不是对真实教师身份的密码学证明；审核文件由维护者管理。

起点必须绑定复核时的真实回执。发出第一段前重新读状态，确认与复核起点一致；后续段依次绑定上一段实际结果。不能拿很早的初始位姿给一条现在才执行的路径作保证。这里的检查仍不是自动全臂碰撞求解器，教师要明确审核实际插补方式和沿途障碍。

## 执行与退出

父流程提供 `authorize-acquisition` 为复核过的下一段生成单次凭据；实际命令仍交给 `robot.ps1 execute` 和同一 session 监督。参数以 `pipeline.ps1 authorize-acquisition --help` 的当前接口为准。凭据不等于动作已经执行，已有回执前不要重复签发同一步。

每段完成后保存实际状态、RGB-D 和结果文件。采集计划完成只说明该组采集结束；之后调用数值工具求解，经过验证再提交对应阶段。已完成的采集动作和正式翻转动作分别计数，换计划后不能从头重放已经执行的抓取或移动。

遇到执行错误、停机或结果未知，停止继续采集，保留夹爪状态，留下待处理教师请求。这里不包含自动纠错、逆向回放或松手恢复。

## 人工示教的最小记录

人工操作前保存一次 status 与 observe，明确本次是谁、要示范哪个片段和当前物体身份。人工完成一个有意义的片段后，再保存 status 与 observe；如果需要研究完整运动轨迹，就必须同步记录控制器连续采样，不能从两个端点补造中间路径。

另存 annotations，明确每个教师事件对应哪个原始回执或采样、操作人、阶段、接触变化和结果。接触标记必须对应能核查的图像或测量。没有实际动作回执的端点状态只算观察，不能被自动导入器升级为“已执行动作”。结果未知如实保留；session 正常退出不等于物体翻转成功。

原始运行日志由材料导入器归一化，人工补充标注与原始文件同时哈希保存。只有在当前标定、米制物体定位、时间同步都成立时，维护者才提取物体相对关键点；否则先保存原始示教材料。
