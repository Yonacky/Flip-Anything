# 按任务需求推进的翻转实验

当前默认配置是 supervised_rigid_flip。它保留标定、几何、路径、支撑释放和结果检查，将 properties 阶段改为策略适用性检查。measured_properties 保留旧的完整物性测量路线。已有任务沿用原模式，不自动迁移；新模式必须新建任务。

## 开始前与命令

在本工作区根目录调用 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\pipeline.ps1 ...`。下面省略前缀。

```text
begin --object <本轮新ID> --installation <devices/installation.json里的ID> --profile supervised_rigid_flip
brief <返回的task目录>
report-template properties workspace/properties.json --task <task目录>
report-template plan workspace/plan.json --task <task目录>
```

brief 是简短决策入口，status 提供完整证据状态。报告模板必须传 --task 才会匹配新模式；模板保留 unknown/unverified，填写并提交实际证据后才能推进。当前安装档案尚未 verified，这是实机实验的阻塞项；完成专门标定后由维护者发布，不由 DS 用旧成功记录补值。路径后端仍是教师复核，不把简化物性门槛误称为无人辅助全流程。

## 题干来自哪里

目标：物体倒置、桌面支撑、松爪退开后稳定。安装参数来自 verified 档案，物体几何来自本轮观察，策略范围来自实验者的原始声明。维护者发布 devices/task_profiles/supervised_rigid_flip.json；DS 不修改它。

properties 报告引用 scope_declaration_path。声明结构如下，所有内容须有实际原始声明支持；此示例不是授权或现场事实：

```json
{
  "profile": "supervised_rigid_flip",
  "source": "operator_declaration",
  "approved_by": "human",
  "installation_id": "INSTALLATION_ID",
  "object_ids": ["OBJECT_ID"],
  "operator_supervised": true,
  "conditions": {
    "empty": true, "rigid": true, "nonfragile": true,
    "cool_and_nonsharp": true, "within_equipment_handling_limits": true
  },
  "evidence": ["workspace/operator_statement.txt"]
}
```

可一次声明列出的整批物体，不需要逐动作询问。只说“有急停”不代表已确认以上物体条件。没有范围声明时，用一个问题确认是否将当前对象纳入此范围，不展开逐项研究。声明只说明实验适用范围，不证明材料、质量或重心。approved_by 是可审阅的署名约定，不是密码学身份验证；原始对话必须保留。

报告 source=physical_evidence，唯一检查 scope_membership 的 result=pass、method=operator_scope_and_observation、detail 写明范围如何对应当前对象，evidence 引用声明和观察。另含 scope_declaration_path、scope_contradictions（确认无矛盾后填空数组）、observation_evidence（本物体当前照片/观测文件）、mass_status=not_required、material_status=not_required。所有声明、原话和观察均放入 submit --evidence；不能把模板或自行写的通过结论当原始证据。

## 决策与阶段

| 阶段 | 需要的答案 | 不必做的事 |
|---|---|---|
| calibration | 当前安装是否有有效配置 | 每换物体重新标定 |
| model | 抓取与支撑几何是否足够 | 为本轮无关属性补全整个世界模型 |
| properties | 是否适用该策略且没有观察矛盾 | 材料牌号、密度、完整重心辨识 |
| plan | 抓取、翻转、落放和实际路径是否成立 | 从头实现 IK、把仿真假设当实测 |
| execution | 上一步是否完成，下一步条件是否成立 | 把调用成功当任务成功 |
| review | 真实结果及材料是否完整 | 学生直接发布技能 |

plan 必须声明 strategy=geometry_guided_pick_invert_place、property_requirements=[]。其几何算法本身不使用质量/重心；不能附带依赖未知质量、摩擦或重心的稳定性计算。若这种策略不适用，保留原因并切换到专门研究任务，不能把未知参数填成默认值。最终稳定性仍以释放、退开后的独立观察验收，事前几何计算不等于实物稳定性保证。

每次追加测量要说明“它改变哪个动作参数或通过条件”。没有这样的关联，不开展测量。无新证据时不重复同一请求；异常停止，基础设施缺项记录后交维护者，不机械反向回退。

## 本轮观察什么

保存模型公开回复、工具记录、原始观测、SDK 回执和人工干预。结束摘要记录任务配置版本、起止时间、对象 ID、真实完成情况、补测及用途、重复查询、教师请求、人工介入和停在哪个阶段。区分基础设施阻塞、模型决策失败和执行失败。准备阶段被未完成标定挡住，不算自主翻转失败或成功。

比较新旧系统时保持模型版本、对象条件、安装和可用工具一致，关注完成率、用时、无效补测和人工介入。当前只有离线验证，不能用测试通过代替实机表现。
