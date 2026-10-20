# 教师示范与学生材料 API

本模块只处理已收集材料，不连接机械臂，不调用 Astra，不自动修改技能库。它输出给维护者的草稿。所有函数输入输出均为 JSON 可表示的字典/列表；证据文件由调用者以 `dict[str, bytes]` 提供。数据和源事件采用 SHA-256 绑定。哈希能发现修改，不能证明内容真实。

## 分工

学生只负责实验及材料准备。教师可以提供建议，也可以实际操作，但两者必须分别记录。维护者核对材料、对照失败记录、决定适用范围，再审阅和发布。任何身份的示范都可能失败。学生运行入口不应暴露发布接口或维护者配置的写权限。

## 事件

`validate_event(event)` 返回深拷贝，错误抛 `ValueError`。`validate_trace(events)` 额外检查单 episode、时间顺序、唯一 event_id。失败和重试仍属于同一 episode 时用不同 attempt_id；不同 episode 在上层材料目录关联，不强行拼成一个时间序列。

完整动作事件示例：

```json
{
  "schema_version": 1,
  "event_id": "event-003",
  "episode_id": "episode-001",
  "attempt_id": "attempt-001",
  "timestamp": "2026-10-19T10:00:00+08:00",
  "actor": "human",
  "source_kind": "physical_execution",
  "kind": "action",
  "observations": ["rgb/003.png"],
  "evidence_refs": ["controller/003.json"],
  "actual_pose": {
    "frame_id": "base",
    "calibration_id": "mount-01-cal-01",
    "units": "m",
    "timestamp": "2026-10-19T10:00:00+08:00",
    "source": "controller",
    "matrix": [[1,0,0,0.5],[0,1,0,0.1],[0,0,1,0.2],[0,0,0,1]]
  },
  "object_pose": {
    "frame_id": "base",
    "calibration_id": "mount-01-cal-01",
    "units": "m",
    "timestamp": "2026-10-19T10:00:00+08:00",
    "metric": true,
    "quality": "validated",
    "matrix": [[1,0,0,0.4],[0,1,0,0.1],[0,0,1,0.2],[0,0,0,1]]
  }
}
```

矩阵约定为 `T_frame_local`，即局部点左乘矩阵后进入 `frame_id`；平移统一为米。验证 SE(3) 的正交性和正行列式，拒绝 NaN、无穷、布尔数值和反射矩阵。物体位姿是可选的；没有可信米制物体位姿时仍保存动作，但不生成相对物体的可迁移路径。

`actor` 只能为 `human/astra/student`。`source_kind` 只能为 `physical_execution/proposal`。教师建议使用 `proposal`，不能带 `actual_pose`。实际动作需要控制器位姿与回执证据。实际结果 success/failure 和接触标注需要证据。

`kind` 为 `action/observation/phase/contact/result/intervention/hypothesis/public_transcript/code/tool_result`。其中 phase 有 `phase` 字符串；contact 有 `contact: free/grasped/supported/released/unknown`；result 有 `result: success/failure/incomplete/unknown`；intervention/hypothesis/public_transcript 有 `text`。其他原始公开字段可保留，但拒收 `chain_of_thought/hidden_reasoning/reasoning_content/private_reasoning` 字段。不向模型索取或绕过接口提取隐藏推理；原始公开对话、行动说明、代码、工具请求与回执照常保存。

## 证据

```python
manifest = evidence_manifest({"rgb/003.png": image_bytes, "controller/003.json": receipt_bytes})
verify_manifest(manifest, same_bytes)
```

manifest 结构为 `{"schema_version":1,"files":[{"path":"...","size":123,"sha256":"..."}],"manifest_sha256":"..."}`。路径必须是相对 POSIX 路径，无 `..`、盘符、反斜杠或大小写重名。验证包含缺失、增加、字节变化和 manifest 本身变化检查。调用者负责只读封存原始文件；函数本身不修改文件。

## 两种处理

```python
scope = {"goal":["invert"], "object_family":["rigid_container"],
         "tool_revision":["mount-01"], "contents":["confirmed_empty"]}
teacher = process_teacher(events, manifest, scope=scope)
student = process_student(events, manifest, scope=scope)
```

教师：按明确的阶段、接触状态变化分段，跨尝试重新开始状态。每段保留原事件、结果和相对物体关键点。关键点仅在位姿同参考系、同标定、米制且验证过、采样与事件时间差不超过 0.2 秒时产生；公式为 `inv(T_base_object) @ T_base_tcp`。没有这些条件时记录原因和原始动作，不猜补位姿。原始轨迹不会自动重放；新物体仍需定位、可达性和碰撞检查、抓稳和支撑判断。这里实现的是基于标注的确定性分段，不是从视频自动学习接触阶段。

学生：按 attempt 分组，保留失败之前的全部事件 ID、失败后的干预和重试；最后一次成功不会覆盖之前失败。解释单列为 `causal_hypotheses`，默认 `unverified`。有教师建议或人工介入时不标为完全自主。程序只汇总记录中的结果，不替代实机结果判定。

两种输出共用 `status:draft`、`scope`、`lineage`、`evidence`、`raw_events`。教师身份不自动提高可信等级。

## 请求教师

```python
request = teacher_request(
    request_id="help-001", teacher="astra",
    missing_information=["contents", "fragility"],
    question="无法确认罐内有无液体，请判断可用证据或示范检查方法。",
    requested_segment="确认内容物，尚不开始夹取",
    state_snapshot={"stage":"material_assessment", "holding":False},
    evidence_refs=["rgb/003.png"], timestamp="2026-10-19T10:00:00+08:00",
    adapter_connected=False)
```

无论 adapter_connected 的值是什么，这个函数只生成请求：`status:pending`、`sent:false`。未接入服务时 `next_action:manual_handoff_required`；接入后也需要实际的发送器单独确认送达，不能把生成文件写成“已叫来教师”。文本答复记录为 proposal；确实通过同一执行器操作且有测量、图像、回执的，才是 physical_execution。

## 维护者审核和检索

维护者配置示例（上层存储，学生只读或不可读）：

```json
{"schema_version":1,"authorized_maintainers":["codex-maintainer"]}
```

```python
reviewed = review_record(draft, reviewer="codex-maintainer", reviewer_role="maintainer",
    authorized_maintainers=registry["authorized_maintainers"],
    evidence_refs=["controller/003.json"], decision="approve", blobs=bytes_by_name,
    notes="核对回执和图像；仅适用于已确认空的刚性容器。")
released = release_record(reviewed, maintainer="codex-maintainer", maintainer_role="maintainer",
    authorized_maintainers=registry["authorized_maintainers"], blobs=bytes_by_name)
revoked = revoke_record(released, maintainer="codex-maintainer", maintainer_role="maintainer",
    authorized_maintainers=registry["authorized_maintainers"], reason="新证据否定此结论")
```

审核与发布分开，拒绝学生角色、未登记维护者、缺失证据和审核后变更。此名单检查是软件流程约束，不是密码学认证；不能把可写配置的学生进程视为已安全隔离。维护者必须掌握独立入口和写权限。

`retrieve(records, context, query_tokens=(), limit=5)` 只返回 released，先检查每个 scope 字段在当前 context 中明确匹配，再用描述词重合排序。scope 最少包含 goal/object_family/tool_revision，实际发布应补材料、内容物和必要几何范围等适用条件。未知条件不会自动匹配。出现同 record_id 的 revoked 记录时，旧 released 副本也不会被返回。相似度不能绕过适用范围。检索结果永远标为 `planning_reference_only`，不等于动作许可。

## 从已有实机日志准备材料

`material_adapter.import_run(root, run_path, episode_id, calibration_id, annotations=None)` 只读取文件，返回 JSON 字典，包含 events、manifest、source_files、warnings、task_outcome。不返回二进制字节，不修改原运行目录。最终审核需要字节时，维护者按 manifest 再读原始文件并校验。

```python
from flip_pipeline.material_adapter import import_run
materials = import_run(workspace_root, run_directory, episode_id, calibration_id,
                       annotations="workspace/annotations.json")
student = process_student(materials["events"], materials["manifest"], scope=scope)
```

导入器保留运行目录的全部原始文件哈希，包括没有被某个事件引用的图像、深度和其他材料。解析 events.jsonl 的 kind/utc/host_monotonic_ns/payload，同时按文件名关联 commands/xxx.json 与 result_xxx.json。实际动作需 hardware 来源、匹配的控制器 robot_sample 和成功回执；缺一项就保留为未确认请求或工具结果。失败回执可能包含部分运动，只记观察与失败范围未知，不声称整段已经执行。

会话正常退出、SDK 返回成功、抓爪命令完成，都不能直接生成物体翻转成功。只有带证据的 task_result 明确标注才会改变 task_outcome。命令失败另有 result_scope=command，保留失败前缀，但不覆盖整个物体任务的最终判断。未标注时保持 unknown。

annotations 是逐事件记录列表，推荐保存成真实 JSON 文件，以便同原始数据一起哈希。示例：

```json
[
  {
    "target": "result_实际文件名.json",
    "actor": "human",
    "signed_by": "现场操作人标识",
    "evidence_refs": ["workspace/teacher-confirmation.json"],
    "phase": "grasp",
    "contact": "grasped"
  }
]
```

target 可以是实际 result 文件名，也可以是 `events.jsonl#L12`。同一 target 最多一条标注。可选 phase、contact、intervention、hypothesis、task_result、object_pose；后者使用本文位姿结构，需要实际物理采样和测量证据。教师归属必须逐事件给 actor、signed_by 和真实证据文件，不能整次运行改成 human/astra。署名属于待审材料，不是密码学认证。原始 phase 字段、文件名和夹爪寄存器不会被猜成接触事实。

原始 JSONL 尾行截断会产生警告并保留原文件；源文件在导入中变化则拒绝，需等会话结束或使用稳定快照。缺少 hardware 来源的日志不会被归类成实体动作。手动示教若只有前后状态，这些仍是 observation，不补造中间动作。首版不自动同步或插值连续控制轨迹，也不自动识别阶段。
