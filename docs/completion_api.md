# 完成判据与证据格式

`validate_completion(root, folder, state, data, refs)` 在执行完成报告被接受之前运行。它不相信“已完成”这一句话，也不把控制器到位当作物体倒立。它核对程序实际执行记录、支撑、开爪和退出后的物体测量。它不能独立证明图像识别器认对了物体和端部；原图与测量输入仍需保存，供教师复核。

## 执行报告入口

在既有 execution 报告字段之外增加：

```text
{
  run_id: nonempty-string,
  final_receipt_path: workspace-relative-result-json-path,
  completion_observations_path: workspace-relative-observation-json-path
}
```

`refs` 必须包含以下全部文件的字节哈希：最终回执、观测文件、用到的每张原始采集材料、每个已消耗 ticket、它们绑定的前一控制器回执、支撑观察文件及其原始证据。嵌套引用也必须列入；这样接受完成报告后，原始材料被改写或丢失会被项目统一的哈希检查发现。

动作 history 必须为当前 plan_hash 的全部 segments，顺序一致，每条 event 为 action_completed、run_id 一致，且没有 inflight。每个动作必须对应唯一 consumed ticket，票据绑定当前任务、plan、segment、run、实际请求和前一回执。回执、签发和完成时间不能倒序。没有执行动作，仅提交一份全 pass 的报告，会被拒绝。

## 支撑、释放与退出

释放不仅由 segment.role 判断：夹爪在持物状态下实际开大也被识别为释放。它必须具有发行该命令时的 support_evidence，原始材料哈希完好，completed_action_count 和 reference_receipt_sha256 对应紧邻释放前的状态。支撑观察必须发生在前一动作完成后、释放签发前，并在规定时间内使用。

开爪完成回执必须同时满足 gOBJ=3、gFLT=0，实际 gPO 与命令 pos 在允许误差内。控制器显示因接触而停止开爪，不等于完成释放。最后一次释放后还必须有完成的 move/retreat 段。

现有执行器的 `grip_required` 表示整个动作期间持续检查持物状态，因此释放命令要设为 false；释放前的承接证明与票据校验仍然必需。设为 true 会在正常张开后因为持物状态消失而触发中断。

最终回执必须是该 run 的 result 文件、ok=true，时间不早于最后动作，末端位置/姿态与最后动作结果相符，夹爪仍处于已释放状态。它不能是旧 run 的截图或动作前回执。

## 物体观测文件

准确字段如下。矩阵和数组均为实际 JSON 数值数组。

```text
{
  object_id: same-as-task,
  installation_id: same-as-task,
  run_id: same-as-report,
  calibration_artifact_sha256: task.artifacts.calibration,
  frame: "base",
  units: "m_rad",
  axis_object: directed-unit-vector-length-3,
  axis_label: nonempty-string-identifying-the-same-physical-end,
  before: observation,
  post_retreat: [observation, observation, ...]
}

observation = {
  capture_id: unique-nonempty-string,
  sample_monotonic_ns: positive-integer,
  base_from_object: rigid-matrix4x4,
  position_uncertainty_m: nonnegative-number,
  orientation_uncertainty_rad: nonnegative-number,
  method: "rgbd_model_fit" | "tracked_keypoints" | "teacher_measured",
  raw_evidence: [{path: workspace-relative-path, sha256: raw-file-byte-hash}, ...]
}
```

`axis_object` 必须有方向，例如“从封闭底端指向开口端”。只拟合出一根正负号不确定的圆柱轴，不能据此判断翻转成功。before 与 after 必须追踪相同物理端部，而不是每次拟合时随意改变轴的符号。

before 时间应早于首个动作签发；post_retreat 全部采自最后动作完成之后。至少两个不同 capture_id、不同原始材料、递增采集时间的观察，并达到规定的稳定观察间隔。复制同一帧并改时间，不是独立观察。

程序将有向轴从物体坐标转到基座坐标，比较前后轴反向程度；将两次姿态的方向不确定度相加到倒置角误差上。稳定性检查比较所有 post-retreat 观测对，将测量误差加入平移/转动变化，再与阈值比较。因此不能把误差较大的观测当成精确数值通过检查。

## 阈值

默认值写在 `completion.DEFAULT_LIMITS`，维护者可以在设备策略文件 `completion_limits` 中修改，学生不能在本轮报告里自行放宽。默认倒置角误差上界 10°、稳定平移上界 5 mm、稳定转角上界 5°、观察跨度至少 1 秒；单次位姿位置不确定度不超过 2 mm、方向不确定度不超过 0.03 rad。两次测量的不确定度会同时计入变化上界。

这些是初版测量判据，并非对任何物体都足够的物理保证。对于会缓慢倾倒、滑动或有活动内容物的物体，需要更长观察和适用的物理判据；当前刚体流程不能声称覆盖这些情况。结果状态为 `measured_completion_checked`，明确范围是数值和来源约束，而不是独立视觉识别认证。
