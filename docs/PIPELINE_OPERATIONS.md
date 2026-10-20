> 标定实验完整入口：[CALIBRATION_EXPERIMENT.md](CALIBRATION_EXPERIMENT.md)。该实验在 unverified 时继续准备、采集与计算，使用获批采集通道；正式翻转仍需要有效标定。

# 工作区入口

普通翻转先读 [任务契约](TASK_DIRECTED_EXPERIMENT.md)。begin 默认 supervised_rigid_flip；研究模式显式加 --profile measured_properties。用 brief <task> 查看当前问题，用 report-template <阶段> <输出> --task <task> 生成匹配模式的模板。以下完整物性文件要求仅针对 measured_properties；默认模式的 properties 使用范围声明和当前观察，material/mass 为 not_required。

命令在当前工作区根目录执行，使用 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\pipeline.ps1 ...`。以下缩写为 `pipeline.ps1`。所有路径均须落在本工作区，程序不会自动操作机械臂。

| 命令 | 用途 |
|---|---|
| `pipeline.ps1 doctor` | 检查计算依赖，绝不连接硬件 |
| `pipeline.ps1 calibration-task --installation <安装ID>` | 建立专门采集任务和缺项说明 |
| `pipeline.ps1 begin --object <新物体ID> --installation <安装ID>` | 建当前物体阶段记录 |
| `pipeline.ps1 status <task路径>` | 看当前阶段、缺项、证据完整性 |
| `pipeline.ps1 compute <方法> <输入JSON> <输出JSON>` | 调用固定数值算法，见对应接口文档 |
| `pipeline.ps1 report-template <阶段> <输出JSON>` | 生成未通过的报告草稿；不是执行授权 |
| `pipeline.ps1 submit <task> <阶段> <报告> --evidence <文件...>` | 校验并归档阶段证据 |
| `pipeline.ps1 authorize <task> --run <runID> --segment <段ID> --request <请求> --from-result <回执> --output <带凭据请求>` | 对已完成规划的下一段生成短时单次凭据，不发送动作 |
| `robot.ps1 execute <带凭据请求> --from-result <同一回执>` | 通过原单会话执行器发动作 |
| `pipeline.ps1 halt <task> <原因>` | 停止流程并记录待处理教师请求；物理停止仍用 robot.ps1 stop |
| `pipeline.ps1 finalize <task> --outcome <success/failure/incomplete/unknown> --summary <摘要> --evidence <文件...>` | 完成或失败均能封存材料 |

`task` 是 begin 返回的目录，状态报告的证据路径相对工作区。`submit` 的六个阶段名为 calibration、model、properties、plan、execution、review。阶段所需文件示例见接口文档，不把真实数值填在技能提示词里。

observe/status/gripper-check/stop/shutdown 仍沿用 ds_control/OPERATIONS.md。start 是长期持有连接的进程，后台启动后等待 ready，不等待它退出。关机 shutdown 不松开持物夹爪。外部停止后不自动恢复。activate-empty 仅空夹爪初始化，新版也检查实际夹爪反馈。

model报告除 checks 外需 model_request_path/model_result_path；measured_properties 模式的 properties报告需 property_observations_path/property_result_path、mass_method、mass_request_path/mass_result_path。所有被引用文件及底层原始证据都纳入 --evidence。算法结果不允许手工改；修正输入后重新计算。采集源、物体和标定身份仍需人工/教师检查。

plan报告包含有序 segments，每项为 id、role、request；request 是固定控制器契约，move 明示 base、mm_deg、extrinsic_xyz_rpy、motion_space 和 speed_units。几何计算使用米制矩阵，必须经 controller_boundary 转换。plan 的全路径证据当前需要教师复核，系统不会凭字符串自行推断真实障碍物。不要将旧回执目标粘贴进新方案。

释放时额外给 authorize 的 `--support-evidence <引用JSON>`；引用含 path、sha256、run_id、method，原始支撑记录含 supported、object_id、run_id、sample_monotonic_ns，并绑定当前回执与最近动作。method 只能是 observed_contact 或 calibrated_contact_measurement。它必须有对应图像/测量证据；填写一个 supported=true 不是测量。

结束摘要包含 text 字符串、interventions/failures/unresolved 数组。原 experiment.ps1 材料入口继续可用；新状态记录不会替代原始硬件日志。

标定/多视角/物性接触采集采用教师预先复核的 `authorize-acquisition` 通道，完整命令和 schema 见 [采集说明](ACQUISITION.md)。采集不自动完成任何阶段。Astra 请求当前写入 teacher_request.json，需要已配置适配器或人工转交；文件存在不表示已发送。

plan报告另需 geometry_request_path/geometry_result_path/path_review_path，路径复核绑定segments与几何结果哈希、task/object/installation，以及机械臂连杆、工具、持物、障碍物和实际插补。该教师记录不是让学生自行填“通过”的表格。

compute properties 的输入是观察列表，与 property_observations_path 完全相同。其它 compute 输入以对应接口为准。控制器请求可用 compute controller-request（输入含base_from_tool、grip_required、motion_space、speed），不自行拼米/毫米换算。

支撑原记录还需 reference_receipt_sha256（当前回执规范JSON哈希）、completed_action_count（当前方案已完成段数）、raw_evidence（原始图像/测量的path、文件sha256）。观察后若又有动作，需要重新观察支撑。

完成报告另需 run_id、final_receipt_path、completion_observations_path。程序核对动作完整执行、支撑释放和退出后的独立稳定，格式见 [完成判据](completion_api.md)。正常释放请求使用 grip_required=false（不在开爪过程中强求夹持码），支撑检查仍强制执行。

`pipeline.ps1 request-teacher <task> --teacher astra --missing <缺项...> --question <问题> --segment <需要帮助的片段> --evidence <文件...>` 记录信息请求，保持当前阶段，当前适配器需人工转交；请求本身不清除任何风险或阶段缺项。发生故障仍用 halt。

`pipeline.ps1 collect-run <run目录> --episode <任务ID> --calibration <标定ID> --output <材料JSON>` 将已结束的原日志整理成经验事件；教师动作需另给 --annotations 逐事件署名与证据，不能给整段日志改演员。详见 [经验接口](experience_api.md)。这只整理事实，正式技能仍由维护者审阅发布。
