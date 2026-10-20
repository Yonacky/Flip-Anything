> 2026-10-19：当前执行架构和阶段判据见项目 docs/PIPELINE_ARCHITECTURE.md 与 docs/PIPELINE_OPERATIONS.md。下文保留原材料、分层及历史接口说明；动作执行以新版单次凭据入口为准。

# DS 留材料，Codex 维护技能

DS 负责做实验、保存实际发生的事情。Codex 读取这些材料，对照历史，判断什么可以复用、什么需要验证，再修改技能库。DS 不需要提出候选、分类经验、维护版本或生成移交包。

## 一次任务怎样记录

开始用 `experiment.ps1 begin`。它建立 logs/tasks/ 下的任务目录，保存当时的设备资料和软件版本，并生成 summary-draft.json。此时还没有启动硬件，也没有继承旧目标。新物体新建任务；同一物体故障重启不新建任务。

操作统一从根 robot.ps1 进入。执行器保存本轮 RGB-D、状态、动作和回执；根入口自动把新 run 关联到当前任务，材料收集失败只给提示，不重新发送动作。也可单独执行 `experiment.ps1 collect` 补关联。不要为观察、补测而反复 shutdown/start。

重要转折、失败原因的猜测、实际恢复、人工帮助，用 `experiment.ps1 note "说明" --artifact <文件绝对路径>` 记录。artifact 可重复传入。只写可审阅的简短依据，不写长篇内部推演。分析脚本与派生图保留在 workspace/，通过 note 或摘要中的 artifacts 引用；原始文件不覆盖。

## 收尾摘要

修改本任务生成的 summary-draft.json。physical_result 和 autonomy_result 各填 success、failure、incomplete、unknown 或 stopped。有人重摆物体、判断接触或提供临时纠正，就记入 human_interventions，不能同时声称自主成功。描述物体、失败、恢复和未解决问题即可，不要求写可复用规则。

evidence.support、release、stability 分别列出相对于工作区根目录的证据文件路径；缺证据留空，不能为了通过检查填真。成功需要三项证据，但文件存在不等于内容正确，最终由维护者审阅。分析脚本、测量结果和其他相关材料列入 artifacts。估计值需要单位、坐标系和对应观察；解释不是实测。

正常结束硬件会话后运行 `experiment.ps1 finish --summary <摘要绝对路径>`。程序自动关联完整原始 run、保存版本、计算耗时与哈希，生成 materials.json。这个文件是给维护者的材料索引，不是技能，也不是自动成功认证。不只收最终成功帧，失败过程同样保留。

用户中止时优先处理停止，保持夹爪，不为收齐材料继续动作；已有任务和原始日志会保留。之后确认会话已经结束再补摘要。如果进程异常使 run 状态仍是 running，保留现状并注明问题，交维护者处理，不伪造 completed。

## Codex 怎样处理材料

Codex 核查原始回执和图像，区分事实、解释及未验证猜测；与历史案例对照，决定是否修改设备资料、操作工具、具体案例或通用做法。被反证推翻的经验由 Codex 撤回或修订。DS 只负责把反证留下，不负责修改经验状态。

历史 logs/candidates/、episodes/ 和 transfers/ 原样保留，不能把其中未验证的候选当作当前技能。旧 exporter 是维护工具，不再是 DS 完成实验的必经步骤。
