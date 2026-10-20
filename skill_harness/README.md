> 2026-10-19：当前执行架构和阶段判据见项目 docs/PIPELINE_ARCHITECTURE.md 与 docs/PIPELINE_OPERATIONS.md。下文保留原材料、分层及历史接口说明；动作执行以新版单次凭据入口为准。

# 新架构使用入口

本目录实现目标与验收、任务方案、可组合能力三层，复用 `ds_control` 硬件执行器。先读 [分层约定](LAYERS.md)。从 `G:\Flip Anything` 项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m skill_harness catalog
.\.venv\Scripts\python.exe -m unittest discover -s skill_harness\tests -v
```

上面两条不连接硬件。`catalog` 显示三层目录，`catalog --layer capabilities` 仅显示能力、前置条件和验证状态。当前所有技能为候选，没有自动翻瓶启动命令。

## 下一次实验

先按 `ds_control/AGENTS.md`、`OPERATIONS.md` 启动一个新的硬件会话。不要抢占或停止当前会话。使用其 active.json 中的真实 run ID 创建 episode：

```text
python -m skill_harness init logs/episodes/新的实验名 --run-id 实际运行ID --plan empty_can_inversion
python -m skill_harness run logs/episodes/新的实验名 observe --input 观察请求.json --hardware --allow-candidate
```

init 会保存目标、方案、能力和别名快照，以及未观测的 scene.json；设备资料通过可选的 `--device-profile` 单独传入。当前物体和桌面信息来自本轮观察。
观察请求文件内容为 `{"payload":{"action":"observe"},"facts":{}}`。所有新架构实机调度都需要 `--hardware`；候选另需 `--allow-candidate`。这两个开关是明确选择，不替代用户授权或宿主权限。不要在未授权的调试中使用。

成功调度后返回 revision 和 last_result。查看该回执所指的 RGB-D 图像，再生成后置证据文件，调用：

```text
python -m skill_harness assess logs/episodes/新的实验名 --facts 后置证据.json
```

证据结构如下（尖括号需要实际值，不是有效示例数值）：

```text
{
  "scene_visible": {
    "value": true,
    "revision": <本episode当前revision>,
    "result_sha256": "<对应回执文件的SHA256>",
    "source": "perception",
    "method": "<实际采用的识别方法及检查过程>",
    "artifacts": ["<原始彩色图绝对路径>", "<分析结果绝对路径>"]
  }
}
```

source 为 perception、controller 或 human。人工证据必须使用 `--supervised`，会累计到结果；不要通过更改标签隐去人工介入。SHA256 可用 Python hashlib.sha256(Path(...).read_bytes()).hexdigest() 取得。所有物理事实都需要当前观测依据，不能单纯从上一阶段的名称复制 true。

下一技能的请求为 `payload`、`reference`、`facts` 三项。reference 必须是当前 episode 最新成功回执。payload 遵循既有 move/grip 接口，facts 满足 catalog 所列 requires，revision 使用当前值。本轮动作后 revision 增加，旧证据失效。旋转与运动的限制仍由现有执行器实施，运行器不计算 IK。

```text
python -m skill_harness retrieve 前置证据.json --include-candidates
python -m skill_harness run logs/episodes/新的实验名 选定技能 --input 请求.json --hardware --allow-candidate
```

retrieve 只辅助检索，不证明证据有效；run 才执行时效与回执检查。没有满足条件的技能时，只解决缺失的具体观测，不编造完整场景模型。失败后需要检查并取得新的观察，再 `reconcile --result 新观察回执 --reason 恢复依据`；未知执行结果必须先查清，不能用新观察掩盖旧动作仍在执行。

## 教师提炼经验

教师读取 episode 的 events.jsonl 与关联的真实观测，将单个可迁移纠正写成 JSON：

```json
{
  "trigger": "何种观测与状态下需要此经验",
  "failure": "原决策的错误及实际证据",
  "correction": "最小的动作选择或判断改动，不含历史目标坐标",
  "verification": "如何用独立案例证实改善及发现反例",
  "scope": "适用设备、物体和观测条件，哪些情形不适用"
}
```

```text
python -m skill_harness propose logs/episodes/来源实验 --lesson 纠正.json
python -m skill_harness promote 候选经验文件 --validation logs/episodes/独立验证实验
```

propose 保留失败和人工介入，不直接写入学生常用经验。promote 检查独立运行、完整成功记录与无人工辅助；它批准的是经验，不是硬件代码。不能伪造验证 episode 来让条目通过。当前没有自动模型调用，也没有已批准经验。这只是早期记录机制；独立运行成功不足以证明经验的因果作用和适用范围，晋升制度将在下一部分讨论，本轮不批准新经验。

Mock 回合单独标记 backend，不能用作真实机器人经验的批准证据。
