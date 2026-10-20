> 2026-10-19 更新：move/grip 必须先在项目根目录通过 pipeline.ps1 authorize 或 authorize-acquisition 取得单次凭据。新版请求还需明确控制器单位和插补方式。下面保留观察、状态、停止等硬件说明；旧直接 execute 示例不再可绕过上级流程。详见 ../docs/PIPELINE_OPERATIONS.md。

# 操作接口

在本目录运行命令。PowerShell 的当前目录不可靠时先 `Set-Location -LiteralPath 'G:\Flip Anything\ds_control'`。`robot.ps1` 自行定位父目录 Python，设置模块路径，不需要 pip 安装或猜 PYTHONPATH。

```powershell
.\robot.ps1 doctor
.\robot.ps1 start
```

`doctor` 只检验导入和配置，不访问硬件。`start` 在前台长期运行，使用 DSH 的后台命令功能运行并保留任务句柄；若手动 Start-Process，使用隐藏窗口并把标准输出与错误输出分别重定向到 `../logs/sdk/ds_control/` 下本次唯一命名的文件。不要等待 start 退出才发观察命令。它连接实机、启动相机、记录初始观察，不主动移动、张爪或清除警告。等初始采样成功后再操作；启动失败先收集输出。`active.json` 给出本次 run 名，`session.lock` 防止本目录重复启动。若异常遗留锁，先证明对应 PID 已退出才能清理。

```powershell
.\robot.ps1 status
.\robot.ps1 observe
.\robot.ps1 gripper-check
.\robot.ps1 activate-empty
.\robot.ps1 execute scene\next_action.json --from-result ..\logs\experiments\ds_control\本次run\result_上一成功命令.json
.\robot.ps1 stop
.\robot.ps1 shutdown
```

`status` 返回新鲜位姿和夹爪读数；`observe` 返回状态及彩色图、深度文件路径。相机内参、深度单位与对齐信息见本次 `events.jsonl` 的 camera_started 和 observation。读取实际图像，不靠文件名推测场景。`activate-empty` 仅在已经观察确认空爪且未激活时使用，不用于持物恢复。`shutdown` 正常结束会话并停止机械臂，保持夹爪，不自动释放。`stop` 写停止标志，执行器在运动轮询中处理；它不替代硬件急停，网络或进程故障时不能保证即时响应。

`execute` 读取 JSON 文件，自动从指定的本次成功回执填入 expected_pose，生成唯一命令 ID 并原子提交。返回 result_file 和实际结果。只串行发动作，上一条没有明确结果不得发送下一动作。超时结果未知，先查对应 result_file 和后台进程，不能重发。执行失败后先诊断，禁止盲目放宽限制。

动作字段如下；尖括号是现场计算值，不是可直接运行的 JSON。用 Python json.dump 写文件，避免 PowerShell 手拼 JSON。

```text
移动：{"action":"move","pose":[<x>,<y>,<z>,<roll>,<pitch>,<yaw>],"grip_required":<持物true或空载false>,"capture":true}
夹爪：{"action":"grip","pos":<位置编码>,"speed":<速度编码>,"force":<力编码>,"grip_required":false,"capture":true}
```

pose 为基座坐标系、毫米与度。夹爪位置编码不是毫米；从张开到闭合方向编码增大，不能把完全闭合当作抓住。夹持后运动必须 grip_required=true；开合过程本来可能改变接触状态，因此 grip 命令为 false。夹爪 gACT/gSTA 是激活状态，gFLT 是故障，gOBJ 是接触/运动状态，gPO 是位置编码。它们不是桌面接触传感器。

执行器内有保守的运动速度、单步位移和旋转上限，见 `scripts/session.py`；这些是接口限制，不是目标动作参数。纯平移采用笛卡尔运动，改变姿态采用关节运动。逆解通过只证明终点可达，不证明中间路径避碰；调用者要检查完整工具与物体包络。不提供盲跑整段翻转接口。

`scene/current.json` 可能保留上一轮内容，不能把它当作本轮已验证场景。将现场估计的桌面、物体轴线/两端、相机到 TCP、指垫中心到 TCP、物体到 TCP、误差及证据路径保存在这里。控制器位姿不能单独补出相机外参。历史变换和轨迹可按 SUCCESS_CASE.md 只读访问，用于理解方法；不能把历史场景或示例当作当前标定。



本次基线模式不要求 harness 或 episode。`.ps1` 被执行策略拦截时，用 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\robot.ps1 doctor` 这样的进程级入口。不要假设本机有 `pwsh` 命令。`start` 同样使用此入口并由宿主后台运行。

CLI 标准输出形式为 `{command_id, result_file, result: {...}}`，而 `result_file` 中直接存 `{ok, state, camera, ...}`；不要对落盘文件再读取 `result.ok`。成功状态参考须同时具有 `ok=true` 与 `state`，并属于本次 active run。
