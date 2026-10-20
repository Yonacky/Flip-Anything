# 刚体落放仿真接口

`simulate_placement(spec)` 使用真实 MuJoCo 无渲染运行，支持 box、cylinder。它检查给定质量、惯量、摩擦下的落放，不从图片或仿真输出反推真实重心，不验证机械臂轨迹，也不模拟内容物晃动或布料。

需要显式传入 `units="m_kg_s"` 和 `world_frame="table_z0"`。`base_from_object` 是物体相对基座的实测或候选姿态；`table_from_base` 将基座坐标转入实测桌面坐标，桌面为 z=0 且 +z 指向桌面上方。程序使用两者相乘的结果建模，禁止直接塞入来源不明的 `world_from_object`。

`gravity_base_m_s2` 明确给出重力在基座坐标系的方向和大小，程序也将它转换到桌面坐标。桌面倾斜时重力不会错误地被设成垂直桌面。

其他必需字段：

| 字段 | 含义 |
|---|---|
| kind | `box` 或 `cylinder` |
| contents | `empty` 或 `known_rigid`；未知、液体等拒绝 |
| mass_source | `measured` 或 `hypothesis` |
| mass_kg | 正的质量 |
| com_m | 物体坐标系下的质心，必须位于模型凸包内 |
| size_m | box 为三个半尺寸；cylinder 为半径、半高 |
| inertia_kg_m2 | 关于质心的三个主惯量，或完整 3×3 惯量张量 |
| inertia_frame | 三元素时为 `object_com_principal_axes`；矩阵时为 `object_com_axes` |
| friction | MuJoCo 的滑动、扭转、滚动三个摩擦参数；其物理含义和单位不同，不能当成一个统一的无量纲系数 |
| evidence | 非空证据引用列表；假设必须写明是假设 |

`object_com_principal_axes` 表示主惯性轴与物体轴对齐，不能随便取完整矩阵的对角线代替；不对齐时传完整矩阵，程序写入 MuJoCo `fullinertia`。惯量必须对称、正定并符合刚体主惯量的三角不等式。

可选 `duration_s` 默认 2 秒、`timestep_s` 默认 0.002 秒；必须在代码允许范围内。初始模型穿透桌面超过 `max_initial_penetration_m` 会拒绝，默认容差 0.0001 米，最大允许 0.001 米，不能靠大穿透让求解器“推回去”后宣称稳定。旋转后的 box/cylinder 最低点通过其支持函数精确计算，不只比较中心高度。

仿真中检查位置、速度、加速度有限性和 MuJoCo 警告计数。发生数值警告或自动重置迹象时抛出错误，不继续输出看似正常的稳定状态。输出分别保存 `final_linear_speed_m_s` 和 `final_angular_speed_rad_s`，不再混合长度和角度取一个范数。

输出包括使用的 XML、全部输入、物体在桌面及基座坐标下的最终矩阵、位移、角度变化、接触数和速度。状态始终是 `conditional_simulation`，没有 `physical_success`。桌面当前是无限刚性平面；有限桌沿、周边障碍、机械臂、夹爪、接触测量、滑动内容物均不在这个仿真范围内。
