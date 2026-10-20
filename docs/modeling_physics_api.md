# 建模与简单质量分布测量

这里提供计算和校验函数，不控制机械臂。所有长度以米计，力以 N 计，力矩以 N*m 计；变换是列向量约定的 `T_目标坐标系_来源坐标系`。每份结果都保留证据引用。证据文件的存在、哈希、任务身份和版本由主流程封装检查；本模块不会因为一个文件名就证明数据真实。

## 已实现到什么程度

`modeling.py` 能处理已分割目标的度量点云、合并多个已标定视角、用真实 Open3D 做局部配准和位姿图优化、拟合圆柱侧壁或定向包围盒，并标出覆盖不足。它没有自动把复杂背景分割成目标，也没有从缺失深度中生成物体背面。

`physics.py` 提供三种简单方法：多姿态静态 F/T 读数求质量和重心；外部支点称重求重心投影，多方向称重后再求三维重心；已知各部件实测质量和重心时做加权求和。没有已验证的 F/T 传感器时，不把 xArm 关节电流、夹爪电流或控制器位姿当成物体的六维力传感器。没有真实称重读数时，质量和重心继续是未知。

MoSE3 接口只导入作者代码输出的表面轨迹，不冒充已经安装或运行了 MoSE3，更不把这些轨迹称为布料的材料模型、接触模型或物理仿真模型。代码不会下载权重。

## RGB-D 反投影

```python
from flip_pipeline.modeling import backproject_rgbd
points = backproject_rgbd(
    depth_image,                       # HxW 数组，原始深度计数
    depth_scale_m=actual_depth_scale,   # 必须来自实际相机配置
    intrinsics={
        'profile_id': '实际流配置ID', 'pixel_frame': 'camera_depth',
        'width': W, 'height': H, 'K': K.tolist(), 'rectified': True,
    },
    valid_mask=object_mask,             # HxW bool，目标分割和有效深度掩膜
    min_depth_m=minimum_valid_depth,
    max_depth_m=maximum_valid_depth,
    frame='camera_depth', evidence=['observations/frame_01/depth.npz'],
)
```

`K` 必须对应当前深度像素。如果深度对齐到了彩色流，要使用对齐后像素对应的内参。尺寸、单位、有效范围和掩膜必须显式给出；无效像素丢弃，不补造。可选 `T_world_camera`、`world_frame` 将点转到场景坐标系。

## 多视角和几何体

每次观察的格式如下，点已经分割成目标，且以相机坐标系表示：

```python
observation = {
    'observation_id': 'frame_01',
    'world_frame': 'robot_base', 'units': 'm',
    'target_segmented': True,
    'points_m': [[x, y, z], ...],
    'T_world_camera': [[...], [...], [...], [...]],
    'evidence': ['observations/frame_01/depth.npz', 'observations/frame_01/mask.png'],
}
```

`merge_known_views(observations, frame='robot_base', voxel_size_m=...)` 使用已知变换，再按体素取均值。它只表示按现有标定合并，不能证明相机外参正确。

`register_multiview_open3d(observations, frame=..., ...)` 先应用已知相机位姿，再实际调用 Open3D 的点到面 ICP 和位姿图优化。相邻视角必须双向重叠，配准残差和相对标定结果的修正必须在限制内，否则报错。只支持已有初始位姿的局部配准，不隐式调用全局识别。对称圆柱即使残差很低，绕轴转角仍可能不可观测。结果中的 `world_corrections` 是施加在初始世界点云上的修正，不能当成直接覆盖设备标定的结果。

圆柱建模调用：

```python
model = build_rigid_model(
    observations, frame='robot_base', primitive='cylinder',
    axis_hint=measured_axis,            # 可省略；提供时仅作优化初值
    endpoint_observations={
        'both_observed': True, 'frame': 'robot_base',
        'points_m': [measured_end_center_1, measured_end_center_2],
        'evidence': ['observations/endpoints.json'],
    },
    voxel_size_m=chosen_voxel_size,
    fit_tolerance_m=measured_depth_tolerance,
)
```

圆柱拟合输入应以侧壁为主；瓶肩、瓶颈、盖子等不应硬塞进单圆柱。没有两端实际观测时只输出 `observed_height_m`，不把可见部分称为完整高度。端点观测要有两端中心坐标，并检查它们与侧壁轴线、可见范围是否一致。圆柱结果明确 `axial_yaw_observable=False`。可见侧壁覆盖不足、视角不足或拟合质量差时，`status='partial'`、`usable_for_planning=False`。

盒体用 `primitive='box'`；可选 `axes_hint` 是三列分别表示盒体局部轴在世界坐标下方向的正规旋转矩阵。缺省使用 PCA 方向包围盒，并非精确最小体积盒。六个面没有得到足够点时返回不完整。透明、反光、柔软变形或几何形状不符合时不要强行通过。

桌面参数可选：`{'normal':[nx,ny,nz], 'offset_m':d, 'frame':..., 'units':'m', 'evidence':[...]}`，方程为 `n·x+d=0`，法向指向物体所在侧。桌面剔除也会删掉靠桌面的物体点，被删边界不能冒充物体真实底端。

输出 `geometry`、`quality`、`limitations`、`evidence`。`usable_for_planning=True` 只表示该简单模型通过自身数值与覆盖检查；它不包含外参系统误差、自由空间、碰撞验证，也不等于已经获得执行授权。模型误差需要与设备标定误差一起进入主流程的几何检查。

## MoSE3 轨迹导入

已核对作者的 `inference.py/run_clip` 格式（2026-10-19）：

| 名称 | 形状/含义 |
|---|---|
| `tracks` | T,H,W,3，第一帧相机坐标下的逐查询像素轨迹 |
| `visibility` | T,H,W，logit 大于 0 表示可见，不是 0～1 概率 |
| `se3_quat` | T,H,W,4，wxyz |
| `se3_trans` | T,H,W,3 |
| `se3_valid` | T,H,W，bool，局部 SE(3) 拟合是否有效 |
| `camera_poses` | T,4,4，第一帧为单位阵 |
| `intrinsics` | 3,3，模型拟合的内参 |
| `query_idx` | 整数，查询帧索引 |

```python
surface = import_mose3_tracks(
    'predictions.npz',
    target_mask=query_frame_object_mask,
    alignment={
        'status': 'verified', 'units': 'm', 'world_frame': 'robot_base',
        'scale_to_m': measured_scale,
        'T_world_first_camera': measured_transform,
        'holdout_rms_m': independent_alignment_error,
        'max_holdout_rms_m': allowed_alignment_error,
        'evidence': ['alignment/fit.json', 'alignment/holdout.json'],
    },
    evidence=['video.mp4', 'mose3_config.json'],
)
```

需要真实尺度和坐标对齐的独立验证。查询像素的物体掩膜必须提供，否则背景也会被当成目标。不可见、无效轨迹输出 JSON `null`，不填补。返回值始终是 `surface_tracks_only`；`usable_for_dynamics` 和 `usable_for_planning` 均为 False，直至另有布料接触、材料、规划等模块补足。此接口只完成可信导入和坐标对齐。

## 静态 F/T 辨识

最简单的模型是 `f=m*g+b_f`，`tau=(m*r)×g+b_tau`。先线性求质量 m、一次矩 h=m*r 和恒定零偏，再求 r。一次姿态或只有绕重力方向的转动不能测到完整三维重心。

```python
sensor_calibration = {
    'status': 'verified', 'source': 'calibrated_ft_sensor',
    'frame': 'ft_sensor', 'force_units': 'N', 'torque_units': 'N*m',
    'gravity_compensated': False,
    'target': 'complete_payload',
    'mass_distribution_fixed': True,
    'evidence': ['device/ft_calibration.json'],
}
sample = {
    'sample_id': 'static_pose_01', 'frame': 'ft_sensor',
    'static': True, 'no_external_contact': True,
    'gravity_m_s2': [gx,gy,gz],
    'force_N': [fx,fy,fz], 'torque_Nm': [tx,ty,tz],
    'evidence': ['samples/static_pose_01.json'],
}
result = estimate_static_payload(
    samples, sensor_calibration,
    holdout_samples=independent_samples,
    max_force_rms_N=sensor_force_tolerance,
    max_torque_rms_Nm=sensor_torque_tolerance,
    minimum_mass_kg=sensor_minimum_reliable_mass,
)
```

`complete_payload` 表示传感器之后全部负载，包括夹爪、支架等，不能误叫物体重心。若要测单个物体，使用 `target='object_difference'`，每个 sample 必须带 `unloaded` 子记录，包含相同姿态下空工具的 `frame/static/no_external_contact/gravity_m_s2/force_N/torque_Nm/evidence`。程序检查加载和空载的重力方向匹配后做差。

默认同时求零偏，至少需要 6 个静态姿态，并做矩阵秩、条件数及拟合残差检查。已知零偏时可用 `bias_mode='known'`，标定还必须提供 `bias_verified=True`、`force_bias_N`、`torque_bias_Nm`，至少 3 个姿态；仍要满足观测秩。姿态数不是充分条件，真正判据是不同重力方向带来的可观测性。

独立检验至少需要两个与训练集合不同且相互不平行的重力方向。没有独立检验时结果只是 `provisional`；检验失败直接报错。输出的随机误差估计不包括坐标外参或传感器系统误差。内容物可能移动、液体晃动、接触桌面、运动中受惯性力等情形不符合该固定静态模型。

## 无腕部 F/T 时的简单称重

`estimate_support_projection(supports, frame=..., gravity_m_s2=..., loadcell_calibration=...)` 接受多个支点位置和向上的实际承重力。每个支点为 `{'frame':..., 'position_m':[...], 'force_N':[...], 'evidence':[...]}`。标定为 `{'status':'verified','external_loads_only_gravity_and_supports':True,'evidence':[...]}`。

它返回质量、垂直于重力的重心投影和未观测轴。一次放平称重不知道重心高度，因此 `com_m=None`，不能把零高度当成完整重心。

在几个不同朝向下称重，把每份投影与重力轴转换到同一个物体坐标系后，调用 `combine_support_projections(projections, frame=...)`。至少两个不平行方向且质量、投影一致才能解三维重心。坐标对齐和物体内部质量分布保持不变是前提。改动朝向本身也必须先经过风险判断，不能为了称重翻转未知液体容器。

`combine_weighed_components(components, frame=...)` 只用于各部件质量和重心均已测定的情况。每项包含 `status='measured'`、`mass_kg`、`com_m`、`frame`、`evidence`；它不把不均匀物体假定为均匀密度。

## 材料和内容物

`assess_properties(observations)` 只整理证据，不执行探测。每项为：

```python
{'field':'liquid_present', 'source':'measurement', 'value':False,
 'evidence':['contents/inspection.json']}
```

字段包括 `material`、`liquid_present`、`fragile`、`hazardous_contents`、`contents_mobile`。来源区分 `measurement`、`human_attestation`、`visual_prior`、`assumption`。视觉先验和假设保留在记录中，但不能据此排除不透明容器中的液体。已有测量互相冲突时不选一个喜欢的结果；返回冲突并升级处理。

未知风险或已知风险都使 `requires_teacher_or_human=True`。这里的 `safe_for_exploration` 仅表示材料/内容物这一项通过，不替代几何检查、设备状态、速度力限制与人在场范围的判断。教师模型也不能凭空消除实际物理信息缺口。

## 依据和边界

- Open3D 官方多视角配准与重建：<https://www.open3d.org/docs/release/tutorial/pipelines/multiway_registration.html>，<https://www.open3d.org/docs/release/tutorial/reconstruction_system/index.html>。使用现成 ICP 和位姿图实现；对称几何、弱重叠和错误初值仍会失败。
- MoSE3 作者仓库和实际输出定义：<https://github.com/mose3-tracker/MoSE3>，<https://github.com/mose3-tracker/MoSE3/blob/main/inference.py>。该模型输出表面运动，不直接输出材料参数或重心。
- 传统静态 F/T 负载辨识与重力补偿：<https://github.com/kth-ros-pkg/force_torque_tools>；可对照的完整静态线性方程与独立验证实现：<https://github.com/idra-lab/end_effector_mass_estimation>。本项目公式和测试是自己的小型实现，不依赖 ROS。

本次数值测试覆盖已知真值的点云/质量重心、相机尺度与坐标错误、单视角覆盖不足、端点缺失、重力方向欠激励、错误力传感器来源、接触污染、独立验证失败、单次称重不可观测高度和先验不能排除风险。它们验证代码，不代替实机标定。

## 报告进入主流程前必须重算

`evaluation.validate_numeric_report(root, stage, data, evidence_paths)` 用于 `model` 和 `properties` 两个阶段。失败抛出 `NumericReportError`（ValueError 子类），通过时返回 `valid=True` 和重算结果摘要。它不是检查 `checks` 是否写了 pass，而是重新读取有哈希绑定的输入并调用上述计算函数，再比较报告数值。

模型报告 data 必须包含：

```json
{
  "model_request_path": "task/model_request.json",
  "model_result_path": "task/model_result.json"
}
```

`model_request.json` 的内容是 `build_rigid_model(**request)` 的完整关键字参数；`model_result.json` 必须等于实际计算结果。只有模型覆盖、拟合、尺度和端点条件通过才允许进入后续。MoSE3 表面轨迹暂不能用圆柱/盒体这条检查通道通过。

物性报告 data 必须包含：

```json
{
  "property_observations_path": "task/property_observations.json",
  "property_result_path": "task/property_result.json",
  "mass_status": "measured",
  "mass_method": "weighed-components",
  "mass_request_path": "task/mass_request.json",
  "mass_result_path": "task/mass_result.json"
}
```

`property_observations.json` 是观测列表，结果重新调用 `assess_properties` 计算；材料仅为视觉先验或内容物风险不清楚均不能通过。`mass_method` 可取 `static-payload`、`weighed-components`、`combine-projections`，request 分别是对应函数的关键字参数。质量/重心结果必须是 `measured`，暂不接受均匀密度或参数区间替代测量；如果后续要允许有界假设，必须先实现对全部允许参数范围的实际规划检验，不能只多填一个字典。

物体任务的 `static-payload` 必须使用 `target='object_difference'`，有匹配的空工具数据和独立验证；整个工具总负载的重心不等于物体重心。部件称重法和支点称重法必须保证输入已在同一物体坐标系。

`evidence_paths` 可传工作区内文件路径列表，或主流程已经计算的 `{path,sha256}` 列表。请求、结果、以及所有嵌套的 `evidence/evidence_refs` 引用都要列入。算法运行前后各验证一次哈希，避免计算期间源文件变化。模型/物性数值使用有限数值比较，不能用 NaN、布尔值或额外字段偷偷覆盖结果。

这一检查证明“结果确实由留存输入计算得到”，不能单凭 JSON 证明输入来自实机；任务、物体、安装版本、传感器来源和人工确认仍由主流程管理。
