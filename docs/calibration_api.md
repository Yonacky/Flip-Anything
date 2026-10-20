# 设备标定：采集、计算、复核和配置

这是独立的设备标定工作，不与抓取新物体混在一起。实现位于 `flip_pipeline/calibration.py`，所有函数只计算输入数据，不连接设备、不移动机械臂、不改变控制器参数。当前没有写入真实相机安装变换、指垫中心、工具尺寸或桌面高度；合成测试中的数字不能作为设备配置。

## 坐标与单位

内部统一米、弧度、列向量。`a_from_b` 的含义是 `p_a = a_from_b @ p_b`。齐次矩阵必须是 4×4，最后一行为 `[0,0,0,1]`，旋转为 SO(3)，禁止把欧拉角直接填进旋转矩阵。

`tool_frame` 必须显式填 `controller_tcp` 或 `flange`。如果使用控制器返回的 TCP 位姿，就使用前者，并把当时 TCP 配置的 SHA256 绑定到安装档案。不同采集点不得混用 TCP 和法兰。控制器提供毫米/角度时，采集适配器先按经确认的约定转换；本模块不会猜测或静默换单位。

眼在手上标定使用 `base_from_tool` 和 `camera_from_target`，结果为 `tool_from_camera`。固定标定板满足 `base_from_tool @ tool_from_camera @ camera_from_target = base_from_target`。

## 采集要求

首先保存机械臂/相机序列号、工具身份、安装版本、TCP 配置原始回执、SDK 流模式与内参。内参必须属于实际投影用的流：原始深度、彩色、对齐后的深度不能混用。保存深度单位以及畸变模型的明确映射；当前实现只接受 OpenCV Brown-Conrady 参数顺序。若 SDK 使用其他模型，应实现并验证转换，不能改一个名称就继续。

使用尺寸经过测量的刚性标定板，固定在机械臂基座坐标系不动的位置。保持角点编号与板坐标一一对应。样本默认至少 6 个训练视角和 3 个验证视角，实际采集可多于下限。必须从两个以上非平行方向改变姿态，只有平移或只绕同一轴转动不够。不要为了满足数目重复保存同一个姿态。

采集前确定 train/validation 划分。验证样本不参与求解，也不用于挑选“最好看的”阈值和算法。每次先停稳，再取得曝光对应的控制器位姿；相机与控制器时间先映射到同一个单调时钟。只有两个时间戳数值相近、却属于不同设备时钟，不算同步。保存原始图像、角点、板尺寸、原始位姿、换算过程与时间记录。

采集动作本身需要使用项目既有执行器和监督机制。本模块没有自动探测安全轨迹、也没有自动执行标定动作的权限。没有可识别的刚性标定板和可靠时间映射时，状态保持 draft，不能用拟合桌面法向的单轴探针冒充完整六自由度标定。

## 函数与准确的数据格式

下列字段均为必需，除非明确标注可选。所有 `evidence_ref` 是项目证据存储中的引用，不是文字结论。计算函数会检查字段内容，但文件存在性、哈希和采集来源由项目证据存储校验。

### `estimate_target_pose(request) -> dict`

输入如下。`matrix3x3`、`pointsNx3`、`pointsNx2` 在实际 JSON 中必须替换为数值数组，不是字符串。

```text
request = {
  units: {translation: "m", rotation: "rad"},
  target_geometry_evidence_ref: nonempty-string,
  image_evidence_ref: nonempty-string,
  intrinsics: {
    width: positive-integer, height: positive-integer,
    camera_matrix: matrix3x3,
    distortion_model: "opencv_brown_conrady",
    distortion_coefficients: finite-array-length-4|5|8|12|14
  },
  object_points_m: pointsNx3,
  image_points_px: pointsNx2,
  quality_limits?: {
    max_reprojection_error_px?: positive-number,
    min_image_span_px?: positive-number,
    min_planar_alternative_gap_px?: positive-number
  }
}
```

至少六个不同的角点，不能共线。平面板应在自己的坐标系中定义为 z=0；它使用 OpenCV IPPE，并比较候选解的重投影误差。候选解无法区分时返回错误，需要改变观察角度，不能任选一个。非平面目标使用 SQPNP。返回 `camera_from_target`、`reprojection_error_px`、候选误差、输入哈希和证据引用。默认误差上限 1 像素、图像两个方向跨度都至少 30 像素、平面候选误差差至少 0.25 像素。这些是数据质量阈值，不是实机安装测量值。

当前函数从已检测、已核对对应关系的角点开始；它不会自动识别任意标定板，也没有臆造棋盘格尺寸。

### `solve_hand_eye(request) -> dict`

```text
request = {
  units: {translation: "m", rotation: "rad"},
  frame_convention: {
    robot_pose: "base_from_tool", target_pose: "camera_from_target"
  },
  tool_frame: "controller_tcp" | "flange",
  installation_id: nonempty-string,
  target_id: nonempty-string,
  target_geometry_evidence_ref: nonempty-string,
  camera_intrinsics_sha256: lowercase-64-hex-string,
  target_fixed_in_base: true,
  split_policy: "predeclared",
  samples: [
    {
      sample_id: unique-nonempty-string,
      capture_id: unique-nonempty-string,
      evidence_ref: unique-nonempty-string,
      split: "train" | "validation",
      base_from_tool: rigid-matrix4x4,
      camera_from_target: rigid-matrix4x4,
      image_timestamp_s: nonnegative-finite-number,
      pose_timestamp_s: nonnegative-finite-number,
      timestamp_clock: "shared_monotonic",
      robot_stationary: true,
      reprojection_error_px: nonnegative-finite-number
    }
  ],
  quality_limits?: known-limit-name-to-positive-number
}
```

`camera_intrinsics_sha256` 必须使用实际用于目标位姿求解的完整内参档案计算，算法为：`sha256(json.dumps(intrinsics, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()`。不能只对 K 做哈希，漏掉分辨率、畸变和流信息。

可调整阈值及默认值见代码中的 `DEFAULT_LIMITS`。未知阈值名会报错。默认时间差上限 0.03 秒；训练/验证平移残差上限分别 0.003/0.005 米，旋转残差上限 0.02/0.03 弧度；非平行旋转、平移系统秩和条件数均检查。5 米样本平移与 1 米手眼平移上限用于拦截明显的毫米/米混淆，不代表机器人可达范围或允许运动范围。具体实验对误差的要求可能更严，必须由维护者制定，不由学生为了通过检查自行放宽。

求解固定使用 OpenCV `calibrateHandEye(..., CALIB_HAND_EYE_PARK)`。训练数据估计手眼变换及固定目标在基座下的位置；验证数据只用于比较，不重新求均值消除验证偏差。样本 ID、采集 ID 和证据引用不得重复，即使改名，只要训练/验证复制了相同位姿对也会拒绝。

返回包括 `tool_from_camera`、训练和验证逐样本残差、运动多样性、样本 ID、输入哈希、阈值与依赖版本。`status` 一律为 `draft`。留出验证证明的是固定目标的一致性，不能排除标定板尺寸错误、控制器模型系统误差、共同尺度错误等问题，也不能当作独立绝对精度测量。

### `validate_factory_intrinsics(profile, observed_stream, verification) -> dict`

`profile` 包含以下字段：

```text
{
  camera_serial: nonempty-string,
  stream_id: nonempty-string,
  alignment: nonempty-string,
  distortion_model: "opencv_brown_conrady",
  width: positive-integer, height: positive-integer,
  camera_matrix: matrix3x3,
  distortion_coefficients: finite-array-length-4|5|8|12|14,
  depth_scale_m_per_unit: positive-number,
  sdk_evidence_ref: nonempty-string
}
```

`observed_stream` 包含相同字段，只有 `sdk_evidence_ref` 可省。它来自本次读取的 SDK，而不是复制档案。序列号、流、对齐方式、尺寸、K、畸变和深度尺度必须匹配。

`verification` 包含：

```text
{
  evidence_ref: nonempty-string,
  independent_reference: true,
  max_depth_error_m: positive-number,
  depth_checks: [
    {reference_camera_z_m: positive-number, raw_depth_units: positive-number}
  ]
}
```

至少两个可区分的距离测量。`reference_camera_z_m` 是相机光学坐标的 Z 深度，不是倾斜表面的欧氏距离。参考值来自独立测量，不能由同一幅深度图反算。此函数验证 SDK 配置绑定和已报告的深度尺度检查，不声称重新标定整个镜头畸变。

### `validate_tool_envelopes(boxes) -> list`

每个元素为：

```text
{
  id: unique-nonempty-string,
  component: "camera" | "mount" | "gripper" | "other",
  frame: "tool", units: "m",
  tool_from_box: rigid-matrix4x4,
  half_extents_m: positive-number-array-length-3,
  uncertainty_margin_m: nonnegative-number,
  measurement_evidence_ref: nonempty-string,
  opening_range_m: [minimum-opening, maximum-opening]  # gripper 时必需
}
```

返回保留原字段，并加入含误差余量的 `corners_tool_m`。包络必须保守覆盖实际硬件，不是只围住相机光心。夹爪包络只在所声明的开口范围内有效；调用者必须核对当前开口。函数不检查机械臂各连杆、自碰撞或轨迹，后者属于规划/监督模块。

### `validate_device_profile(profile, current_installation, require_verified=True) -> dict`

完整档案必须含以下字段：

```text
{
  status: "draft" | "verified",
  units: {translation: "m", rotation: "rad"},
  tool_frame: "controller_tcp" | "flange",
  installation: {
    installation_id: nonempty-string,
    arm_serial: nonempty-string,
    camera_serial: nonempty-string,
    tool_id: nonempty-string,
    mount_revision: nonempty-string,
    tcp_config_sha256: nonempty-string
  },
  tool_from_camera: rigid-matrix4x4,
  tool_from_grasp_center: rigid-matrix4x4,
  grasp_center_calibration: {
    evidence_ref: nonempty-string,
    method: nonempty-string,
    opening_range_m: [minimum-opening, maximum-opening],
    max_position_error_m: positive-number
  },
  tool_envelopes: boxes-as-above,
  intrinsics: intrinsics-profile-as-above,
  observed_stream: current-sdk-stream-as-above,
  intrinsics_verification: depth-verification-as-above,
  hand_eye_request: full-solve-request-as-above,
  hand_eye_result: full-solve-result,
  review: {
    reviewer_role: "maintainer" | "human",
    reviewer_id: nonempty-string,
    reviewed_at: nonempty-string,
    evidence_ref: nonempty-string,
    profile_sha256: lowercase-64-hex-string,
    independent_physical_check_passed: true,
    physical_check_evidence_ref: nonempty-string
  }
}
```

draft 可省 review，调用 `require_verified=False` 仅验证其结构和计算证据，不能据此执行。当前结构不接受未测量的 null 作为有效设备档案，初始未测模板需先完成采集。

`current_installation` 与 `installation` 结构相同，来自本次设备和安装检查。完整工具包络至少覆盖 camera、mount、gripper 三类。指垫的有效夹持中心不一定对所有开口恒定，因此需要测量证据、允许的开口范围和误差上限；超出范围应补测，不能延用一个固定点。

校验会重新计算 `hand_eye_request`，对照结果、内参哈希、工具帧和安装身份。review 中的 `profile_sha256` 使用 `profile_content_sha256(profile)` 计算，任何几何或标定输入改变都会使原复核失效。

独立实物检查应采用未参与求解的可测目标，检验基座坐标位置和方向、夹持中心，以及工具实际包络。必须保留测量原始值与误差，不得把训练残差截图当作独立实物检查。维护者或人类完成这项复核后才建立 verified 档案；学生只准备材料。该复核按安装版本进行，不要求每个动作重新人工批准。

本函数是数据一致性检查，不是抵御恶意修改的权限边界。维护者身份、原始证据不可篡改、review 写入权限应由外层系统保证。

## 已验证与尚未完成的范围

合成测试覆盖非平凡手眼变换恢复、留出验证失败、重复样本、单轴/纯平移退化、单位和变换方向错误、时间偏差、平面 PnP 歧义、内参串用、安装变化、工具包络和复核哈希。它们证明计算及拒绝逻辑可重复，不能声称真实 xArm 与 Gemini 已完成标定。

相机采集适配器、目标角点检测、SDK 到共同时间的映射、已测物理目标、夹持中心测量和独立实物复核仍需专门标定任务完成。程序缺数据时明确拒绝，不用历史罐子坐标或随手估计填补。

## 实现依据

采用 OpenCV 官方定义的手眼输入输出和 Park 方法，以及 IPPE/SQPNP 目标位姿求解。参考：[OpenCV Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)。本项目增加了样本绑定、数据质量、留出验证及配置复核规则；这些规则不是 OpenCV 对实机安全的保证。
