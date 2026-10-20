"""Write the task outcome and the verified scene file (no hardware access)."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(r'G:\Flip Anything\ds_control')
mount = json.loads((ROOT.parent / 'logs' / 'experiments' / 'ds_control' / 'DS_20261018T203231_b2ac5b99' / 'mount.json').read_text())
Rc = np.array(mount['R_tcp_cam']); t = np.array(mount['t_tcp_cam_mm'])
T = np.eye(4); T[:3, :3] = Rc; T[:3, 3] = t

out = {
    'schema_version': 1,
    'task': '桌面容器翻转与放置（倒立、松爪、独立站稳）',
    'date_local': '2026-10-18',
    'executor': 'deepseek-flash via DSH',
    'runs': [
        'G:/Flip Anything/logs/experiments/ds_control/DS_20261018T203231_b2ac5b99 (第一轮失败 + 相机/TCP 自标定 14 位姿)',
        'G:/Flip Anything/logs/experiments/ds_control/DS_20261018T205921_77e61598 (俯仰位姿补充)',
        'G:/Flip Anything/logs/experiments/ds_control/DS_20261018T210456_00845c9c (抓取-翻转-落放全过程)',
        'G:/Flip Anything/logs/experiments/ds_control/DS_20261018T211223_dcd3ca39 (落放后两次稳定确认)'],
    'physical_result': {
        'container_flipped': True,
        'container_placed': True,
        'container_state': '倒立、脱离夹爪、独立站在桌面；绿色轮廓包围盒间隔 45 s 两次观测完全一致 (u 565-788, v 275-719)，质心位移 0.5 px',
        'lean_deg_estimate': '约 8-10 度（深度圆拟合两高度段轴心差），远小于翻倒角约 21 度；用户确认站稳',
        'note': '罐子为已开罐、空罐；第一轮掉落后罐口可能轻微变形'},
    'robot_state_at_end': {
        'controller_state': 4, 'error_code': 0, 'warning_code': 0,
        'pose_mm_deg': [405.9, -4.6, 97.3, 180.0, 89.5, 180.0],
        'gripper': {'gACT': 1, 'gSTA': 3, 'gPO': 3, 'gOBJ': 3, 'gFLT': 0, 'state': '空载张开'},
        'session': 'shutdown 正常结束，无遗留 session.lock'},
    'autonomous_success': False,
    'human_intervention': [
        '第一轮把罐子推倒并掉落后，用户重新摆放罐子并移动机械臂',
        '用户要求放弃旧参数与标定板，只用相机+机械臂',
        '用户确认稳定、要求停止进一步核实'],
    'support_evidence': '罐口离桌 2.8 mm 时保持不动张爪 (gPO=3, gOBJ=3)；随后倒立站稳的观测帧',
    'stability_evidence': 'G:/Flip Anything/logs/experiments/ds_control/DS_20261018T211223_dcd3ca39 frame_000001 vs frame_000002（间隔 45 s）',
    'verified_relationships': {
        'camera_to_tcp': '本次自标定：t=(-72.91,10.05,-80.06) mm，|t|=109.0 mm',
        'calibration_method': '桌平面作水平参考 + 已知工具位姿（14 位姿：平移/滚转 ±50,±90/俯仰 ±20）；Procrustes 求旋转、线性最小二乘求平移；未用标定板、未用历史参数',
        'residuals': '法向角残差均值 0.59 度 最大 2.08 度；高度方程残差 std 4.08 mm',
        'table_height_base_z': -8.3,
        'depth_scale': '名义 1.0；<250 mm 不可用（罐面有效率 5.9%），~300 mm 有效 78%'},
    'target_measurements': {
        'object': '绿色细高饮料罐（已开、空）',
        'axis_base_xy': [525.9, -4.6],
        'diameter_mm': 55.0,
        'top_height_above_table_mm': 146,
        'method': '罐身深度点水平圆拟合，残差中位 0.35 mm；两视角轴心差 1.2 mm'},
    'grasp_rule_used': '把报告 TCP 直接走到罐轴（两指中间），不再使用历史上的 +44 mm 指垫偏置',
    'failures_and_anomalies': [
        '第一轮按旧参数把 TCP 送到罐轴 +44 mm，把站立罐推倒；罐子随后掉到地上（真实事故，未自动补救）',
        '会话三次意外结束：live_state.json.tmp OSError 22；Incomplete source RGB-D frameset 两次',
        '已最小修复两个已定位接口缺陷：session.write_result 唯一临时名+重试不致命；camera.capture 不完整帧集重试 6 次',
        '多次规划层拒绝（未下发动作）：Joint 49.0 deg、Translation 110.1 mm、Rotation 100.0/91.7/78.7 deg',
        '深度传感器近距离不可靠，改用彩色轮廓 + 可用距离的深度圆拟合'],
    'next_steps': [
        '若需更精确的落放姿态，可在落放前用小步测量修正罐口-桌面间隙',
        '更换指垫或相机支架后应重做本次自标定并更新 mount.json'],
}
(ROOT.parent / 'logs' / 'experiments' / 'ds_control' / 'outcome.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf8')

scene = {
    'schema_version': 1, 'status': 'verified', 'translation_unit': 'mm',
    'camera_serial': 'CP9E1630019L',
    'verified_at': '2026-10-18T21:14 host',
    'method': 'camera+arm only: table plane as horizontal reference across 14 diverse tool poses; no calibration board, no historical parameters',
    'T_tcp_camera': np.round(T, 6).tolist(),
    't_tcp_camera_mm': [round(float(v), 2) for v in t],
    'table_height_base_z_mm': -8.3,
    'residuals': {'plane_normal_deg_mean': 0.59, 'plane_normal_deg_max': 2.08,
                  'height_lstsq_std_mm': 4.08, 'n_poses': 14},
    'target': {'object': 'green slim can (open, empty)', 'axis_base_xy_mm': [525.9, -4.6],
               'diameter_mm': 55.0, 'top_above_table_mm': 146.0,
               'localized_by': 'horizontal circle fit on can body depth points (res median 0.35 mm)'},
    'grasp_convention': 'reported TCP placed on the can axis (between the fingers); old +44 mm pad offset not used',
    'result': 'inverted, released, standing on the table; interval stability confirmed',
}
(ROOT / 'scene' / 'current.json').write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding='utf8')
print('records written')

