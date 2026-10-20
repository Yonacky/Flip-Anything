"""Round-scoped read-only analysis helpers for run DS_20261019T000057_a687f267.

Nothing here moves the arm or talks to the camera: it only reads this run's
recorded frames/results (and, for comparison, the historical mount as an
unverified prior). Historical scene coordinates are never written back.
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r'G:\Flip Anything\ds_control')
LOGS = ROOT.parent / 'logs' / 'experiments' / 'ds_control'
def _active_run_id():
    """Follow the session that is actually holding the hardware."""
    try:
        return json.loads((ROOT / 'active.json').read_text(encoding='utf-8'))['run']
    except Exception:
        return 'DS_20261019T000057_a687f267'


RUN_ID = _active_run_id()
RUN = LOGS / RUN_ID
PRIOR_MOUNT = LOGS / 'DS_20261018T203231_b2ac5b99' / 'mount.json'

sys.path.insert(0, str(ROOT / 'src'))
from flip_anything.geometry import pose_matrix  # noqa: E402


def run_dir(run_id=None):
    return LOGS / (run_id or RUN_ID)


def _events(run):
    with (run / 'events.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def intrinsics(run):
    """Color intrinsics from this run's own camera_started event."""
    for row in _events(run):
        if row['kind'] == 'camera_started':
            return row['payload']['metadata']['factory_camera_param']['color_intrinsic']
    raise RuntimeError('no camera_started event in %s' % run)


def load_mount(path):
    data = json.loads(Path(path).read_text())
    return {'R_tcp_cam': np.array(data['R_tcp_cam'], dtype=float),
            't_tcp_cam_mm': np.array(data['t_tcp_cam_mm'], dtype=float),
            'table_height_mm': float(data['table_height_mm']),
            'n_samples': data.get('n_samples'),
            'source': str(path)}


def camera_pose(tcp_pose, mount):
    """Camera origin and orientation (columns = camera axes) in base coordinates."""
    T = np.array(pose_matrix([float(v) for v in tcp_pose]))
    R_tool, tcp = T[:3, :3], T[:3, 3]
    R_cam = R_tool @ mount['R_tcp_cam']
    cam = tcp + R_tool @ mount['t_tcp_cam_mm']
    return R_cam, cam


def backproject(depth, k, R_cam, cam, u, v):
    """Base-frame points for pixels: deproject at recorded depth, convert to base."""
    u = np.atleast_1d(np.asarray(u, dtype=float))
    v = np.atleast_1d(np.asarray(v, dtype=float))
    ui = np.clip(u.astype(int), 0, depth.shape[1] - 1)
    vi = np.clip(v.astype(int), 0, depth.shape[0] - 1)
    d = depth[vi, ui].astype(float)
    ok = d > 0
    pts = np.full((u.size, 3), np.nan)
    if ok.any():
        p_cam = np.stack([(u[ok] - k['cx']) * d[ok] / k['fx'],
                          (v[ok] - k['cy']) * d[ok] / k['fy'], d[ok]], 1)
        pts[ok] = p_cam @ R_cam.T + cam
    return pts


def fit_plane(depth, k, lo=150.0, hi=900.0, iters=600, thresh=4.0, seed=7, mask=None):
    """RANSAC plane in the camera frame -> dict with n, d, dist, inliers, res_p95."""
    h, w = depth.shape
    step = 3
    vv, uu = np.mgrid[0:h:step, 0:w:step]
    zz = depth[::step, ::step]
    sel = (zz > lo) & (zz < hi)
    if mask is not None:
        sel = sel & mask[::step, ::step]
    if sel.sum() < 500:
        return None
    pts = np.stack([(uu[sel] - k['cx']) * zz[sel] / k['fx'],
                    (vv[sel] - k['cy']) * zz[sel] / k['fy'], zz[sel]], 1)
    rng = np.random.default_rng(seed)
    sub = pts[rng.choice(len(pts), min(20000, len(pts)), replace=False)]
    best = None
    for _ in range(iters):
        a, b, c = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(b - a, c - a)
        L = np.linalg.norm(n)
        if L < 1e-9:
            continue
        n = n / L
        d = -n @ a
        cnt = int((np.abs(sub @ n + d) < thresh).sum())
        if best is None or cnt > best[0]:
            best = (cnt, n, d)
    if best is None:
        return None
    _, n, d = best
    inl = np.abs(pts @ n + d) < thresh
    if inl.sum() < 300:
        return None
    # Cap the refinement set: this host runs out of memory on a full-frame SVD.
    ref = pts[inl]
    if len(ref) > 15000:
        ref = ref[rng.choice(len(ref), 15000, replace=False)]
    m = ref.mean(0)
    n = np.linalg.svd(ref - m)[2][-1]
    d = -n @ m
    if n[2] > 0:                     # normal points back toward the camera
        n, d = -n, -d
    res = np.abs(pts[inl] @ n + d)
    return {'n': n, 'd': float(d), 'dist': float(abs(d)), 'inliers': int(inl.sum()),
            'res_p95': float(np.percentile(res, 95))}


def plane_in_base(plane, R_cam, cam):
    """Express a camera-frame plane in base coordinates."""
    n_base = R_cam @ plane['n']
    d_base = plane['d'] - float(n_base @ cam)
    tilt_deg = math.degrees(math.acos(min(1.0, abs(float(n_base[2])))))
    return {'n': n_base, 'd': float(d_base), 'z_at_origin': float(-d_base),
            'tilt_deg': float(tilt_deg)}


def results_with_camera(run):
    """Recorded results that carry a frame, oldest first, pose paired with frame."""
    out = []
    for path in sorted(run.glob('result_*.json'), key=lambda p: p.stat().st_mtime):
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        cam = data.get('camera')
        if data.get('ok') and cam and cam.get('color_file'):
            out.append({'result': path, 'pose': data['state']['pose'],
                        'color': run / cam['color_file'], 'depth': run / cam['depth_file']})
    return out


def load_depth(path):
    return np.load(path).astype(np.float32)


def read_color(path, flags=cv2.IMREAD_COLOR):
    return cv2.imread(str(path), flags)


def rpy_of(R):
    return [math.degrees(math.atan2(R[2, 1], R[2, 2])),
            math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0])))),
            math.degrees(math.atan2(R[1, 0], R[0, 0]))]
