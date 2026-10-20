"""Measured geometry only: metric registration, primitive fitting and MoSE3 import.

All transforms are column-vector T_target_source. Distances are metres. The
functions do not capture images, contact hardware, infer density or authorize moves.
Open3D is optional; missing dependencies produce an explicit error, never a mock.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares


class ModelInputError(ValueError):
    pass


def _array(value, shape=None, name="array"):
    arr = np.asarray(value, dtype=float)
    if (shape is not None and arr.shape != shape) or not np.isfinite(arr).all():
        raise ModelInputError(f"{name}: wrong shape or nonfinite values")
    return arr


def _transform(value):
    arr = _array(value, (4, 4), "T_world_camera")
    if not np.allclose(arr[3], [0, 0, 0, 1], atol=1e-7):
        raise ModelInputError("transform must be homogeneous")
    rot = arr[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rot), 1, atol=1e-5):
        raise ModelInputError("transform rotation must be proper, not a scale/reflection")
    return arr


def _evidence(value):
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x.strip() for x in value):
        raise ModelInputError("nonempty evidence references required")
    return list(value)


def _points(value, minimum=12):
    arr = _array(value, name="points_m")
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) < minimum:
        raise ModelInputError(f"points_m must be Nx3 with at least {minimum} points")
    return arr


def _clouds(observations, frame):
    if not isinstance(frame, str) or not frame.strip() or not observations:
        raise ModelInputError("explicit world frame and observations required")
    ids, clouds, transforms, evidence = set(), [], [], []
    for obs in observations:
        if obs.get("units") != "m" or obs.get("target_segmented") is not True:
            raise ModelInputError("supply segmented target points in metres, not an entire scene")
        if not isinstance(obs.get("observation_id"), str) or obs["observation_id"] in ids:
            raise ModelInputError("each observation needs a unique observation_id")
        if obs.get("world_frame") != frame:
            raise ModelInputError("observation/world frame mismatch")
        ids.add(obs["observation_id"])
        points, T = _points(obs["points_m"]), _transform(obs["T_world_camera"])
        clouds.append(points @ T[:3, :3].T + T[:3, 3])
        transforms.append(T)
        evidence.extend(_evidence(obs.get("evidence")))
    return clouds, transforms, list(dict.fromkeys(evidence))


def merge_known_views(observations, *, frame, voxel_size_m=0.001):
    """Merge segmented views using already calibrated camera poses; no ICP implied."""
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ModelInputError("positive voxel_size_m required")
    clouds, transforms, evidence = _clouds(observations, frame)
    points = np.concatenate(clouds)
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    merged = np.column_stack([np.bincount(inverse, weights=points[:, i]) / counts for i in range(3)])
    diverse = False
    for i, a in enumerate(transforms):
        for b in transforms[i + 1:]:
            angle = np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1))
            diverse |= bool(np.linalg.norm(a[:3, 3] - b[:3, 3]) >= 0.01 or angle >= np.deg2rad(10))
    return {"status": "merged", "method": "calibrated_poses_voxel_centroids", "frame": frame,
            "units": "m", "points_m": merged.tolist(), "view_count": len(clouds),
            "view_diversity": bool(diverse), "evidence": evidence,
            "note": "View diversity alone does not establish surface coverage or correct extrinsics."}


def register_multiview_open3d(observations, *, frame, voxel_size_m=0.002,
                            max_correspondence_m=0.006, min_fitness=0.35,
                            max_rmse_m=0.003, max_correction_m=0.01,
                            max_correction_deg=5):
    """Actual Open3D point-plane ICP and pose-graph refinement of known poses.

    The first calibrated view fixes the world frame. Each adjacent pair must
    overlap in both directions. Symmetric surfaces can still be ambiguous;
    accepted registration is an estimate, not calibration certification.
    """
    for x in (voxel_size_m, max_correspondence_m, max_rmse_m, max_correction_m, max_correction_deg):
        if not np.isfinite(x) or x <= 0:
            raise ModelInputError("registration tolerances must be positive")
    if not 0 < min_fitness <= 1:
        raise ModelInputError("min_fitness must lie in (0,1]")
    clouds, _, evidence = _clouds(observations, frame)
    if len(clouds) < 2:
        raise ModelInputError("multi-view registration needs at least two views")
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is not installed; no registration was performed") from exc
    reg = o3d.pipelines.registration
    pcds = []
    for points in clouds:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd = pcd.voxel_down_sample(voxel_size_m)
        if len(pcd.points) < 20:
            raise ModelInputError("too few points after downsampling")
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size_m * 4, max_nn=40))
        pcds.append(pcd)
    graph = reg.PoseGraph()
    graph.nodes.extend([reg.PoseGraphNode(np.eye(4)) for _ in pcds])
    edges = []
    for i in range(len(pcds)):
        for j in range(i + 1, len(pcds)):
            result = reg.registration_icp(pcds[i], pcds[j], max_correspondence_m, np.eye(4),
                                          reg.TransformationEstimationPointToPlane())
            reverse = reg.evaluate_registration(pcds[j], pcds[i], max_correspondence_m,
                                                np.linalg.inv(result.transformation))
            accepted = min(result.fitness, reverse.fitness) >= min_fitness and result.inlier_rmse <= max_rmse_m
            if not accepted:
                if j == i + 1:
                    raise ModelInputError(f"adjacent views {i},{j} do not satisfy overlap/residual limits")
                continue
            info = reg.get_information_matrix_from_point_clouds(pcds[i], pcds[j], max_correspondence_m,
                                                                 result.transformation)
            graph.edges.append(reg.PoseGraphEdge(i, j, result.transformation, info, uncertain=j != i + 1))
            edges.append({"source": i, "target": j, "fitness": float(result.fitness),
                          "reverse_fitness": float(reverse.fitness), "rmse_m": float(result.inlier_rmse)})
    reg.global_optimization(graph, reg.GlobalOptimizationLevenbergMarquardt(),
                            reg.GlobalOptimizationConvergenceCriteria(),
                            reg.GlobalOptimizationOption(max_correspondence_distance=max_correspondence_m,
                                                         edge_prune_threshold=0.25, reference_node=0))
    combined = o3d.geometry.PointCloud()
    corrections = []
    for pcd, node in zip(pcds, graph.nodes):
        correction = _transform(node.pose)
        shift = np.linalg.norm(correction[:3, 3])
        turn = np.rad2deg(np.arccos(np.clip((np.trace(correction[:3, :3]) - 1) / 2, -1, 1)))
        if shift > max_correction_m or turn > max_correction_deg:
            raise ModelInputError("registration contradicts calibrated poses; inspect scene/extrinsics")
        pcd.transform(correction)
        combined += pcd
        corrections.append(correction.tolist())
    return {"status": "estimated", "method": "open3d_pose_graph_point_plane_icp", "frame": frame,
            "units": "m", "points_m": np.asarray(combined.voxel_down_sample(voxel_size_m).points).tolist(),
            "world_corrections": corrections, "edges": edges, "evidence": evidence,
            "limitations": ["Symmetric geometry does not uniquely determine rotation.",
                            "ICP is local and needs calibrated initial poses."]}


def separate_table(points_m, table_plane, *, frame, clearance_m=0.003):
    points = _points(points_m)
    if table_plane.get("frame") != frame or table_plane.get("units") != "m":
        raise ModelInputError("table plane frame/units mismatch")
    _evidence(table_plane.get("evidence"))
    normal = _array(table_plane.get("normal"), (3,), "table normal")
    norm = np.linalg.norm(normal)
    if norm < 1e-8 or not np.isfinite(clearance_m) or clearance_m < 0:
        raise ModelInputError("invalid table plane")
    d = float(table_plane["offset_m"])
    if not np.isfinite(d):
        raise ModelInputError("nonfinite plane offset")
    # Equation n.x + d = 0; n points from the table toward the object.
    distances = (points @ normal + d) / norm
    keep = distances > clearance_m
    return {"points_m": points[keep].tolist(), "removed_count": int((~keep).sum()),
            "note": "The removed band can also hide the true object endpoint; it is not evidence of the base."}


def _basis(axis):
    axis = axis / np.linalg.norm(axis)
    seed = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    return np.column_stack([u, np.cross(axis, u), axis])


def _fit_cylinder(points, axis_hint, tolerance):
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    candidates = [np.asarray(axis_hint, float)] if axis_hint is not None else list(vt)
    fits = []
    for seed in candidates:
        if seed.shape != (3,) or not np.isfinite(seed).all() or np.linalg.norm(seed) < 1e-8:
            raise ModelInputError("axis_hint must be a nonzero finite 3-vector")
        axis = seed / np.linalg.norm(seed)
        radius = np.median(np.linalg.norm(np.cross(points - center, axis), axis=1))
        def residual(p):
            a = p[3:6] / max(np.linalg.norm(p[3:6]), 1e-12)
            delta = points - p[:3]
            radial = np.linalg.norm(np.cross(delta, a), axis=1) - p[6]
            # Fix the axial gauge and keep the optimized axis norm well behaved.
            return np.r_[radial, np.dot(p[:3] - center, a), (np.linalg.norm(p[3:6]) - 1) * max(radius, 0.01)]
        fit = least_squares(residual, np.r_[center, axis, max(radius, 1e-5)],
                            bounds=(np.r_[[-np.inf] * 6, 1e-6], np.full(7, np.inf)),
                            loss="soft_l1", f_scale=tolerance, max_nfev=600)
        if not fit.success:
            continue
        a = fit.x[3:6] / np.linalg.norm(fit.x[3:6])
        radial_residual = residual(fit.x)[:-2]
        fits.append((float(np.median(np.abs(radial_residual))), fit.x[:3], a, float(fit.x[6]), radial_residual))
    if not fits:
        raise ModelInputError("cylinder optimization did not converge")
    _, c, axis, radius, residuals = min(fits, key=lambda x: x[0])
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    local = (points - c) @ _basis(axis)
    low, high = np.quantile(local[:, 2], [0.005, 0.995])
    angular = np.mod(np.arctan2(local[:, 1], local[:, 0]), 2 * np.pi)
    bin_count = len(np.unique(np.floor(angular / (2 * np.pi) * 36).astype(int)))
    return {"kind": "cylinder", "axis": axis.tolist(), "center_m": (c + axis * (low + high) / 2).tolist(),
            "radius_m": radius, "observed_height_m": float(high - low),
            "axial_endpoints_m": [(c + low * axis).tolist(), (c + high * axis).tolist()],
            "angular_bin_coverage": bin_count / 36, "axial_yaw_observable": False,
            "rms_m": float(np.sqrt(np.mean(residuals ** 2))),
            "inlier_fraction": float(np.mean(np.abs(residuals) <= 2 * tolerance)),
            "fit_scope": "segmented cylinder sidewall; endpoint extent is observation-limited"}


def _fit_box(points, axes_hint, tolerance):
    center = points.mean(axis=0)
    if axes_hint is None:
        _, _, vt = np.linalg.svd(points - center, full_matrices=False)
        axes = vt.T
        if np.linalg.det(axes) < 0:
            axes[:, 2] *= -1
    else:
        axes = _array(axes_hint, (3, 3), "box axes")
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5) or np.linalg.det(axes) < 0.99999:
            raise ModelInputError("box axes must be a proper rotation")
    local = (points - center) @ axes
    low, high = np.quantile(local, [0.005, 0.995], axis=0)
    extent = high - low
    residuals = np.min(np.minimum(np.abs(local - low), np.abs(local - high)), axis=1)
    face_counts = [int(np.sum(np.abs(local[:, i] - edge) <= 2 * tolerance)) for i in range(3) for edge in (low[i], high[i])]
    return {"kind": "box", "center_m": (center + axes @ ((low + high) / 2)).tolist(),
            "axes_world": axes.tolist(), "extents_m": extent.tolist(),
            "method": "pca_oriented_bounding_box" if axes_hint is None else "known_axes_bounding_box",
            "rms_m": float(np.sqrt(np.mean(residuals ** 2))),
            "inlier_fraction": float(np.mean(residuals <= 2 * tolerance)), "face_point_counts": face_counts,
            "fit_scope": "observed surface bounding box; not a minimum-volume or watertight reconstruction"}


def build_rigid_model(observations, *, frame, primitive, table_plane=None,
                      axis_hint=None, axes_hint=None, endpoint_observations=None,
                      voxel_size_m=0.001, fit_tolerance_m=0.003):
    """Bounded cylinder/box fit; returns a partial model when coverage is weak.

    Cylinder observations should isolate the sidewall. Endpoint evidence is a
    separate explicit observation, not the bottom of a table-removal band.
    """
    if primitive not in {"cylinder", "box"} or not np.isfinite(fit_tolerance_m) or fit_tolerance_m <= 0:
        raise ModelInputError("supported primitives are cylinder/box with positive tolerance")
    merged = merge_known_views(observations, frame=frame, voxel_size_m=voxel_size_m)
    points = merged["points_m"]
    if table_plane is not None:
        points = separate_table(points, table_plane, frame=frame)["points_m"]
    points = _points(points, minimum=30)
    geometry = _fit_cylinder(points, axis_hint, fit_tolerance_m) if primitive == "cylinder" else _fit_box(points, axes_hint, fit_tolerance_m)
    issues = []
    if not merged["view_diversity"]:
        issues.append("independent_viewpoints_missing")
    if geometry["rms_m"] > 2 * fit_tolerance_m or geometry["inlier_fraction"] < 0.8:
        issues.append("primitive_fit_poor")
    if primitive == "cylinder":
        if geometry["angular_bin_coverage"] < 0.7:
            issues.append("sidewall_coverage_incomplete")
        ends = endpoint_observations or {}
        if ends.get("both_observed") is not True or not ends.get("evidence"):
            issues.append("endpoints_not_independently_observed")
        else:
            merged["evidence"].extend(_evidence(ends["evidence"]))
            if ends.get("frame") != frame or "points_m" not in ends:
                raise ModelInputError("observed cylinder endpoints need two measured axis-center points in the model frame")
            ep = _array(ends["points_m"], (2,3), "endpoint centers")
            center, axis = np.array(geometry["center_m"]), np.array(geometry["axis"])
            delta = ep-center
            offsets = delta @ axis
            radial = np.linalg.norm(delta - offsets[:,None]*axis,axis=1)
            if np.max(radial) > 2*fit_tolerance_m or abs(offsets[1]-offsets[0]) < 4*fit_tolerance_m:
                raise ModelInputError("endpoint measurements disagree with the fitted cylinder axis")
            side_extents = np.array(geometry["axial_endpoints_m"]) @ axis
            measured_extents = np.sort(ep @ axis)
            if np.max(np.abs(side_extents-measured_extents)) > 4*fit_tolerance_m:
                issues.append("endpoint_and_sidewall_extents_disagree")
            geometry["center_m"] = (center+axis*np.mean(offsets)).tolist()
            geometry["height_m"] = float(abs(offsets[1]-offsets[0]))
            geometry["axial_endpoints_m"] = [ep[i].tolist() for i in np.argsort(offsets)]
    elif min(geometry["face_point_counts"]) < max(5, int(len(points) * 0.01)):
        issues.append("box_faces_incomplete")
    return {"status": "partial" if issues else "estimated", "frame": frame, "units": "m",
            "geometry": geometry, "quality": {"issues": issues, "point_count": len(points),
            "view_count": merged["view_count"], "fit_tolerance_m": fit_tolerance_m, "registration_uncertainty_not_included": True},
            "usable_for_planning": not issues, "evidence": list(dict.fromkeys(merged["evidence"])),
            "limitations": ["No inference of hidden contents, material or mass distribution.",
                            "Coverage and fit metrics do not certify free space."]}


def import_mose3_tracks(npz_path, *, alignment, target_mask, evidence, max_tracks=2048):
    """Import official MoSE3 predictions.npz; never infer a cloth physics model.

    Official schema source (checked 2026-10-19):
    https://github.com/mose3-tracker/MoSE3/blob/main/inference.py (run_clip).
    tracks are in the first camera frame; visibility is a logit, not probability.
    alignment must state a measured scale and T_world_first_camera in metres.
    """
    refs = _evidence(evidence)
    if alignment.get("status") != "verified" or alignment.get("units") != "m":
        raise ModelInputError("verified metric/frame alignment is required for monocular tracks")
    refs += _evidence(alignment.get("evidence"))
    frame = alignment.get("world_frame")
    if not isinstance(frame, str) or not frame:
        raise ModelInputError("alignment world_frame required")
    scale = float(alignment.get("scale_to_m", 0))
    error = float(alignment.get("holdout_rms_m", np.inf))
    limit = float(alignment.get("max_holdout_rms_m", 0.005))
    if not np.isfinite([scale, error, limit]).all() or scale <= 0 or error < 0 or limit <= 0 or error > limit:
        raise ModelInputError("metric alignment scale/independent residual is invalid")
    T = _transform(alignment["T_world_first_camera"])
    if not isinstance(max_tracks, int) or max_tracks < 1:
        raise ModelInputError("positive max_tracks required")
    path = Path(npz_path)
    with np.load(path, allow_pickle=False) as data:
        required = {"tracks", "visibility", "se3_quat", "se3_trans", "se3_valid", "camera_poses", "intrinsics", "query_idx"}
        if not required.issubset(data.files):
            raise ModelInputError("not the official MoSE3 predictions schema: " + str(sorted(required - set(data.files))))
        tracks = np.asarray(data["tracks"], dtype=float)
        if tracks.ndim != 4 or tracks.shape[-1] != 3 or min(tracks.shape[:3]) < 1:
            raise ModelInputError("MoSE3 tracks must have shape T,H,W,3")
        nt, h, w, _ = tracks.shape
        for name, shape in {"visibility": (nt,h,w), "se3_valid": (nt,h,w), "se3_quat": (nt,h,w,4),
                            "se3_trans": (nt,h,w,3), "camera_poses": (nt,4,4), "intrinsics": (3,3)}.items():
            if data[name].shape != shape:
                raise ModelInputError(f"MoSE3 {name} shape mismatch")
        query_array = np.asarray(data["query_idx"])
        if query_array.size != 1 or query_array.dtype.kind not in "iu":
            raise ModelInputError("MoSE3 query_idx must be an integer scalar")
        query = int(query_array.item())
        if not 0 <= query < nt:
            raise ModelInputError("invalid query_idx")
        visibility = np.asarray(data["visibility"], dtype=float)
        se3_valid_raw = np.asarray(data["se3_valid"])
        if se3_valid_raw.dtype != np.bool_:
            raise ModelInputError("MoSE3 se3_valid must be boolean")
        se3_valid = se3_valid_raw
        poses = np.asarray(data["camera_poses"],dtype=float)
        for pose in poses:
            _transform(pose)
        if not np.allclose(poses[0],np.eye(4),atol=1e-4):
            raise ModelInputError("official MoSE3 first-camera convention was not preserved")
        quat = np.asarray(data["se3_quat"],dtype=float)
        trans = np.asarray(data["se3_trans"],dtype=float)
        motion_finite = np.isfinite(quat).all(axis=-1) & np.isfinite(trans).all(axis=-1)
        unit_quat = np.isclose(np.linalg.norm(quat,axis=-1),1,atol=1e-3)
        if np.any(se3_valid & ~(motion_finite & unit_quat)):
            raise ModelInputError("valid MoSE3 SE3 must have finite translation and unit wxyz quaternion")
        mask = np.asarray(target_mask)
        if mask.shape != (h,w) or mask.dtype != np.bool_ or not mask.any():
            raise ModelInputError("nonempty boolean target mask in query-pixel coordinates required")
        indexes = np.argwhere(mask)
        indexes = indexes[np.linspace(0, len(indexes)-1, min(max_tracks,len(indexes)), dtype=int)]
        selected = tracks[:,indexes[:,0],indexes[:,1],:]
        visible = visibility[:,indexes[:,0],indexes[:,1]] > 0
        valid = np.isfinite(selected).all(axis=2) & visible & se3_valid[:,indexes[:,0],indexes[:,1]]
        if not valid[query].any():
            raise ModelInputError("no visible valid target tracks in query frame")
        world = selected * scale @ T[:3,:3].T + T[:3,3]
        # Invalid points become JSON null, not NaN or invented coordinates.
        world_json = [[point.tolist() if ok else None for point, ok in zip(row, flags)] for row,flags in zip(world,valid)]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"status": "surface_tracks_only", "frame": frame, "units": "m", "source_sha256": digest,
            "query_idx": query, "query_pixels_hw": indexes.tolist(), "tracks_m": world_json,
            "valid": valid.tolist(), "alignment_holdout_rms_m": error, "evidence": list(dict.fromkeys(refs)),
            "usable_for_dynamics": False, "usable_for_planning": False,
            "limitations": ["Tracks do not identify cloth thickness, stiffness, friction, mass or contact.",
                            "Occluded pixels have no valid geometry; no hidden surface is fabricated.",
                            "A deformable manipulation planner and material identification are still required."]}


def backproject_rgbd(depth, *, depth_scale_m, intrinsics, valid_mask, min_depth_m,
                     max_depth_m, frame, evidence, T_world_camera=None, world_frame=None):
    """Convert the target mask of a depth image to metric points.

    Intrinsics must belong to the depth pixels after the actual alignment mode;
    supplying colour intrinsics for unaligned depth is rejected by the caller's
    explicit profile_id/alignment contract. This routine does not invent a mask.
    """
    image = np.asarray(depth)
    if image.ndim != 2 or image.dtype.kind not in "uif":
        raise ModelInputError("depth must be a numeric HxW image")
    if not np.isfinite([depth_scale_m,min_depth_m,max_depth_m]).all() or depth_scale_m <= 0 or min_depth_m <= 0 or max_depth_m <= min_depth_m:
        raise ModelInputError("explicit physical depth scale/range required")
    if intrinsics.get("profile_id") is None or intrinsics.get("pixel_frame") != frame:
        raise ModelInputError("intrinsics must identify the actual depth pixel profile and frame")
    if intrinsics.get("width") != image.shape[1] or intrinsics.get("height") != image.shape[0]:
        raise ModelInputError("depth/intrinsic resolution mismatch")
    K = _array(intrinsics.get("K"),(3,3),"camera intrinsics")
    if K[0,0] <= 0 or K[1,1] <= 0 or not np.allclose(K[2],[0,0,1]) or abs(K[0,1]) > 1e-8 or abs(K[1,0]) > 1e-8:
        raise ModelInputError("rectified pinhole intrinsics required")
    if intrinsics.get("rectified") is not True:
        raise ModelInputError("distortion must be rectified before pinhole backprojection")
    mask = np.asarray(valid_mask)
    if mask.shape != image.shape or mask.dtype != np.bool_:
        raise ModelInputError("boolean target/depth mask with matching resolution required")
    z = image.astype(float) * depth_scale_m
    accepted = mask & np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    rows,cols = np.nonzero(accepted)
    if len(rows) == 0:
        raise ModelInputError("no valid depth in target mask")
    points = np.column_stack([(cols-K[0,2])*z[accepted]/K[0,0],
                              (rows-K[1,2])*z[accepted]/K[1,1],z[accepted]])
    result_frame = frame
    if T_world_camera is not None:
        if not world_frame:
            raise ModelInputError("world_frame required for transformed point cloud")
        T = _transform(T_world_camera)
        points = points @ T[:3,:3].T + T[:3,3]
        result_frame = world_frame
    return {"status":"measured_depth_points","units":"m","frame":result_frame,
            "points_m":points.tolist(),"pixel_rows_cols":np.column_stack([rows,cols]).tolist(),
            "valid_fraction_of_mask":float(accepted.sum()/max(mask.sum(),1)),
            "intrinsics_profile_id":intrinsics["profile_id"],"evidence":_evidence(evidence),
            "limitations":["Invalid and occluded depth is absent, not filled with geometry."]}


