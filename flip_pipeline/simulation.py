"""MuJoCo rigid-placement experiment with explicit frames and inertial inputs.

Conditional numerical experiment, never a COM measurement or robot path validator.
"""
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .calibration import validate_transform


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"{name} outside allowed range")
    return float(value)


def _array(value, shape, name):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return a


def _inertia(spec):
    a = np.asarray(spec.get("inertia_kg_m2"), dtype=float)
    frame = spec.get("inertia_frame")
    if a.shape == (3,) and frame == "object_com_principal_axes":
        tensor = np.diag(a)
        field = {"diaginertia": " ".join(str(float(x)) for x in a)}
    elif a.shape == (3, 3) and frame == "object_com_axes":
        tensor = a
        field = {"fullinertia": " ".join(str(float(a[i, j])) for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)))}
    else:
        raise ValueError("inertia requires explicit COM frame: aligned diagonal or full object-frame tensor")
    if not np.isfinite(tensor).all() or not np.allclose(tensor, tensor.T, rtol=0, atol=1e-12):
        raise ValueError("inertia tensor must be finite and symmetric")
    eigenvalues = np.linalg.eigvalsh(tensor)
    if min(eigenvalues) <= 0 or 2 * max(eigenvalues) > sum(eigenvalues) + 1e-12:
        raise ValueError("inertia must be positive and satisfy rigid-body triangle inequalities")
    return field


def _lowest_surface_z(kind, size, T):
    # Exact primitive support function along the table-frame Z axis.
    direction = T[2, :3]
    if kind == "box":
        extent = float(np.sum(abs(direction) * size))
    else:
        radius, half_height = size
        extent = float(radius * np.linalg.norm(direction[:2]) + half_height * abs(direction[2]))
    return float(T[2, 3] - extent)


def _check_dynamics(data, warning_baseline):
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or not np.isfinite(data.qacc).all():
        raise RuntimeError("Nonfinite simulation state")
    current = np.array([int(item.number) for item in data.warning])
    if np.any(current > warning_baseline):
        codes = np.flatnonzero(current > warning_baseline).tolist()
        raise RuntimeError(f"MuJoCo emitted numerical/capacity warnings: {codes}")


def simulate_placement(spec):
    import mujoco
    kind = spec.get("kind")
    if kind not in ("box", "cylinder"):
        raise ValueError("Initial simulator supports rigid boxes/cylinders")
    if spec.get("contents") not in ("empty", "known_rigid"):
        raise ValueError("Unknown/moving contents cannot be simulated as a known rigid body")
    if spec.get("mass_source") not in ("measured", "hypothesis"):
        raise ValueError("Declare measured or hypothesis")
    if spec.get("units") != "m_kg_s" or spec.get("world_frame") != "table_z0":
        raise ValueError("Simulation requires SI units m_kg_s and explicit table_z0 frame")
    if "world_from_object" in spec:
        raise ValueError("Use base_from_object and measured table_from_base; ambiguous legacy world pose is rejected")
    table_from_base = validate_transform(spec.get("table_from_base"), "table_from_base")
    base_from_object = validate_transform(spec.get("base_from_object"), "base_from_object")
    T = table_from_base @ base_from_object
    gravity_base = _array(spec.get("gravity_base_m_s2"), (3,), "gravity_base_m_s2")
    gravity = table_from_base[:3, :3] @ gravity_base
    if np.linalg.norm(gravity) < 1e-8 or gravity[2] >= -1e-8:
        raise ValueError("gravity must act toward the table support plane")
    mass = _number(spec.get("mass_kg"), "mass_kg", positive=True)
    com = _array(spec.get("com_m"), (3,), "com_m")
    size = _array(spec.get("size_m"), (3,) if kind == "box" else (2,), "size_m")
    if np.any(size <= 0):
        raise ValueError("size is positive box half-extents or cylinder radius,half-height")
    if kind == "box" and np.any(abs(com) > size + 1e-12):
        raise ValueError("COM outside modeled box convex hull")
    if kind == "cylinder" and (np.linalg.norm(com[:2]) > size[0] + 1e-12 or abs(com[2]) > size[1] + 1e-12):
        raise ValueError("COM outside modeled cylinder convex hull")
    inertial_parameters = _inertia(spec)
    friction = _array(spec.get("friction"), (3,), "friction")
    if np.any(friction < 0):
        raise ValueError("friction coefficients must be nonnegative")
    duration = _number(spec.get("duration_s", 2), "duration_s", positive=True)
    dt = _number(spec.get("timestep_s", .002), "timestep_s", positive=True)
    if not 0 < dt <= .01 or not dt <= duration <= 20:
        raise ValueError("Invalid time horizon")
    evidence = spec.get("evidence")
    if not isinstance(evidence, list) or not evidence or any(not isinstance(x, str) or not x.strip() for x in evidence):
        raise ValueError("Record source evidence or named hypothesis")
    penetration_limit = _number(spec.get("max_initial_penetration_m", 0.0001), "max_initial_penetration_m", nonnegative=True)
    if penetration_limit > .001:
        raise ValueError("Initial penetration allowance cannot exceed 1 mm")
    lowest = _lowest_surface_z(kind, size, T)
    if lowest < -penetration_limit:
        raise ValueError("Initial object penetrates table; correct geometry/pose before simulation")

    s = lambda vector: " ".join(str(float(x)) for x in vector)
    xml = ET.Element("mujoco", model="conditional_placement")
    ET.SubElement(xml, "option", timestep=str(dt), gravity=s(gravity))
    w = ET.SubElement(xml, "worldbody")
    ET.SubElement(w, "geom", name="table", type="plane", size="2 2 .01", friction=s(friction))
    xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    body = ET.SubElement(w, "body", name="object", pos=s(T[:3, 3]), quat=s(xyzw[[3, 0, 1, 2]]))
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "inertial", pos=s(com), mass=str(mass), **inertial_parameters)
    ET.SubElement(body, "geom", name="shape", type=kind, size=s(size), friction=s(friction))
    xmltext = ET.tostring(xml, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xmltext)
    data = mujoco.MjData(model)
    baseline = np.array([int(item.number) for item in data.warning])
    mujoco.mj_forward(model, data)
    _check_dynamics(data, baseline)
    initial = data.xpos[1].copy()
    R0 = data.xmat[1].reshape(3, 3).copy()
    peak_linear_speed, peak_angular_speed = 0.0, 0.0
    for _ in range(int(math.ceil(duration / dt))):
        mujoco.mj_step(model, data)
        _check_dynamics(data, baseline)
        peak_linear_speed = max(peak_linear_speed, float(np.linalg.norm(data.qvel[:3])))
        peak_angular_speed = max(peak_angular_speed, float(np.linalg.norm(data.qvel[3:6])))
    mujoco.mj_forward(model, data)
    _check_dynamics(data, baseline)
    final = np.eye(4)
    final[:3, :3] = data.xmat[1].reshape(3, 3)
    final[:3, 3] = data.xpos[1]
    angle = float(Rotation.from_matrix(R0.T @ final[:3, :3]).magnitude())
    return {"status": "conditional_simulation", "engine": "MuJoCo", "engine_version": mujoco.__version__,
            "mass_source": spec["mass_source"], "simulation_frame": "table_z0", "gravity_table_m_s2": gravity.tolist(),
            "initial_table_from_object": T.tolist(), "initial_minimum_clearance_m": lowest,
            "pose_displacement_m": float(np.linalg.norm(final[:3, 3] - initial)),
            "orientation_change_rad": angle,
            "final_linear_speed_m_s": float(np.linalg.norm(data.qvel[:3])),
            "final_angular_speed_rad_s": float(np.linalg.norm(data.qvel[3:6])),
            "peak_linear_speed_m_s": peak_linear_speed, "peak_angular_speed_rad_s": peak_angular_speed,
            "contacts": int(data.ncon), "numerical_warnings": [],
            "table_from_object_final": final.tolist(),
            "base_from_object_final": (np.linalg.inv(table_from_base) @ final).tolist(),
            "model_xml": xmltext, "inputs": spec,
            "limits": "Rigid placement on an infinite plane under supplied inertia/friction. Not measured COM, grasp proof, full-scene/robot collision validation, or physical success."}
