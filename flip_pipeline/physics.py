"""Simple, auditable static mass/CoM estimates and property evidence.

No simulator can discover density or contents from a surface mesh alone. These
functions need actual measured forces/weights; an xArm pose or motor-current
readout is not a calibrated six-axis wrench. All values are SI.
"""
from __future__ import annotations

import numpy as np


class IdentificationError(ValueError):
    pass


def _vec(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise IdentificationError(f"{name} must be a finite 3-vector")
    return result


def _refs(value):
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x.strip() for x in value):
        raise IdentificationError("nonempty evidence references required")
    return list(value)


def _skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def _prepare(samples, calibration):
    gravities, forces, torques, evidence = [], [], [], []
    if not isinstance(samples, list) or not samples:
        raise IdentificationError("static measurements required")
    identities = set()
    for sample in samples:
        if sample.get("static") is not True or sample.get("no_external_contact") is not True:
            raise IdentificationError("only settled poses without contact can identify static payload")
        if sample.get("frame") != calibration["frame"]:
            raise IdentificationError("wrench/gravity frame mismatch")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in identities:
            raise IdentificationError("unique nonempty sample_id required")
        identities.add(sample_id)
        g = _vec(sample.get("gravity_m_s2"), "gravity_m_s2")
        if not 9.0 <= np.linalg.norm(g) <= 10.5:
            raise IdentificationError("gravity must be physical acceleration in the sensor frame")
        force = _vec(sample.get("force_N"), "force_N")
        torque = _vec(sample.get("torque_Nm"), "torque_Nm")
        if calibration["target"] == "object_difference":
            baseline = sample.get("unloaded")
            if not isinstance(baseline, dict) or baseline.get("frame") != calibration["frame"]:
                raise IdentificationError("object estimate needs paired unloaded measurements in the same frame")
            if baseline.get("static") is not True or baseline.get("no_external_contact") is not True:
                raise IdentificationError("unloaded baseline must be settled and contact-free")
            if not np.allclose(_vec(baseline["gravity_m_s2"], "baseline gravity"), g, atol=1e-4):
                raise IdentificationError("loaded/unloaded gravity directions do not match")
            force -= _vec(baseline["force_N"], "unloaded force")
            torque -= _vec(baseline["torque_Nm"], "unloaded torque")
            evidence += _refs(baseline.get("evidence"))
        gravities.append(g)
        forces.append(force)
        torques.append(torque)
        evidence += _refs(sample.get("evidence"))
    return np.array(gravities), np.array(forces), np.array(torques), evidence, identities


def _solve(A, b, condition_limit, name):
    singular = np.linalg.svd(A, compute_uv=False)
    rank = int(np.linalg.matrix_rank(A))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    if rank != A.shape[1] or condition > condition_limit:
        raise IdentificationError(f"{name} underexcited: use genuinely different gravity directions; rank={rank}")
    params = np.linalg.lstsq(A, b, rcond=None)[0]
    dof = max(len(b) - A.shape[1], 1)
    variance = float(np.sum((A @ params - b) ** 2) / dof)
    covariance = variance * np.linalg.inv(A.T @ A)
    return params, covariance, condition


def estimate_static_payload(samples, sensor_calibration, *, holdout_samples=None,
                            bias_mode="estimate", max_condition=1000,
                            max_force_rms_N=0.05, max_torque_rms_Nm=0.005,
                            minimum_mass_kg=0.005):
    """Linear static wrench identification, including optional constant biases.

    f = m*g + b_f; tau = h cross g + b_tau; h=m*CoM.
    Target is either everything beyond the sensor, or an object isolated by
    paired empty-tool measurements. Independent holdout poses are required for
    status='measured'. Identifying a fixed CoM never proves contents are solid.
    """
    cal = sensor_calibration
    if cal.get("status") != "verified" or cal.get("source") not in {"calibrated_ft_sensor", "calibrated_support_loadcells"}:
        raise IdentificationError("a verified calibrated force/torque source is required; joint currents are not a wrench")
    if not isinstance(cal.get("frame"), str) or not cal["frame"]:
        raise IdentificationError("explicit sensor frame required")
    if cal.get("force_units") != "N" or cal.get("torque_units") != "N*m":
        raise IdentificationError("force N and torque N*m are required")
    if cal.get("gravity_compensated") is not False:
        raise IdentificationError("gravity-compensated wrench cannot identify gravitational payload")
    if cal.get("target") not in {"complete_payload", "object_difference", "object_on_supports"}:
        raise IdentificationError("declare complete_payload, object_difference or object_on_supports")
    if cal.get("target") == "object_on_supports" and cal.get("source") != "calibrated_support_loadcells":
        raise IdentificationError("support target requires calibrated support load cells")
    if cal.get("mass_distribution_fixed") is not True:
        raise IdentificationError("static rigid distribution not established; no single fixed CoM can be claimed")
    evidence = _refs(cal.get("evidence"))
    if bias_mode not in {"estimate", "known"}:
        raise IdentificationError("bias_mode must be estimate or known")
    limits = [max_condition, max_force_rms_N, max_torque_rms_Nm, minimum_mass_kg]
    if not np.isfinite(limits).all() or min(limits) <= 0:
        raise IdentificationError("quality limits must be positive finite numbers")
    gravity, force, torque, sample_refs, train_ids = _prepare(samples, cal)
    evidence += sample_refs
    if len(samples) < (6 if bias_mode == "estimate" else 3):
        raise IdentificationError("too few independent static poses")
    if bias_mode == "estimate":
        force_A = np.vstack([np.column_stack([g, np.eye(3)]) for g in gravity])
        torque_A = np.vstack([np.column_stack([-_skew(g), np.eye(3)]) for g in gravity])
        force_y, torque_y = force.ravel(), torque.ravel()
    else:
        if cal.get("bias_verified") is not True:
            raise IdentificationError("known bias mode requires verified force/torque biases")
        bf = _vec(cal.get("force_bias_N"), "force_bias_N")
        bt = _vec(cal.get("torque_bias_Nm"), "torque_bias_Nm")
        force_A = gravity.reshape(-1, 1)
        torque_A = np.vstack([-_skew(g) for g in gravity])
        force_y, torque_y = (force-bf).ravel(), (torque-bt).ravel()
    pf, cov_f, cond_f = _solve(force_A, force_y, max_condition, "force")
    pt, cov_t, cond_t = _solve(torque_A, torque_y, max_condition, "torque")
    mass, moment = float(pf[0]), pt[:3]
    if mass < minimum_mass_kg:
        raise IdentificationError("mass is nonphysical or below configured measurement capability")
    if bias_mode == "estimate":
        bf, bt = pf[1:], pt[3:]
    com = moment / mass
    predicted_force = mass * gravity + bf
    predicted_torque = np.cross(moment, gravity) + bt
    force_rms = float(np.sqrt(np.mean((force - predicted_force) ** 2)))
    torque_rms = float(np.sqrt(np.mean((torque - predicted_torque) ** 2)))
    if force_rms > max_force_rms_N or torque_rms > max_torque_rms_Nm:
        raise IdentificationError("static model residual exceeds limits; inspect contact, slosh, calibration or noise")
    # Propagates independent force/torque LS covariance; systematic error remains outside this bound.
    com_cov = cov_t[:3,:3] / mass**2 + np.outer(moment,moment) * cov_f[0,0] / mass**4
    validation = {"independent_holdout": False, "force_rms_N": None, "torque_rms_Nm": None}
    status = "provisional"
    if holdout_samples:
        hg, hf, ht, hrefs, holdout_ids = _prepare(holdout_samples, cal)
        if train_ids & holdout_ids:
            raise IdentificationError("holdout measurements must not reuse training sample ids")
        if len(hg) < 2 or np.linalg.norm(np.cross(hg[0], hg[1])) < 1:
            raise IdentificationError("at least two diverse independent holdout poses required")
        # Repeated training poses are not independent coverage of the solved model.
        if any(np.any(np.linalg.norm(gravity - g, axis=1) < 1e-3) for g in hg):
            raise IdentificationError("holdout gravity directions repeat identification poses")
        hfr = float(np.sqrt(np.mean((hf - (mass*hg + bf))**2)))
        htr = float(np.sqrt(np.mean((ht - (np.cross(moment,hg) + bt))**2)))
        if hfr > max_force_rms_N or htr > max_torque_rms_Nm:
            raise IdentificationError("independent holdout validation failed")
        validation = {"independent_holdout": True, "force_rms_N": hfr, "torque_rms_Nm": htr}
        evidence += hrefs
        status = "measured"
    return {"status": status, "method": "static_gravity_linear_least_squares", "frame": cal["frame"],
            "target": cal["target"], "mass_kg": mass, "com_m": com.tolist(),
            "force_bias_N": bf.tolist(), "torque_bias_Nm": bt.tolist(),
            "mass_std_kg": float(np.sqrt(max(cov_f[0,0],0))),
            "com_std_m": np.sqrt(np.maximum(np.diag(com_cov),0)).tolist(),
            "fit": {"force_rms_N": force_rms, "torque_rms_Nm": torque_rms,
                    "force_condition": cond_f, "torque_condition": cond_t, "sample_count": len(samples)},
            "validation": validation, "evidence": list(dict.fromkeys(evidence)),
            "limitations": ["Covariance covers residual random error, not sensor or transform systematics.",
                            "Does not establish material, contents, inertia tensor or absence of liquid."]}


def combine_weighed_components(components, *, frame):
    """Mass-weighted sum when each component mass AND local CoM are measured."""
    if not components or not isinstance(frame,str) or not frame:
        raise IdentificationError("components and common frame required")
    mass, moment, evidence = 0.0, np.zeros(3), []
    for component in components:
        if component.get("frame") != frame or component.get("status") != "measured":
            raise IdentificationError("component centers and masses must be measured in the same frame")
        m = float(component.get("mass_kg", 0))
        if not np.isfinite(m) or m <= 0:
            raise IdentificationError("positive measured component mass required")
        mass += m
        moment += m * _vec(component.get("com_m"), "component CoM")
        evidence += _refs(component.get("evidence"))
    return {"status": "measured", "method": "weighed_component_sum", "frame": frame,
            "mass_kg": mass, "com_m": (moment/mass).tolist(), "evidence": list(dict.fromkeys(evidence)),
            "limitations": ["Valid only while component masses and relative positions remain fixed."]}


def estimate_support_projection(supports, *, frame, gravity_m_s2, loadcell_calibration):
    """Simple static weighing: return only CoM projection perpendicular to gravity.

    A single support arrangement cannot identify height along gravity. Contact
    positions and calibrated support forces must be expressed in the same frame.
    The object is stationary, with gravity and measured supports as its only loads.
    """
    if not isinstance(frame,str) or not frame:
        raise IdentificationError("explicit frame required")
    cal = loadcell_calibration
    if cal.get("status") != "verified" or cal.get("external_loads_only_gravity_and_supports") is not True:
        raise IdentificationError("verified static support measurement conditions required")
    evidence = _refs(cal.get("evidence"))
    if len(supports) < 2:
        raise IdentificationError("at least two measured support contacts required")
    g = _vec(gravity_m_s2, "gravity")
    gnorm = np.linalg.norm(g)
    if not 9 <= gnorm <= 10.5:
        raise IdentificationError("gravity in m/s2 required")
    resultant, moment = np.zeros(3), np.zeros(3)
    for support in supports:
        if support.get("frame") != frame:
            raise IdentificationError("support frame mismatch")
        f = _vec(support.get("force_N"), "support force")
        r = _vec(support.get("position_m"), "support position")
        if np.dot(f,g) >= 0 or np.linalg.norm(np.cross(f,g)) > max(np.linalg.norm(f)*gnorm*0.01,1e-6):
            raise IdentificationError("simple support method requires upward forces parallel to gravity")
        resultant += f
        moment += np.cross(r,f)
        evidence += _refs(support.get("evidence"))
    mass = -float(np.dot(resultant,g) / np.dot(g,g))
    if mass <= 0:
        raise IdentificationError("nonpositive inferred mass")
    # tau_support + r_com x (m*g) = 0. Only the projection is observable.
    projected_com = -np.cross(g,moment) / (mass*np.dot(g,g))
    return {"status": "partial", "method": "static_support_weighing", "frame": frame,
            "mass_kg": mass, "com_projection_m": projected_com.tolist(),
            "unobserved_axis": (g/gnorm).tolist(), "com_m": None,
            "evidence": list(dict.fromkeys(evidence)),
            "limitations": ["CoM height along gravity is not measured; no complete 3D CoM is claimed.",
                            "Additional independently oriented weighing measurements are needed for 3D CoM."]}


def combine_support_projections(projections, *, frame, max_residual_m=0.003, max_mass_difference_fraction=0.02):
    """Recover 3D CoM from >=2 nonparallel projected-CoM weighing experiments.

    All projections must already be transformed into the SAME rigid object frame,
    including their gravity/null axes. The object and contents must not shift.
    """
    if not np.isfinite([max_residual_m,max_mass_difference_fraction]).all() or min(max_residual_m,max_mass_difference_fraction) <= 0:
        raise IdentificationError("support consistency limits must be positive finite numbers")
    if len(projections) < 2 or not frame:
        raise IdentificationError("two or more object-frame projections required")
    A, b, masses, evidence = [], [], [], []
    for p in projections:
        if p.get("frame") != frame or p.get("method") != "static_support_weighing" or p.get("status") != "partial":
            raise IdentificationError("expected static support projections in one object frame")
        axis = _vec(p["unobserved_axis"], "unobserved axis")
        if not np.isclose(np.linalg.norm(axis),1,atol=1e-5):
            raise IdentificationError("unobserved axis must be unit length")
        matrix = np.eye(3) - np.outer(axis,axis)
        A.append(matrix)
        b.append(_vec(p["com_projection_m"], "projected CoM"))
        masses.append(float(p["mass_kg"]))
        evidence += _refs(p.get("evidence"))
    if min(masses) <= 0 or not np.isfinite(masses).all():
        raise IdentificationError("invalid measured mass")
    mass = float(np.mean(masses))
    if (max(masses)-min(masses))/mass > max_mass_difference_fraction:
        raise IdentificationError("inconsistent weights across support experiments")
    A,b = np.vstack(A),np.concatenate(b)
    com,_,condition = _solve(A,b,100,"support projection")
    rms = float(np.sqrt(np.mean((A@com-b)**2)))
    if not np.isfinite(max_residual_m) or max_residual_m <= 0 or rms > max_residual_m:
        raise IdentificationError("support projection consistency check failed")
    return {"status":"measured", "method":"multiple_static_support_projections", "frame":frame,
            "mass_kg":mass,"com_m":com.tolist(),"fit":{"rms_m":rms,"condition":condition},
            "evidence":list(dict.fromkeys(evidence)),
            "limitations":["Requires rigid unchanged mass distribution and independently measured object-frame alignment.",
                           "Residual consistency does not replace scale calibration or exclude systematic error."]}


def assess_properties(observations):
    """Keep priors separate from established properties; unresolved hazards escalate.

    Allowed fields include material, liquid_present, fragile, hazardous_contents,
    contents_mobile. Visual appearance can suggest material but never prove an
    opaque container empty. Every conclusion retains its evidence and source.
    """
    allowed = {"material", "liquid_present", "fragile", "hazardous_contents", "contents_mobile"}
    by_field = {field: [] for field in allowed}
    for obs in observations:
        field, source = obs.get("field"), obs.get("source")
        if field not in allowed or source not in {"measurement", "human_attestation", "visual_prior", "assumption"}:
            raise IdentificationError("unknown property/source")
        _refs(obs.get("evidence"))
        if field != "material" and not isinstance(obs.get("value"),bool):
            raise IdentificationError("hazard values must be boolean; missing is unknown")
        if field == "material" and (not isinstance(obs.get("value"),str) or not obs["value"].strip()):
            raise IdentificationError("material must be a nonempty description")
        by_field[field].append(dict(obs))
    properties, unresolved, hazards = {}, [], []
    for field, claims in by_field.items():
        established = [x for x in claims if x["source"] in {"measurement","human_attestation"}]
        values = {str(x["value"]) for x in established}
        if len(values) == 1:
            properties[field] = {"status":"established", "value":established[-1]["value"], "claims":claims}
        elif len(values) > 1:
            properties[field] = {"status":"conflict", "value":None, "claims":claims}
        else:
            properties[field] = {"status":"unknown", "value":None, "claims":claims}
        if field != "material":
            if properties[field]["status"] != "established":
                unresolved.append(field)
            elif properties[field]["value"] is True:
                hazards.append(field)
    return {"properties":properties, "requires_teacher_or_human":bool(unresolved or hazards),
            "unresolved_hazards":sorted(unresolved), "identified_hazards":sorted(hazards),
            "safe_for_exploration":not (unresolved or hazards),
            "scope":"Property screen only; geometry, force limits and human exclusion still require their own checks."}

