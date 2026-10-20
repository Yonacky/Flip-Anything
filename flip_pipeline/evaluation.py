"""Recompute numeric model/property reports before accepting workflow claims.

This validates that a report is the result of the recorded computation. It cannot
prove a real sensor produced the input; source provenance remains a separate gate.
"""
from __future__ import annotations

import math
from numbers import Real
from pathlib import Path

from . import modeling, physics
from .storage import digest, evidence, inside, read, verify_evidence


class NumericReportError(ValueError):
    pass


def _same(expected, actual, path="result", *, atol=1e-8, rtol=1e-6):
    """Compare full JSON structure and finite numbers, never coerce bool to int."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        if type(expected) is not type(actual) or expected != actual:
            raise NumericReportError(f"numeric report differs at {path}")
    elif isinstance(expected, Real) and isinstance(actual, Real):
        if not math.isfinite(expected) or not math.isfinite(actual) or not math.isclose(expected, actual, rel_tol=rtol, abs_tol=atol):
            raise NumericReportError(f"numeric report differs at {path}")
    elif isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            raise NumericReportError(f"numeric report keys differ at {path}")
        for key in expected:
            _same(expected[key], actual[key], path+"."+str(key),atol=atol,rtol=rtol)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise NumericReportError(f"numeric report length differs at {path}")
        for index,(left,right) in enumerate(zip(expected,actual)):
            _same(left,right,path+f"[{index}]",atol=atol,rtol=rtol)
    elif type(expected) is not type(actual) or expected != actual:
        raise NumericReportError(f"numeric report differs at {path}")


def _normalized(root, path):
    if not isinstance(path,str) or not path.strip():
        raise NumericReportError("evidence paths must be nonempty strings")
    return inside(root,path).relative_to(Path(root).resolve()).as_posix()


def _check_nested_refs(root, value, named, location="report"):
    if isinstance(value,dict):
        for key,child in value.items():
            if key in {"evidence","evidence_refs"}:
                if not isinstance(child,list):
                    raise NumericReportError(f"{location}.{key} must be a list of evidence paths")
                for ref in child:
                    path=ref.get("path") if isinstance(ref,dict) else ref
                    name=_normalized(root,path)
                    if name not in named:
                        raise NumericReportError(f"unhashed nested evidence at {location}.{key}: {name}")
                    if isinstance(ref,dict) and ref.get("sha256") not in (None,named[name]["sha256"]):
                        raise NumericReportError(f"nested evidence digest mismatch: {name}")
            else:
                _check_nested_refs(root,child,named,location+"."+str(key))
    elif isinstance(value,list):
        for index,child in enumerate(value):
            _check_nested_refs(root,child,named,location+f"[{index}]")


def _load_pair(root, data, request_key, result_key, named):
    paths=[]
    for key in (request_key,result_key):
        if key not in data:
            raise NumericReportError("missing numeric binding: "+key)
        name=_normalized(root,data[key])
        if name not in named:
            raise NumericReportError("numeric input/result file is not hashed evidence: "+name)
        paths.append(inside(root,name))
    if paths[0] == paths[1]:
        raise NumericReportError("numeric request and result must be separate immutable files")
    request,result=(read(path) for path in paths)
    _check_nested_refs(root,request,named,request_key)
    _check_nested_refs(root,result,named,result_key)
    return request,result


def validate_numeric_report(root, stage, data, evidence_paths):
    """Recompute model/properties reports; raise ValueError on any mismatch.

    evidence_paths may be canonical {path,sha256} refs or source path strings.
    All request/result JSONs and every nested evidence reference must be listed.
    This function does not replace the workflow's identity/installation binding.
    """
    if stage not in {"model","properties"}:
        raise NumericReportError("numeric report stage must be model or properties")
    if not isinstance(data,dict):
        raise NumericReportError("report must be a JSON object")
    provided=list(evidence_paths)
    if not provided:
        raise NumericReportError("hashed numeric evidence required")
    if all(isinstance(x,dict) for x in provided):
        refs=provided
    elif all(isinstance(x,str) for x in provided):
        refs=evidence(root,provided)
    else:
        raise NumericReportError("evidence must be paths or hashed refs, not a mixed list")
    verify_evidence(root,refs)
    named={_normalized(root,ref["path"]):ref for ref in refs}
    if len(named)!=len(refs):
        raise NumericReportError("duplicate canonical evidence paths")
    _check_nested_refs(root,data,named)
    if stage=="model":
        request,recorded=_load_pair(root,data,"model_request_path","model_result_path",named)
        if not isinstance(request,dict):
            raise NumericReportError("model request must be build_rigid_model keyword arguments")
        computed=modeling.build_rigid_model(**request)
        _same(computed,recorded)
        if computed.get("status")!="estimated" or computed.get("usable_for_planning") is not True:
            raise NumericReportError("model lacks measured coverage/quality for planning")
        checked={"model_result_digest":digest(computed),"model_kind":computed["geometry"]["kind"],
                 "frame":computed["frame"],"model_status":computed["status"]}
    else:
        request,recorded=_load_pair(root,data,"property_observations_path","property_result_path",named)
        if not isinstance(request,list):
            raise NumericReportError("property_observations_path must contain the observation list")
        properties=physics.assess_properties(request)
        _same(properties,recorded)
        if properties["requires_teacher_or_human"] or not properties["safe_for_exploration"]:
            raise NumericReportError("content/material risks are unresolved or present")
        if properties["properties"]["material"]["status"]!="established":
            raise NumericReportError("material screen still contains only a prior or unknown")
        if data.get("mass_status")!="measured":
            raise NumericReportError("bounded or assumed mass is not supported by the current execution gate")
        mass_request,mass_recorded=_load_pair(root,data,"mass_request_path","mass_result_path",named)
        if not isinstance(mass_request,dict):
            raise NumericReportError("mass request must contain keyword arguments")
        methods={"static-payload":physics.estimate_static_payload,
                 "weighed-components":physics.combine_weighed_components,
                 "combine-projections":physics.combine_support_projections}
        method=data.get("mass_method")
        if method not in methods:
            raise NumericReportError("unsupported mass_method; no uniform-density or bounded-physics surrogate")
        mass=methods[method](**mass_request)
        _same(mass,mass_recorded)
        if mass.get("status")!="measured" or mass.get("com_m") is None:
            raise NumericReportError("a validated measured 3D mass distribution is required")
        if method=="static-payload" and mass.get("target")!="object_difference":
            raise NumericReportError("complete tool payload is not the object's mass/CoM")
        checked={"property_result_digest":digest(properties),"mass_result_digest":digest(mass),
                 "mass_method":method,"mass_status":mass["status"],"frame":mass["frame"],
                 "mass_kg":mass["mass_kg"],"com_m":mass["com_m"]}
    # If any raw file changed during fitting, reject instead of binding two versions.
    verify_evidence(root,refs)
    return {"valid":True,"stage":stage,"method":"recomputed_from_hashed_numeric_inputs",**checked,
            "limitation":"Checks the recorded computation, not physical truth or sensor provenance."}
