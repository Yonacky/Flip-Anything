"""Ordered evidence gates. A fault rolls back logical work, never reverses the robot."""
from pathlib import Path
from copy import deepcopy
import math, uuid
from .storage import read,write,digest,identity,utcnow,evidence,verify_evidence,inside,locked
from .contracts import load_contract, verify_contract, required_checks, task_directed, validate_scope
STAGES=("calibration","model","properties","plan","execution","review")
REQUIRED={
 "calibration":("installation_verified","hand_eye_heldout","intrinsics_depth_verified","tcp_tool_envelope_verified","table_workspace_verified"),
 "model":("metric_frame","multi_view_fit","coverage_for_grasp_and_support"),
 "properties":("content_screen","material_screen","mass_distribution"),
 "plan":("experience_search","support_goal","grasp_and_clearance","path_validation"),
 "execution":("inverted","supported_before_release","released","independently_stable"),
 "review":("raw_materials_complete","outcome_and_interventions_recorded")}
ALLOWED_METHODS={
 "scope_membership":{"operator_scope_and_observation"},
 "installation_verified":{"installation_inspection"},"hand_eye_heldout":{"hand_eye_validation"},
 "intrinsics_depth_verified":{"factory_intrinsics_depth_check"},"tcp_tool_envelope_verified":{"measured_tool_geometry"},
 "table_workspace_verified":{"rgbd_table_workspace"},"metric_frame":{"metric_registration"},
 "multi_view_fit":{"primitive_fit","open3d_registration","metric_mose3_surface"},
 "coverage_for_grasp_and_support":{"coverage_review"},"content_screen":{"visual_and_contact","known_empty_inspection","teacher_inspection"},
 "material_screen":{"visual_and_contact","teacher_inspection"},"mass_distribution":{"weighed_components","static_wrench","bounded_mass_hypothesis"},
 "experience_search":{"released_catalog_query"},"support_goal":{"support_geometry"},
 "grasp_and_clearance":{"tool_object_envelope"},"path_validation":{"measured_joint_path","teacher_validated_segment"},
 "inverted":{"object_pose_observation"},"supported_before_release":{"observed_contact","calibrated_contact_measurement"},
 "released":{"gripper_receipt"},"independently_stable":{"post_retreat_observation"},
 "raw_materials_complete":{"manifest_verification"},"outcome_and_interventions_recorded":{"trace_review"}}

def new_task(root,object_id,installation_id,mode="experiment",profile=None):
    if mode not in ("experiment","calibration"):raise ValueError("Invalid task mode")
    contract=load_contract(root,profile) if profile else None
    if contract and mode!="experiment":raise ValueError("Task profile is only for object experiments")
    folder=Path(root)/"logs/pipeline"/("TASK_"+uuid.uuid4().hex[:12]); folder.mkdir(parents=True,exist_ok=False)
    state={"schema_version":1,"task_id":folder.name,"object_id":identity(object_id),"installation_id":identity(installation_id),
           "mode":mode,"stage":"calibration","status":"active","artifacts":{},"history":[],"created_at":utcnow(),"inflight":None}
    if contract:state.update(task_contract=contract,contract_sha256=digest(contract))
    write(folder/"state.json",state);return folder

def load_task(root,task):
    p=inside(root,task);return p,read(p/"state.json")

def validate_report(stage,data,refs,state=None):
    if stage not in STAGES:raise ValueError("Unknown stage")
    claims=data.get("checks",{}); missing=[]; evidence_paths={r["path"] for r in refs}
    for key in required_checks(stage,state,REQUIRED):
        c=claims.get(key,{})
        if c.get("result")!="pass" or c.get("method") not in ALLOWED_METHODS[key]: missing.append(key);continue
        if not c.get("evidence") or not set(c["evidence"]).issubset(evidence_paths):missing.append(key+":evidence")
        if not isinstance(c.get("detail"),str) or not c["detail"].strip():missing.append(key+":detail")
    if data.get("source")!="physical_evidence":missing.append("physical_evidence_required")
    if stage=="properties" and not task_directed(state):
        hazards=data.get("hazards",{})
        for key in ("spill","breakage","sharp_or_hot"):
            if hazards.get(key)!="excluded":missing.append("hazard:"+key)
        mass=data.get("mass_status")
        if mass not in ("measured","bounded_hypothesis"):missing.append("mass_status")
        if mass=="bounded_hypothesis" and not data.get("robust_parameter_bounds"):
            missing.append("explicit_mass_bounds")
    if stage=="plan":
        if data.get("execution_path") not in ("imitate_adapt","bounded_trial","teacher_demonstration"):missing.append("execution_path")
        if data.get("path_scope")!="robot_tool_object":missing.append("whole_path_scope")
        if not data.get("segments"):missing.append("segments")
        ids=[x.get("id") for x in data.get("segments",[])]
        if len(set(ids))!=len(ids) or any(not x for x in ids):missing.append("unique_segments")
    if missing:raise ValueError("Incomplete "+stage+": "+", ".join(missing))

def attach_report(root,task,stage,data,evidence_paths):
    folder,state=load_task(root,task)
    with locked(folder/"state.lock"):
        state=read(folder/"state.json")
        if state["inflight"]:raise ValueError("Cannot replace evidence while action is outstanding")
        if state["status"]!="active":raise ValueError("Task halted; maintainer must inspect before re-entry")
        if state["mode"]=="calibration" and stage!="calibration":raise ValueError("Calibration task does not execute an object trial")
        index=STAGES.index(stage)
        for earlier in STAGES[:index]:
            if earlier not in state["artifacts"]:raise ValueError("Missing previous stage: "+earlier)
        verify_task(root,folder,state)
        refs=evidence(root,evidence_paths);validate_report(stage,data,refs,state)
        if stage=="properties" and task_directed(state):
            validate_scope(root,data,refs,state)
        elif stage in ("model","properties"):
            from .evaluation import validate_numeric_report
            validate_numeric_report(root,stage,data,refs)
        if stage=="execution":
            from .completion import validate_completion
            validate_completion(root,folder,state,data,refs)
        if stage=="plan":
            from .plan_evaluation import validate_plan_report
            validate_plan_report(root,data,refs,state)
        if stage=="calibration":
            from .calibration import validate_device_profile
            profile_path=inside(root,data["profile_path"])
            installation_path=inside(root,data["installation_path"])
            named={r["path"] for r in refs}
            if any(p.relative_to(Path(root).resolve()).as_posix() not in named for p in (profile_path,installation_path)):raise ValueError("Profile and installation must be hashed evidence")
            profile=read(profile_path)
            review=profile.get("review",{})
            if review.get("basis")=="operator_accepted_measured_geometry":
                acceptance_path=review.get("operator_acceptance_ref")
                if acceptance_path not in named:raise ValueError("Operator acceptance must be hashed evidence")
                acceptance=read(inside(root,acceptance_path))
                if (acceptance.get("source")!="operator_declaration" or acceptance.get("accepted") is not True
                    or acceptance.get("installation_id")!=state["installation_id"]
                    or acceptance.get("task_profiles")!=review.get("task_profiles")
                    or acceptance.get("profile_sha256")!=review.get("profile_sha256")):
                    raise ValueError("Operator acceptance does not match calibration")
            result=validate_device_profile(profile,read(installation_path),task_profile=state.get("task_contract",{}).get("id"))
            if result["installation_id"]!=state["installation_id"]:raise ValueError("Wrong installation profile")
        deps={k:state["artifacts"][k] for k in STAGES[:index]}
        artifact={"schema_version":1,"kind":stage,"task_id":state["task_id"],"object_id":state["object_id"],
                  "installation_id":state["installation_id"],"dependencies":deps,"data":data,"evidence":refs,"created_at":utcnow()}
        h=digest(artifact);write(folder/"artifacts"/(h+".json"),artifact)
        state["artifacts"]={k:v for k,v in state["artifacts"].items() if STAGES.index(k)<index}
        state["artifacts"][stage]=h;state["stage"]=STAGES[min(index+1,len(STAGES)-1)]
        state["history"].append({"event":"accepted_report","stage":stage,"sha256":h,"time":utcnow()})
        if stage=="review":
            state["status"]="complete";state["outcome"]="success"
        if stage=="calibration" and state["mode"]=="calibration":
            state["status"]="complete";state["stage"]="calibration";state["outcome"]="calibrated"
        write(folder/"state.json",state);return artifact

def verify_task(root,folder,state=None):
    state=state or read(Path(folder)/"state.json")
    verify_contract(root,state)
    for stage,h in state["artifacts"].items():
        a=read(Path(folder)/"artifacts"/(h+".json"))
        if digest(a)!=h:raise ValueError("Artifact changed: "+stage)
        for key in ("task_id","object_id","installation_id"):
            if a[key]!=state[key]:raise ValueError("Identity mismatch: "+key)
        expected={k:state["artifacts"][k] for k in STAGES[:STAGES.index(stage)]}
        if a["dependencies"]!=expected:raise ValueError("Stale dependencies: "+stage)
        verify_evidence(root,a["evidence"]);validate_report(stage,a["data"],a["evidence"],state)
        if stage=="properties" and task_directed(state):validate_scope(root,a["data"],a["evidence"],state)
    if state.get("finalization"):
        h=state["finalization"];closed=read(Path(folder)/"finalizations"/(h+".json"))
        if digest(closed)!=h:raise ValueError("Finalization changed")
        for key in ("task_id","object_id","installation_id"):
            if closed[key]!=state[key]:raise ValueError("Finalization identity mismatch")
        if closed["dependencies"]!=state["artifacts"]:raise ValueError("Finalization dependencies changed")
        verify_evidence(root,closed["evidence"])
    return state

def status(root,task):
    folder,state=load_task(root,task);error=None
    try:verify_task(root,folder,state)
    except (ValueError,KeyError,OSError) as exc:error=str(exc)
    terminal=state["status"] in {"complete","closed"}
    if error:
        next_action="inspect_and_call_teacher"
    elif state.get("inflight"):
        next_action="await_human_action_resolution" if terminal or state["status"]=="halted" else "await_action_receipt"
    elif terminal:
        next_action="none"
    elif state["status"]=="halted":
        next_action="inspect_and_call_teacher"
    else:
        next_action="collect_"+state["stage"]+"_evidence"
    return {**state,"integrity_error":error,"required_checks":[] if terminal else list(required_checks(state["stage"],state,REQUIRED)),
            "next_action":next_action,"terminal":terminal}


def halt(root,task,reason,run_id=None):
    from .experience import teacher_request
    if not isinstance(reason,str) or not reason.strip():raise ValueError("Halt reason required")
    folder,_=load_task(root,task)
    with locked(folder/"state.lock"):
        state=read(folder/"state.json")
        if state["status"] in {"complete","closed"}:raise ValueError("Closed task cannot be restarted or halted again")
        state["status"]="halted"
        state["history"].append({"event":"halt","reason":reason,"run_id":run_id,"time":utcnow(),
                                 "rollback":"logical stage only; no reverse motion"})
        request=teacher_request(request_id="help-"+uuid.uuid4().hex[:12],teacher="human",
            missing_information=["failure cause", "current physical state", "safe next step"],
            question=reason,requested_segment="Inspect the interrupted stage before any further movement",
            state_snapshot={"task_id":state["task_id"],"object_id":state["object_id"],
                "installation_id":state["installation_id"],"stage":state["stage"],
                "inflight":state["inflight"],"run_id":run_id,"artifacts":state["artifacts"]},
            evidence_refs=["artifacts/"+h+".json" for h in state["artifacts"].values()],
            timestamp=utcnow(),adapter_connected=False)
        request["evidence_base"]="task_directory"
        request["physical_recovery"]=False
        write(folder/"teacher_request.json",request)
        write(folder/"state.json",state);return state


def finalize(root,task,outcome,evidence_paths,summary):
    """Seal materials at any stage. A failed/unknown run is never promoted to success.

    Closing with an outstanding action records its uncertainty, never clears the
    action or authorizes physical rollback. This is record keeping, not recovery.
    """
    if outcome not in {"success","failure","incomplete","unknown"}:raise ValueError("Invalid outcome")
    if not isinstance(summary,dict) or not isinstance(summary.get("text"),str) or not summary["text"].strip():
        raise ValueError("Summary text required")
    for key in ("interventions","failures","unresolved"):
        if not isinstance(summary.get(key),list):raise ValueError("Summary needs "+key+" list")
    from .experience import _reject_private
    _reject_private(summary)
    folder,_=load_task(root,task)
    with locked(folder/"state.lock"):
        state=read(folder/"state.json")
        if state.get("finalization"):raise ValueError("Materials already finalized; preserve the original and create a new task")
        integrity_error=None
        try:verify_task(root,folder,state)
        except (ValueError,KeyError,OSError) as exc:integrity_error=str(exc)
        if outcome=="success":
            required="calibration" if state["mode"]=="calibration" else "execution"
            if integrity_error or state["status"]=="halted" or state.get("inflight") or required not in state["artifacts"]:
                raise ValueError("Success requires accepted completion evidence and no unresolved action")
        refs=evidence(root,evidence_paths)
        report={"schema_version":1,"kind":"material_finalization","task_id":state["task_id"],
            "object_id":state["object_id"],"installation_id":state["installation_id"],
            "scope":"calibration" if state["mode"]=="calibration" else "object_flip",
            "outcome":outcome,"summary":deepcopy(summary),"evidence":refs,"dependencies":dict(state["artifacts"]),
            "prior_status":state["status"],"stopped_at_stage":state["stage"],
            "inflight":state["inflight"],"source_integrity_error":integrity_error,
            "history":deepcopy(state["history"]),"created_at":utcnow(),"skill_status":"draft_material_only"}
        h=digest(report);write(folder/"finalizations"/(h+".json"),report)
        state["finalization"]=h;state["outcome"]=outcome
        state["status"]="complete" if outcome=="success" else "closed"
        state["history"].append({"event":"materials_finalized","outcome":outcome,"sha256":h,"time":utcnow()})
        write(folder/"state.json",state)
        return report
