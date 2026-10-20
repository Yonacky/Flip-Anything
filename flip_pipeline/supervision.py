"""Single-use, hash-bound command authorizations for the existing hardware worker."""
import math,time,uuid
import numpy as np
from pathlib import Path
from .storage import read,write,digest,inside,locked,utcnow,verify_evidence
from .workflow import load_task,verify_task,halt
from .controller_boundary import validate_controller_request,validate_motion_against_start,controller_pose_to_matrix

def _command(request):
    return {k:v for k,v in request.items() if k not in ("pipeline_ticket",)}

def acquisition_plan(root,path,state):
    plan=read(inside(root,path));copy={k:v for k,v in plan.items() if k!="reviewed_plan_sha256"}
    if plan.get("reviewed_plan_sha256")!=digest(copy):raise ValueError("Acquisition review hash mismatch")
    policy=read(inside(root,"devices/pipeline_policy.json"))
    if plan.get("reviewed") is not True or plan.get("reviewer_role") not in ("human","teacher","maintainer") or plan.get("reviewer_id") not in policy["authorized_path_reviewers"]:raise ValueError("Acquisition needs teacher review")
    for key in ("task_id","object_id","installation_id"):
        if plan.get(key)!=state[key]:raise ValueError("Acquisition identity mismatch")
    allowed={"calibration_views":"calibration","object_views":"model","material_probe":"properties","tcp_grip_probe":"calibration"}
    if plan.get("purpose") not in allowed or state["stage"]!=allowed[plan["purpose"]]:raise ValueError("Acquisition purpose does not match current stage")
    if plan.get("checks",{}).get("path_reviewed") is not True:raise ValueError("Acquisition path not reviewed")
    if plan["purpose"] in ("material_probe","tcp_grip_probe"):
        if plan["checks"].get("contact_risks_excluded") is not True:raise ValueError("Contact risks unresolved")
    elif plan["checks"].get("empty_tool") is not True:raise ValueError("Observation acquisition requires empty tool")
    if plan["purpose"]=="tcp_grip_probe" and plan["checks"].get("tcp_reference_reviewed") is not True:
        raise ValueError("TCP probe needs measured reference review")
    verify_evidence(root,plan.get("evidence",[]))
    if plan.get("initial_result") not in plan["evidence"]:raise ValueError("Teacher-reviewed initial receipt must be hashed evidence")
    if not read(inside(root,plan["initial_result"]["path"])).get("ok"):raise ValueError("Acquisition initial receipt failed")
    ids=[]
    for segment in plan.get("segments",[]):
        validate_controller_request(segment["request"]);ids.append(segment["id"])
        if plan["purpose"]=="tcp_grip_probe":
            req=segment["request"]
            if req["action"]=="move":
                if req.get("motion_space")!="cartesian_translation" or req.get("speed",float("inf"))>5 or req.get("max_translation_mm",float("inf"))>10 or req.get("max_rotation_deg",float("inf"))>0.49 or req.get("max_joint_deg",float("inf"))>8:
                    raise ValueError("TCP probe move exceeds declared bounds")
                if req.get("grip_required"):raise ValueError("TCP probe cannot move a held object")
            elif req.get("speed",40)>40 or req.get("force",10)>10:
                raise ValueError("TCP probe gripper exceeds declared bounds")

        if plan["purpose"] not in ("material_probe","tcp_grip_probe") and (segment["request"]["action"]!="move" or segment["request"]["grip_required"]):raise ValueError("View acquisition cannot manipulate an object")
    if not ids or len(set(ids))!=len(ids):raise ValueError("Acquisition needs unique ordered segments")
    return plan

def issue(root,task,run_id,segment_id,request,from_result,ttl_s=30,support_evidence=None,collection_plan=None):
    validate_controller_request(request)
    folder,state=load_task(root,task)
    with locked(folder/"state.lock"):
        state=verify_task(root,folder)
        if state["status"]!="active":raise ValueError("Task halted or complete")
        if not collection_plan and (state["mode"]!="experiment" or state["stage"]!="execution"):raise ValueError("Execution requires completed calibration/model/properties/plan")
        if state["inflight"]:raise ValueError("Previous command outcome unknown; inspect receipt, never resend")
        if not 0<ttl_s<=60:raise ValueError("Ticket lifetime must be 0..60 seconds")
        plan=acquisition_plan(root,collection_plan,state) if collection_plan else read(folder/"artifacts"/(state["artifacts"]["plan"]+".json"))["data"]
        plan_hash=digest(plan) if collection_plan else state["artifacts"]["plan"]
        segments=plan["segments"]
        done=[h["segment_id"] for h in state["history"] if h["event"]=="action_completed" and h.get("plan_hash")==plan_hash]
        if len(done)>=len(segments) or segments[len(done)]["id"]!=segment_id:raise ValueError("Execute next uncompleted segment only")
        segment=segments[len(done)]
        if request!=segment["request"]:raise ValueError("Command differs from the checked plan segment")
        if request.get("action") not in ("move","grip") or not isinstance(request.get("grip_required"),bool):raise ValueError("Invalid executable request")
        ref=inside(root,from_result)
        active=read(Path(root)/"ds_control/active.json")
        if active["run"]!=run_id or ref.parent.name!=run_id or not ref.name.startswith("result_"):raise ValueError("Receipt/run mismatch")
        from flip_pipeline.storage import evidence
        prior=read(ref)
        if not prior.get("ok") or "pose" not in prior.get("state",{}):raise ValueError("Successful current pose receipt required")
        age=time.monotonic_ns()-prior["state"]["sample_end_monotonic_ns"]
        if not 0<=age<=60e9:raise ValueError("Refresh robot status; receipt is stale")
        if collection_plan and not done:
            initial=read(inside(root,plan["initial_result"]["path"]))
            A=controller_pose_to_matrix(initial["state"]["pose"]);B=controller_pose_to_matrix(prior["state"]["pose"])
            angle=math.degrees(math.acos(float(np.clip((np.trace(A[:3,:3].T@B[:3,:3])-1)/2,-1,1))))
            if np.linalg.norm(A[:3,3]-B[:3,3])>.002 or angle>2:raise ValueError("Acquisition start differs from teacher-reviewed pose")
        if done:
            last=[h for h in state["history"] if h["event"]=="action_completed" and h.get("plan_hash")==plan_hash][-1]
            end_ns=last.get("result",{}).get("state",{}).get("sample_end_monotonic_ns",0)
            if prior["state"]["sample_end_monotonic_ns"]<end_ns:raise ValueError("Use feedback after the preceding action")
        validate_motion_against_start(request,prior["state"]["pose"])
        if collection_plan and plan["purpose"]=="tcp_grip_probe":
            _validate_tcp_probe_live(request,prior["state"],read(inside(root,plan["initial_result"]["path"]))["state"]["pose"])

        payload=dict(request,expected_pose=prior["state"]["pose"])
        gripper=prior["state"].get("gripper") or {}
        if collection_plan and plan["purpose"] not in ("material_probe","tcp_grip_probe") and (gripper.get("gOBJ")!=3 or gripper.get("gACT")!=1 or gripper.get("gSTA")!=3 or gripper.get("gFLT")!=0 or not isinstance(gripper.get("gPO"),int) or not 0<=gripper["gPO"]<=255):raise ValueError("Observation acquisition requires current empty-tool feedback")
        opening=request.get("action")=="grip" and ("gPO" not in gripper or request.get("pos",0)<gripper["gPO"])
        possible_held=gripper.get("gOBJ") in (1,2) or request.get("grip_required") or not gripper
        if segment.get("role")=="release" or (opening and possible_held):
            support=support_evidence
            if not support or support.get("run_id")!=run_id or support.get("method") not in ("observed_contact","calibrated_contact_measurement"):raise ValueError("Release needs current measured/observed support")
            support_path=inside(root,support["path"])
            if digest(read(support_path))!=support["sha256"]:raise ValueError("Support evidence changed")
            support_data=read(support_path)
            if support_data.get("supported") is not True or support_data.get("object_id")!=state["object_id"] or support_data.get("run_id")!=run_id:raise ValueError("Support identity or result mismatch")
            if support_data.get("reference_receipt_sha256")!=digest(prior) or support_data.get("completed_action_count")!=len(done):raise ValueError("Support evidence must bind current receipt and completed action sequence")
            verify_evidence(root,support_data.get("raw_evidence",[]))
            elapsed=time.monotonic_ns()-support_data["sample_monotonic_ns"]
            if not 0<=elapsed<=30e9:raise ValueError("Support observation stale")
        ident=uuid.uuid4().hex
        ticket={"schema_version":1,"ticket_id":ident,"task":str(folder.relative_to(Path(root).resolve())),
                "run_id":run_id,"segment_id":segment_id,"command_hash":digest(payload),"plan_hash":plan_hash,"collection_plan":str(inside(root,collection_plan).relative_to(Path(root).resolve())) if collection_plan else None,"dependencies":dict(state["artifacts"]),
                "issued_monotonic_ns":time.monotonic_ns(),"expires_monotonic_ns":time.monotonic_ns()+int(ttl_s*1e9),
                "state":"issued","reference":evidence(root,[ref]),"support_evidence":support_evidence,"time":utcnow()}
        ticketpath=folder/"tickets"/(ident+".json");write(ticketpath,ticket)
        payload["pipeline_ticket"]=str(ticketpath.relative_to(Path(root).resolve()))
        return payload

def consume(root,run_id,payload):
    ticketpath=inside(root,payload.get("pipeline_ticket","MISSING_TICKET"))
    ticket=read(ticketpath);folder,state=load_task(root,ticket["task"])
    if ticketpath.parent!=folder/"tickets":raise ValueError("Ticket location mismatch")
    with locked(folder/"state.lock"):
        ticket=read(ticketpath);state=verify_task(root,folder)
        now=time.monotonic_ns()
        verify_evidence(root,ticket["reference"])
        if ticket["state"]!="issued" or not ticket["issued_monotonic_ns"]<=now<=ticket["expires_monotonic_ns"]:raise ValueError("Ticket consumed/expired")
        if ticket["run_id"]!=run_id or ticket["command_hash"]!=digest(_command(payload)):raise ValueError("Ticket binding mismatch")
        if state["status"]!="active" or state["inflight"]:raise ValueError("Task not ready for command")
        if state["artifacts"]!=ticket["dependencies"]:raise ValueError("Stage evidence changed")
        if ticket.get("collection_plan"):
            plan=acquisition_plan(root,ticket["collection_plan"],state)
            if digest(plan)!=ticket["plan_hash"]:raise ValueError("Acquisition plan changed")
        else:
            if state["stage"]!="execution" or state["artifacts"]["plan"]!=ticket["plan_hash"]:raise ValueError("Plan changed")
            plan=read(folder/"artifacts"/(ticket["plan_hash"]+".json"))["data"]
        done=[h for h in state["history"] if h["event"]=="action_completed" and h.get("plan_hash")==ticket["plan_hash"]]
        if len(done)>=len(plan["segments"]) or plan["segments"][len(done)]["id"]!=ticket["segment_id"]:raise ValueError("Segment already executed or no longer next")
        support=ticket.get("support_evidence")
        if support:
            observed=read(inside(root,support["path"]))
            if digest(observed)!=support["sha256"] or not 0<=now-observed["sample_monotonic_ns"]<=30e9:raise ValueError("Support evidence stale/changed")
            if observed.get("completed_action_count")!=len(done):raise ValueError("An action happened after support measurement")
            verify_evidence(root,observed.get("raw_evidence",[]))
        now=time.monotonic_ns()
        if now>ticket["expires_monotonic_ns"]:raise ValueError("Ticket expired during evidence verification")
        if support and now-observed["sample_monotonic_ns"]>30e9:raise ValueError("Support expired during verification")
        ticket["state"]="consumed";write(ticketpath,ticket)
        state["inflight"]={"ticket_id":ticket["ticket_id"],"run_id":run_id,"segment_id":ticket["segment_id"],"command_hash":ticket["command_hash"]}
        write(folder/"state.json",state)
    return folder,ticket

def complete(root,folder,ticket,result):
    with locked(Path(folder)/"state.lock"):
        state=read(Path(folder)/"state.json")
        if not state["inflight"] or state["inflight"]["ticket_id"]!=ticket["ticket_id"]:raise ValueError("Unknown inflight action")
        state["history"].append({"event":"action_completed" if result.get("ok") else "action_failed","segment_id":ticket["segment_id"],
                                 "ticket_id":ticket["ticket_id"],"plan_hash":ticket["plan_hash"],"run_id":ticket["run_id"],"result":result,"time":utcnow()})
        state["inflight"]=None;write(Path(folder)/"state.json",state)
    if not result.get("ok"):halt(root,folder,str(result.get("error","Action failed")),ticket["run_id"])

def execute_guarded(root,run_id,payload,callback):
    folder,ticket=consume(root,run_id,payload)
    try:
        result=callback()
        if result.get("ok") is False:raise RuntimeError(str(result.get("error","Callback reported failure")))
    except BaseException as exc:
        complete(root,folder,ticket,{"ok":False,"error":type(exc).__name__+": "+str(exc)});raise

    complete(root,folder,ticket,{"ok":True,**result});return result

def halt_run_tasks(root,run_id,reason):
    for path in (Path(root)/"logs/pipeline").glob("*/state.json"):
        state=read(path)
        tickets=list((path.parent/"tickets").glob("*.json"))
        if state["status"]=="active" and any(read(t).get("run_id")==run_id for t in tickets):
            halt(root,path.parent,reason,run_id)

def validate_grip_transition(root,payload,before):
    if payload.get("action")!="grip":return
    grip=before.get("gripper") or {}
    opening="gPO" not in grip or payload["pos"]<grip["gPO"]
    held=grip.get("gOBJ") in (1,2) or payload.get("grip_required") or not grip
    if opening and held:
        ticket=read(inside(root,payload["pipeline_ticket"]))
        support=ticket.get("support_evidence")
        if not support:raise ValueError("Opening a possibly loaded gripper requires support evidence")
        data=read(inside(root,support["path"]))
        if digest(data)!=support["sha256"] or not 0<=time.monotonic_ns()-data["sample_monotonic_ns"]<=30e9:raise ValueError("Support evidence expired/changed")

def _validate_tcp_probe_live(request,before,initial_pose):
    g=before.get("gripper") or {}
    if g.get("gACT")!=1 or g.get("gSTA")!=3 or g.get("gFLT")!=0 or not isinstance(g.get("gPO"),int):
        raise ValueError("TCP probe needs activated fault-free gripper feedback")
    if request["action"]=="move":
        if g.get("gOBJ")!=3:raise ValueError("TCP probe move requires confirmed empty gripper")
        start=controller_pose_to_matrix(initial_pose);target=controller_pose_to_matrix(request["pose"])
        angle=math.degrees(math.acos(float(np.clip((np.trace(start[:3,:3].T@target[:3,:3])-1)/2,-1,1))))
        if np.linalg.norm(target[:3,3]-start[:3,3])>.1 or angle>.49:
            raise ValueError("TCP probe target outside reviewed local envelope")
    elif g.get("gOBJ")==0 and request.get("pos",255)>g["gPO"]:
        raise ValueError("Initialize TCP probe with an open request, not an unknown-state closure")

def validate_live_acquisition(root,payload,before):
    ticket=read(inside(root,payload["pipeline_ticket"]))
    if not ticket.get("collection_plan"):return
    plan=read(inside(root,ticket["collection_plan"]))
    if digest(plan)!=ticket["plan_hash"]:raise ValueError("Acquisition plan changed")
    if plan["purpose"]=="tcp_grip_probe":
        initial=read(inside(root,plan["initial_result"]["path"]))
        _validate_tcp_probe_live(payload,before,initial["state"]["pose"])
    if plan["purpose"] not in ("material_probe","tcp_grip_probe"):
        g=before.get("gripper") or {}
        if g.get("gOBJ")!=3 or g.get("gACT")!=1 or g.get("gSTA")!=3 or g.get("gFLT")!=0 or not isinstance(g.get("gPO"),int) or not 0<=g["gPO"]<=255:
            raise ValueError("Acquisition requires fresh empty stationary tool feedback")
