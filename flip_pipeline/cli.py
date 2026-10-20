"""Offline CLI. Hardware dispatch remains in the single-session executor."""
import argparse,importlib,json,sys
from pathlib import Path
from .storage import read,write,inside,utcnow
from . import workflow

COMPUTATIONS={
 "controller-request":("controller_boundary","matrix_to_controller_move","kwargs"),
 "hand-eye":("calibration","solve_hand_eye","request"),
 "target-pose":("calibration","estimate_target_pose","request"),
 "profile-check":("calibration","validate_device_profile","kwargs"),
 "rgbd-cloud":("modeling","backproject_rgbd","kwargs"),
 "rigid-model":("modeling","build_rigid_model","kwargs"),
 "register-views":("modeling","register_multiview_open3d","kwargs"),
 "mose3-import":("modeling","import_mose3_tracks","kwargs"),
 "static-payload":("physics","estimate_static_payload","kwargs"),
 "weighed-components":("physics","combine_weighed_components","kwargs"),
 "support-projection":("physics","estimate_support_projection","kwargs"),
 "combine-projections":("physics","combine_support_projections","kwargs"),
 "properties":("physics","assess_properties","request"),
 "propose-flip":("planning","propose_rigid_flip","kwargs"),
 "support-margin":("planning","support_margin","kwargs"),
 "simulate-placement":("simulation","simulate_placement","request"),
 "teacher-materials":("experience","process_teacher","kwargs"),
 "student-materials":("experience","process_student","kwargs")}

def normalize(value):
    if hasattr(value,"tolist"):return value.tolist()
    if isinstance(value,dict):return {k:normalize(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [normalize(v) for v in value]
    return value

def main(root=None):
    root=Path(root or Path(__file__).resolve().parents[1]).resolve()
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="action",required=True)
    q=sub.add_parser("begin");q.add_argument("--object",required=True);q.add_argument("--installation",required=True);q.add_argument("--mode",choices=("experiment","calibration"),default="experiment");q.add_argument("--profile",choices=("supervised_rigid_flip","measured_properties"),default="supervised_rigid_flip")
    q=sub.add_parser("status");q.add_argument("task")
    q=sub.add_parser("reuse-calibration");q.add_argument("task")
    q=sub.add_parser("brief");q.add_argument("task")
    q=sub.add_parser("report-template");q.add_argument("stage",choices=workflow.STAGES);q.add_argument("output");q.add_argument("--task")
    q=sub.add_parser("submit");q.add_argument("task");q.add_argument("stage",choices=workflow.STAGES);q.add_argument("report");q.add_argument("--evidence",nargs="+",required=True)
    q=sub.add_parser("halt");q.add_argument("task");q.add_argument("reason")
    q=sub.add_parser("compute");q.add_argument("operation",choices=COMPUTATIONS);q.add_argument("input");q.add_argument("output")
    q=sub.add_parser("authorize");q.add_argument("task");q.add_argument("--run",required=True);q.add_argument("--segment",required=True);q.add_argument("--request",required=True);q.add_argument("--from-result",required=True);q.add_argument("--output",required=True);q.add_argument("--support-evidence")
    q=sub.add_parser("authorize-acquisition");q.add_argument("task");q.add_argument("--collection-plan",required=True);q.add_argument("--run",required=True);q.add_argument("--segment",required=True);q.add_argument("--request",required=True);q.add_argument("--from-result",required=True);q.add_argument("--output",required=True);q.add_argument("--support-evidence")
    q=sub.add_parser("calibration-task");q.add_argument("--installation",required=True)
    q=sub.add_parser("finalize");q.add_argument("task");q.add_argument("--outcome",choices=("success","failure","incomplete","unknown"),required=True);q.add_argument("--summary",required=True);q.add_argument("--evidence",nargs="+",required=True)
    q=sub.add_parser("collect-run");q.add_argument("run_path");q.add_argument("--episode",required=True);q.add_argument("--calibration",required=True);q.add_argument("--annotations");q.add_argument("--output",required=True)
    q=sub.add_parser("request-teacher");q.add_argument("task");q.add_argument("--teacher",choices=("astra","human"),default="human");q.add_argument("--missing",nargs="+",required=True);q.add_argument("--question",required=True);q.add_argument("--segment",required=True);q.add_argument("--evidence",nargs="*",default=[])
    sub.add_parser("doctor")
    a=p.parse_args()
    if a.action=="begin":result={"task":str(workflow.new_task(root,a.object,a.installation,a.mode,profile=a.profile if a.mode=="experiment" and a.profile!="measured_properties" else None)),"hardware_connected":False}
    elif a.action=="brief":
        from .contracts import brief
        result=brief(root,a.task)
    elif a.action=="status":result=workflow.status(root,a.task)
    elif a.action=="reuse-calibration":
        from .calibration_reuse import reuse_calibration
        result=reuse_calibration(root,a.task)
    elif a.action=="submit":result=workflow.attach_report(root,a.task,a.stage,read(inside(root,a.report)),a.evidence)
    elif a.action=="finalize":result=workflow.finalize(root,a.task,a.outcome,a.evidence,read(inside(root,a.summary)))
    elif a.action=="halt":result=workflow.halt(root,a.task,a.reason)
    elif a.action=="collect-run":
        from .material_adapter import import_run
        result=import_run(root,a.run_path,a.episode,a.calibration,a.annotations);write(inside(root,a.output),result)
    elif a.action=="request-teacher":
        from .experience import teacher_request
        import uuid
        folder,state=workflow.load_task(root,a.task)
        result=teacher_request(request_id=uuid.uuid4().hex,teacher=a.teacher,missing_information=a.missing,question=a.question,requested_segment=a.segment,state_snapshot=state,evidence_refs=a.evidence,timestamp=utcnow())
        write(folder/"teacher_requests"/(result["request_id"]+".json"),result)
    elif a.action=="report-template":
        from .contracts import required_checks, task_directed
        state=workflow.status(root,a.task) if a.task else None
        result={"source":"unverified","checks":{k:{"result":"unknown","method":sorted(workflow.ALLOWED_METHODS[k])[0],"evidence":[],"detail":""} for k in required_checks(a.stage,state,workflow.REQUIRED)}}
        extras={"calibration":{"profile_path":None,"installation_path":"devices/installation.json"},"model":{"model_request_path":None,"model_result_path":None},"properties":{"property_observations_path":None,"property_result_path":None,"mass_method":None,"mass_request_path":None,"mass_result_path":None,"mass_status":"unknown","hazards":{x:"unknown" for x in ("spill","breakage","sharp_or_hot")}},"plan":{"geometry_request_path":None,"geometry_result_path":None,"path_review_path":None,"path_scope":None,"execution_path":None,"segments":[]},"execution":{"run_id":None,"final_receipt_path":None,"completion_observations_path":None}}
        if task_directed(state):
            extras["properties"]={"scope_declaration_path":None,"scope_contradictions":None,"observation_evidence":[],"mass_status":"not_required","material_status":"not_required"}
            extras["plan"].update(strategy=state["task_contract"]["strategy"],property_requirements=[])
        result.update(extras.get(a.stage,{}))
        write(inside(root,a.output),result)
    elif a.action=="compute":
        module,name,mode=COMPUTATIONS[a.operation];f=getattr(importlib.import_module("flip_pipeline."+module),name);args=read(inside(root,a.input))
        result=normalize(f(args) if mode=="request" else f(**args));write(inside(root,a.output),result)
    elif a.action in ("authorize","authorize-acquisition"):
        from .supervision import issue
        result=issue(root,a.task,a.run,a.segment,read(inside(root,a.request)),a.from_result,support_evidence=read(inside(root,a.support_evidence)) if a.support_evidence else None,collection_plan=getattr(a,"collection_plan",None));write(inside(root,a.output),result)
    elif a.action=="calibration-task":
        folder=workflow.new_task(root,"installation",a.installation,"calibration")
        result={"task":str(folder),"status":"awaiting_measurements","hardware_connected":False,"created_at":utcnow(),
          "required":["fresh controller TCP/payload/mount identity","camera SDK intrinsics, distortion, alignment and depth scale","measured rigid target and corners","stationary synchronized multi-axis robot/target observations with predeclared heldout samples","measured camera, bracket and gripper envelopes plus TCP/grasp center","table plane, workspace obstacles and independent physical verification"],
          "capture":"Read-only observe/status use existing robot entry. Before verified geometry, view changes use supervised teaching; do not synthesize a calibration trajectory.",
          "reuse":"Keep installation calibration across new objects; invalidate after mounting, TCP, camera stream or depth-scale changes."}
        write(folder/"acquisition.json",result)
    else:
        mods={}
        for name in ("numpy","scipy","cv2","open3d","mujoco"):
            try:m=importlib.import_module(name);mods[name]={"version":getattr(m,"__version__","unknown"),"path":m.__file__}
            except ImportError as exc:mods[name]={"unavailable":str(exc)}
        result={"modules":mods,"hardware_connected":False,"python":sys.executable}
    print(json.dumps(normalize(result),ensure_ascii=False,allow_nan=False,indent=2))

if __name__=="__main__":
    try:main()
    except Exception as exc:raise SystemExit(type(exc).__name__+": "+str(exc))
