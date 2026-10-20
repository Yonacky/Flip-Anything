"""Recompute supported geometry; bind external whole-robot path review explicitly."""
from .storage import inside,read,digest,verify_evidence
from .controller_boundary import validate_controller_request

def validate_plan_report(root,data,refs,state):
    from .contracts import validate_plan_contract
    validate_plan_contract(data,state)
    from .planning import propose_rigid_flip
    # Do not equate a table-only envelope calculation with a robot path check.
    indexed={r["path"]:r for r in refs}
    def documented(key):
        value=data.get(key)
        if value not in indexed:raise ValueError("Plan missing hashed "+key)
        return read(inside(root,value))
    request=documented("geometry_request_path");saved=documented("geometry_result_path")
    result=propose_rigid_flip(**request)
    # JSON results are deterministic for this algorithm/version; tolerate float rounding.
    def equivalent(a,b):
        import numpy as np
        if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equivalent(a[k],b[k]) for k in a)
        if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
        if isinstance(a,(int,float)) and not isinstance(a,bool):return isinstance(b,(int,float)) and not isinstance(b,bool) and bool(np.isclose(a,b,rtol=1e-7,atol=1e-9))
        return a==b
    if not equivalent(result,saved):raise ValueError("Geometry result differs from recomputation")
    for segment in data["segments"]:validate_controller_request(segment["request"])
    review=documented("path_review_path")
    policy=read(inside(root,"devices/pipeline_policy.json"))
    if review.get("reviewer_role") not in ("maintainer","human","teacher") or review.get("reviewer_id") not in policy["authorized_path_reviewers"]:raise ValueError("Whole-robot path review requires configured teacher/maintainer")
    for key in ("task_id","object_id","installation_id"):
        if review.get(key)!=state[key]:raise ValueError("Path review identity mismatch")
    if review.get("segments_sha256")!=digest(data["segments"]) or review.get("geometry_sha256")!=digest(result):raise ValueError("Path review not bound to this plan")
    if review.get("result")!="accepted" or review.get("method")!="teacher_review":raise ValueError("Automatic full-robot path solver not configured; current backend requires actual teacher review")
    for key in ("robot_links","tool_and_camera","held_object","obstacles","actual_interpolation"):
        check=review.get("checks",{}).get(key,{})
        if check.get("result")!="pass" or not check.get("evidence") or not set(check["evidence"]).issubset(indexed):raise ValueError("Path review missing "+key)
    verify_evidence(root,refs)
    return {"geometry_recomputed":True,"full_robot_backend":"teacher_review","automatic_full_robot_certification":False}
