"""Reuse a maintainer-published calibration; no hardware access."""
from pathlib import Path
from .storage import read, inside, verify_evidence
from .release import verify_release
from . import workflow

def reuse_calibration(root, task):
    root=Path(root).resolve()
    verify_release(root)
    index=read(root/"devices/calibration_release.json")
    if index.get("status")!="verified":raise ValueError("No verified calibration release")
    refs=index["evidence"]
    verify_evidence(root,refs)
    folder,state=workflow.load_task(root,task)
    workflow.verify_task(root,folder,state)
    if index.get("task_profiles") and state.get("task_contract",{}).get("id") not in index["task_profiles"]:
        raise ValueError("Calibration release does not cover this task profile")
    installation=read(inside(root,index["installation_path"]))
    if installation.get("status")!="verified" or state["installation_id"]!=installation["installation_id"] or index["installation_id"]!=installation["installation_id"]:
        raise ValueError("Task/current installation differs from calibration release")
    if state["status"]!="active" or state.get("inflight"):raise ValueError("Task not idle and active")
    if "calibration" in state["artifacts"]:
        return {"reused":True,"already_attached":True,"task":folder.relative_to(root).as_posix(),"stage":state["stage"]}
    if state["stage"]!="calibration":raise ValueError("Task is not awaiting calibration")
    report=read(inside(root,index["report_path"]))
    artifact=workflow.attach_report(root,folder,"calibration",report,[x["path"] for x in refs])
    return {"reused":True,"installation_id":installation["installation_id"],"task":folder.relative_to(root).as_posix(),"stage":workflow.status(root,folder)["stage"],"artifact":artifact["kind"]}
