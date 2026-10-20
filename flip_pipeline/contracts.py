"""Task-directed requirements, scoped declarations and concise decision briefs.
Declarations record operator input; they are neither measurements nor ACLs.
"""
from .storage import read, inside, digest, verify_evidence

PROFILE = "supervised_rigid_flip"
PROFILE_PATH = "devices/task_profiles/supervised_rigid_flip.json"
CONDITIONS = ("empty", "rigid", "nonfragile", "cool_and_nonsharp", "within_equipment_handling_limits")

def load_contract(root, profile):
    if profile != PROFILE:
        raise ValueError("Unknown task profile")
    value = read(inside(root, PROFILE_PATH))
    if (value.get("id") != PROFILE or value.get("strategy") != "geometry_guided_pick_invert_place"
        or value.get("property_requirements") != [] or value.get("operator_supervised") is not True):
        raise ValueError("Unsupported task contract")
    return value

def verify_contract(root, state):
    contract = state.get("task_contract")
    if contract is None:
        if state.get("contract_sha256") is not None:
            raise ValueError("Task contract missing")
        return
    if digest(contract) != state.get("contract_sha256") or contract != load_contract(root, contract.get("id")):
        raise ValueError("Task contract changed; start a new task")

def task_directed(state):
    return bool(state and state.get("task_contract", {}).get("id") == PROFILE)

def required_checks(stage, state, defaults):
    if stage == "properties" and task_directed(state):
        return ("scope_membership",)
    return defaults[stage]

def validate_scope(root, data, refs, state):
    verify_contract(root, state)
    if not task_directed(state):
        raise ValueError("Scoped property report requires a task contract")
    verify_evidence(root, refs)
    named = {r["path"] for r in refs}
    path = data.get("scope_declaration_path")
    if path not in named:
        raise ValueError("Missing hashed scope declaration")
    declaration = read(inside(root, path))
    policy = read(inside(root, "devices/pipeline_policy.json"))
    if declaration.get("approved_by") not in policy.get("authorized_scope_reviewers", ["human"]):
        raise ValueError("Scope declaration needs configured operator/maintainer")
    if declaration.get("source") != "operator_declaration" or declaration.get("profile") != PROFILE:
        raise ValueError("Scope must come from an operator declaration")
    if declaration.get("installation_id") != state["installation_id"] or state["object_id"] not in declaration.get("object_ids", []):
        raise ValueError("Scope does not cover this object/installation")
    if declaration.get("operator_supervised") is not True:
        raise ValueError("Operator-supervised protocol required")
    if any(declaration.get("conditions", {}).get(k) is not True for k in CONDITIONS):
        raise ValueError("Scope declaration incomplete")
    source = declaration.get("evidence", [])
    if not source or path in source or not set(source).issubset(named):
        raise ValueError("Scope needs separately recorded operator evidence")
    if data.get("scope_contradictions") != []:
        raise ValueError("Scope contradiction or missing observation; resolve with operator")
    observations = data.get("observation_evidence", [])
    if not observations or not set(observations).issubset(named) or path in observations:
        raise ValueError("Current object observation evidence required")
    if data.get("mass_status") != "not_required" or data.get("material_status") != "not_required":
        raise ValueError("Unmeasured properties must remain not_required for this strategy")
    if any(k in data for k in ("mass_kg", "com_m", "mass_result_path", "material", "hazards")):
        raise ValueError("Scoped admission must not manufacture measured properties")
    verify_evidence(root, refs)
    return {"valid": True, "strategy": state["task_contract"]["strategy"],
            "mass_status": "not_required", "material_status": "not_required",
            "basis": "scoped operator declaration plus current observation; not a measurement"}

def validate_plan_contract(data, state):
    if not task_directed(state):
        return
    if data.get("strategy") != state["task_contract"]["strategy"] or data.get("property_requirements") != []:
        raise ValueError("Plan must declare the admitted geometry strategy and no physical-parameter dependency")
    if any(data.get(k) is not None for k in ("mass_result_path", "simulation_result_path", "com_m", "mass_kg")):
        raise ValueError("A parameter-dependent plan requires the measured-properties workflow")

QUESTIONS = {
    "calibration": "能否复用当前安装的已验证标定？",
    "model": "当前抓取和落放需要的几何信息是否足够？",
    "properties": "本物体是否在已声明的实验范围内，现场是否有矛盾？",
    "plan": "哪个抓取和翻转方案能满足落放目标及实际路径条件？",
    "execution": "上一动作是否完成，下一动作的条件是否成立？",
    "review": "结果、干预和原始材料是否记录齐全？",
}

def brief(root, task):
    from .workflow import status
    s = status(root, task)
    stage = s["stage"]
    blockers = []
    if s["integrity_error"]:
        blockers.append({"code": "integrity", "detail": s["integrity_error"], "owner": "maintainer"})
    if not s["terminal"] and stage == "calibration":
        p = inside(root, "devices/installation.json")
        installation = read(p) if p.exists() else {}
        if installation.get("status") != "verified":
            blockers.append({"code": "installation_unverified", "detail": "完成专门标定并发布档案；不要重复采集物性或自行求 IK。", "owner": "maintainer"})
    if not s["terminal"] and stage == "plan":
        blockers.append({"code": "path_review_required", "detail": "当前全臂检查后端是教师复核；提交具体候选路径，只请求复核该路径。", "owner": "teacher"})
    return {"task_id": s["task_id"], "object_id": s["object_id"], "stage": stage,
            "status": s["status"], "terminal": s["terminal"], "next_action": s["next_action"],
            "question": None if s["terminal"] else QUESTIONS[stage],
            "required_checks": s["required_checks"], "blockers": blockers,
            "property_policy": "not_required_by_geometry_strategy" if task_directed(s) else "measured_properties",
            "decision_record": ["当前事实及来源", "下一动作及简短理由", "动作完成的检查", "若需补测，说明它改变哪个动作参数或判据"],
            "stop_rule": "无新证据不重复同一测量/请求；基础设施缺项交维护者，实机异常停止并报告。"}
