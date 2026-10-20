"""Three abstraction layers. Object attributes are applicability, not a fourth layer."""
import copy
import hashlib
import json
from pathlib import Path

LIBRARY = Path(__file__).resolve().parent/'library'
KINDS = {'goals': 'goal', 'plans': 'plan', 'capabilities': 'capability'}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_catalog(root=LIBRARY):
    root=Path(root)
    layers={}
    for directory,kind in KINDS.items():
        records={}
        for path in sorted((root/directory).glob('*.json')):
            item=_read(path)
            if item.get('kind')!=kind or item.get('schema_version')!=1:
                raise ValueError(f'Wrong catalog kind/schema: {path}')
            if item.get('id')!=path.stem or type(item.get('version')) is not int or item['version']<1:
                raise ValueError(f'Invalid identity/version: {path}')
            records[item['id']]=item
        layers[directory]=records
    layers['aliases']=_read(root/'aliases.json')
    validate_catalog(layers)
    return layers


def validate_catalog(layers):
    for goal in layers['goals'].values():
        if any(key in goal for key in ('action','requires','capabilities','completion','specializes')):
            raise ValueError('Goal cannot prescribe an execution route')
        if not goal.get('binding_fields') or not goal.get('acceptance'):
            raise ValueError('Goal needs task bindings and acceptance semantics')
    for alias,target in layers['aliases'].items():
        if alias in layers['capabilities'] or target not in layers['capabilities']:
            raise ValueError('Invalid or shadowed capability alias')
    for capability in layers['capabilities'].values():
        if capability.get('action') not in ('observe','move','grip'):
            raise ValueError('Unsupported capability adapter action')
        if any(key in capability for key in ('goal','specializes','completion','goal_binding')):
            raise ValueError('Capability cannot own task-goal completion')
    for ident,plan in layers['plans'].items():
        if plan.get('goal') not in layers['goals']:
            raise ValueError(f'Unknown goal for plan: {ident}')
        if any(key in plan for key in ('action','pose','joint_angles')):
            raise ValueError('Plan cannot contain a direct hardware command')
        resolved=resolve_plan(ident,layers)
        caps=resolved.get('capabilities',[])
        if not caps or any(c not in layers['capabilities'] for c in caps):
            raise ValueError(f'Plan references unknown capability: {ident}')
        completion=resolved.get('completion',{})
        if completion.get('capability') not in caps:
            raise ValueError('Completion capability missing from plan')
        if not set(completion.get('requires_completed',[])).issubset(caps):
            raise ValueError('Completion prerequisite missing from plan')
        if not resolved.get('goal_binding') or not resolved.get('applicability'):
            raise ValueError('Plan needs explicit goal binding and applicability')


def resolve_plan(ident,layers=None,_seen=()):
    layers=load_catalog() if layers is None else layers
    if ident in _seen:raise ValueError('Plan specialization cycle')
    plan=layers['plans'][ident]
    if not plan.get('specializes'):return copy.deepcopy(plan)
    parent=resolve_plan(plan['specializes'],layers,_seen+(ident,))
    if plan['goal']!=parent['goal']:raise ValueError('Specialization must retain parent goal')
    # Specialization narrows applicability: preserve BOTH parent and child constraints.
    applicability=copy.deepcopy(parent.get('applicability',{}))
    for group,conditions in plan.get('applicability',{}).items():
        applicability[group]=list(dict.fromkeys(applicability.get(group,[])+conditions))
    resolved={**parent,**copy.deepcopy(plan),'applicability':applicability}
    return resolved


def resolve_capability(ident,layers=None):
    layers=load_catalog() if layers is None else layers
    canonical=layers['aliases'].get(ident,ident)
    if canonical not in layers['capabilities']:
        raise ValueError(f'Not an executable capability: {ident}')
    return layers['capabilities'][canonical]


def plan_snapshot(ident,root=LIBRARY):
    root=Path(root);layers=load_catalog(root);plan=resolve_plan(ident,layers)
    selected={'aliases.json',f'goals/{plan["goal"]}.json',*(f'capabilities/{c}.json' for c in plan['capabilities'])}
    current=ident
    while current:
        selected.add(f'plans/{current}.json');current=layers['plans'][current].get('specializes')
    return {'goal':copy.deepcopy(layers['goals'][plan['goal']]),'plan':plan,
            'aliases':{alias:target for alias,target in layers['aliases'].items() if target in plan['capabilities']},
            'capabilities':{c:copy.deepcopy(layers['capabilities'][c]) for c in plan['capabilities']},
            'source_sha256':{rel:hashlib.sha256((root/rel).read_bytes()).hexdigest() for rel in sorted(selected)}}
