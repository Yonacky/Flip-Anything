"""Small stateful skill runtime. It does not infer geometry or contact from text."""
import hashlib
import json
import math
import subprocess
import sys
import time
import uuid
from pathlib import Path
from .catalog import load_catalog, resolve_capability, plan_snapshot

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def library():
    return load_catalog()['capabilities']


def retrieve(facts, include_candidates=False):
    return [s for s in library().values()
            if (s['status'] == 'validated' or include_candidates)
            and set(s['requires']).issubset(facts)]


def new_episode(folder, run_id, plan_id="pick_invert_place", device_profile=None):
    snapshot = plan_snapshot(plan_id)
    profile = read(device_profile) if device_profile else None
    if profile is not None and profile.get('kind') != 'device_profile':
        raise ValueError('Expected separate device profile')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    state = {'id': folder.name, 'run_id': run_id, 'revision': 0,
             'phase': 'ready', 'pending': None, 'completed': [],
             'human_interventions': [], 'created_at': time.time(),
             'goal_id':snapshot['goal']['id'], 'plan_id':plan_id, 'catalog_snapshot':'catalog_snapshot.json'}
    write(folder/'catalog_snapshot.json', snapshot)
    if profile is not None:
        write(folder/'device_profile.json',profile)
        state['device_profile'] = 'device_profile.json'
    write(folder/'scene.json',{'kind':'scene','status':'unobserved','object_ref':None,
          'table':None,'object_pose':None,'grasp_relation':None,'evidence':[]})
    write(folder/'state.json', state)
    return state


def episode_plan(folder,state):
    if state.get('catalog_snapshot'):
        return read(Path(folder)/state['catalog_snapshot'])['plan']
    # Compatibility for pre-layer episodes; no historical state is rewritten.
    return plan_snapshot('pick_invert_place')['plan']


def episode_capability(folder,state,ident):
    if state.get('catalog_snapshot'):
        snapshot=read(Path(folder)/state['catalog_snapshot'])
        caps=snapshot['capabilities']
        # Early layer snapshots predate alias capture; their two aliases are fixed.
        aliases=snapshot.get('aliases',{'flip':'invert_held_object','verify':'verify_inverted_placement'})
        canonical=aliases.get(ident,ident)
        if canonical not in caps:
            raise ValueError('Capability not part of this episode plan')
        return caps[canonical]
    canonical=resolve_capability(ident)['id']
    # Preserve pre-layer semantics from the archived definitions, not current edits.
    reverse={v:k for k,v in read(ROOT/'library/aliases.json').items()}
    legacy_id=reverse.get(canonical,canonical)
    old=read(ROOT/'library/legacy_v0'/f'{legacy_id}.json')
    old['id']=canonical
    return old


def event(folder, kind, **data):
    with (Path(folder)/'events.jsonl').open('a', encoding='utf8') as f:
        f.write(json.dumps({'time': time.time(), 'kind': kind, **data}, ensure_ascii=False)+'\n')


def evidence(facts, required, state, reference, allow_human=False):
    """Checks provenance/freshness, NOT the semantic correctness of perception."""
    for name in required:
        item = facts.get(name, {})
        if item.get('value') is not True:
            raise ValueError(f'Missing positive evidence: {name}')
        if item.get('revision') != state['revision']:
            raise ValueError(f'Stale evidence: {name}')
        if item.get('result_sha256') != digest(reference):
            raise ValueError(f'Evidence belongs to another result: {name}')
        if item.get('source') not in ('perception', 'controller', 'human') or not item.get('method'):
            raise ValueError(f'Unknown evidence source/method: {name}')
        if item['source'] == 'human' and not allow_human:
            raise ValueError('Human evidence requires supervised mode')
        paths = item.get('artifacts', [])
        if not paths or not all(Path(p).is_file() for p in paths):
            raise ValueError(f'Missing evidence artifacts: {name}')
        if name == 'supported' and item.get('source') == 'controller':
            raise ValueError('Controller/gripper status alone does not prove table support')


class ExistingControl:
    """Only adapter allowed to dispatch hardware. Never imports or initializes an SDK."""
    backend = "hardware"

    def __init__(self, run_id):
        self.run_id = run_id
        self.workspace = PROJECT/'ds_control'

    def dispatch(self, payload, reference):
        active = read(self.workspace/'active.json')
        if active['run'] != self.run_id:
            raise RuntimeError('Active hardware run changed')
        config_path = self.workspace/'configs/platform.json'
        recording_root = read(config_path)['recording']['root']
        run = (config_path.parent/recording_root/self.run_id).resolve()
        if read(run/'manifest.json')['status'] != 'running' or (run/'STOP').exists():
            raise RuntimeError('Hardware run not running')
        if reference is not None and Path(reference).resolve().parent != run.resolve():
            raise ValueError('Reference is not from active hardware run')
        args = [str(PROJECT/'.venv/Scripts/python.exe'), '-X', 'utf8',
                str(self.workspace/'scripts/control.py')]
        if payload['action'] == 'observe':
            args += ['observe']
        else:
            path = self.workspace/'scene'/('harness_'+uuid.uuid4().hex+'.json')
            write(path, payload)
            args += ['execute', str(path), '--from-result', str(reference)]
        proc = subprocess.run(args, cwd=self.workspace, capture_output=True,
                              encoding='utf8', timeout=135)
        for line in reversed(proc.stdout.splitlines()):
            try:
                result = json.loads(line)
                if 'result_file' in result and 'result' in result:
                    return result
            except ValueError:
                continue
        raise RuntimeError(f'No definitive receipt; do not retry. {proc.stderr[-1000:]}')


def run_skill(folder, skill_id, payload, facts, reference, adapter,
              allow_candidate=False, supervised=False):
    folder = Path(folder)
    state = read(folder/'state.json')
    if state['phase'] != 'ready':
        raise RuntimeError('Previous step must be assessed; unknown/failure needs reconciliation')
    requested_id = skill_id
    skill = episode_capability(folder, state, skill_id)
    skill_id = skill['id']
    if skill['status'] != 'validated' and not allow_candidate:
        raise ValueError('Candidate skill requires explicit development opt-in')
    if payload.get('action') != skill['action']:
        raise ValueError('Payload action does not match skill')
    if skill['requires']:
        if reference is None or not read(reference).get('ok'):
            raise ValueError('Successful previous receipt required')
        if state.get('last_result') and Path(reference).resolve() != Path(state['last_result']).resolve():
            raise ValueError('Use the latest episode receipt')
        evidence(facts, skill['requires'], state, reference, supervised)
    if skill['action'] == 'move':
        pose = payload.get('pose', [])
        if len(pose) != 6 or not all(type(v) in (int, float) and math.isfinite(v) for v in pose):
            raise ValueError('Move needs six finite pose values')
        if payload.get('grip_required') is not skill['hold_required']:
            raise ValueError('Wrong held-object guard for skill')
    if skill['action'] == 'grip':
        if type(payload.get('pos')) is not int or not 0 <= payload['pos'] <= 255:
            raise ValueError('Invalid gripper position')
        if payload.get('grip_required') is not False:
            raise ValueError('Gripper transitions need grip_required=false')
    for name in skill['requires']:
        if facts[name]['source'] == 'human':
            state['human_interventions'].append({'skill': skill_id, 'fact': name})
    # Persist BEFORE dispatch: crashes must never make a command silently retryable.
    backend = getattr(adapter, 'backend', 'mock')
    if state.get('backend') not in (None, backend):
        raise ValueError('Cannot mix mock and hardware in an episode')
    state.update(phase='dispatching', pending=skill_id, backend=backend)
    write(folder/'state.json', state)
    event(folder, 'skill_requested', skill=skill_id, requested_id=requested_id,
          goal=state.get('goal_id'),plan=state.get('plan_id'),payload=payload,facts=facts,
          skill_sha256=hashlib.sha256(json.dumps(skill,sort_keys=True,ensure_ascii=False).encode('utf8')).hexdigest())
    try:
        receipt = adapter.dispatch(payload, reference)
    except Exception as exc:
        state['phase'] = 'unknown'
        write(folder/'state.json', state)
        event(folder, 'outcome_unknown', error=str(exc))
        raise
    state['revision'] += 1
    state['last_result'] = str(Path(receipt['result_file']).resolve())
    state['phase'] = 'awaiting_assessment' if receipt['result'].get('ok') else 'blocked'
    write(folder/'state.json', state)
    event(folder, 'skill_receipt', receipt=receipt)
    return state


def assess(folder, facts, supervised=False):
    folder = Path(folder)
    state = read(folder/'state.json')
    if state['phase'] != 'awaiting_assessment':
        raise RuntimeError('No successful action awaiting assessment')
    skill = episode_capability(folder, state, state['pending'])
    reference = state['last_result']
    evidence(facts, skill['ensures'], state, reference, supervised)
    result = read(reference)
    if not result.get('ok'):
        raise ValueError('Failed receipt')
    grip = result.get('state', {}).get('gripper') or {}
    if skill['id'] in ('grasp','lift_test','invert_held_object','place'):
        if grip.get('gFLT') != 0 or grip.get('gOBJ') not in (1,2):
            raise ValueError('Grasp evidence contradicts controller feedback')
    if skill['id'] == 'release' and (grip.get('gFLT') != 0 or grip.get('gOBJ') != 3):
        raise ValueError('Release not corroborated by gripper feedback')
    for name in skill['ensures']:
        if facts[name]['source'] == 'human':
            state['human_interventions'].append({'skill':skill['id'],'fact':name})
    state['completed'].append(skill['id'])
    state.update(phase='ready', pending=None)
    plan = episode_plan(folder,state)
    completion = plan['completion']
    if skill['id'] == completion['capability']:
        completed = {episode_capability(folder,state,c)['id'] for c in state['completed']}
        if not set(completion['requires_completed']).issubset(completed):
            raise ValueError('Task plan completion prerequisites missing')
        state['phase'] = 'completed'
        write(folder/'outcome.json', {'backend':state['backend'], 'goal_id':plan['goal'],
              'plan_id':plan['id'], 'physical_success': True,
              'autonomous_success': not bool(state['human_interventions']),
              'human_interventions':state['human_interventions'],
              'evidence':facts, 'assessment':'provided evidence plus controller checks; not independent vision certification'})
    write(folder/'state.json', state)
    event(folder, 'skill_assessed', skill=skill['id'], facts=facts)
    return state


def reconcile(folder, reference, reason):
    """Explicit return to observation after inspecting unknown/rejected results; never auto-resend."""
    folder=Path(folder);state=read(folder/'state.json')
    if state['phase'] not in ('blocked','unknown','dispatching','awaiting_assessment'):
        raise ValueError('No recovery needed')
    if not reason.strip() or not read(reference).get('ok'):
        raise ValueError('Recovery needs reason and a fresh successful observation receipt')
    # Freshness and provenance: same hardware run, newer than the episode's last receipt.
    ref=Path(reference).resolve()
    if ref.parent.name != state['run_id'] or not ref.name.startswith('result_'):
        raise ValueError('Recovery receipt must be from the same run')
    r=read(ref)
    if 'camera' not in r or 'state' not in r:
        raise ValueError('Recovery needs an observation, not a movement result without camera')
    previous=state.get('last_result')
    if previous and ref.stat().st_mtime_ns <= Path(previous).stat().st_mtime_ns:
        raise ValueError('Recovery observation is not newer')
    state.update(phase='ready', pending=None, last_result=str(ref), revision=state['revision']+1)
    state['completed']=[] # Object may have moved; do not preserve previous success chain.
    write(folder/'state.json',state);event(folder,'reconciled',reason=reason,result=str(ref))
    return state
