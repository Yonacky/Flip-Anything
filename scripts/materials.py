"""Local experiment materials. No SDK imports, no hardware commands, no skill editing."""
import argparse, hashlib, json, os, re, time, uuid
from pathlib import Path
from contextlib import contextmanager
ROOT = Path(__file__).resolve().parents[1]

def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def write(p, obj):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def safe(p):
    p=Path(p); p=p if p.is_absolute() else ROOT/p
    p=Path(os.path.abspath(p)); base=ROOT.resolve()
    if not p.is_relative_to(base): raise ValueError('File must be inside workspace')
    for q in [p,*p.parents]:
        if q==base: break
        if q.exists() and (q.is_symlink() or (getattr(q.stat(follow_symlinks=False),'st_file_attributes',0)&0x400)):
            raise ValueError('Linked paths are not material inputs')
    if not p.resolve().is_relative_to(base):raise ValueError('Resolved path outside workspace')
    return p

def run_root():
    cfg=ROOT/'ds_control/configs/platform.json'
    return safe(cfg.parent/read(cfg)['recording']['root'])

@contextmanager
def lock():
    p=ROOT/'logs/tasks/.writer.lock';p.parent.mkdir(parents=True,exist_ok=True)
    for _ in range(100):
        try: fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY);break
        except FileExistsError: time.sleep(.02)
    else:raise RuntimeError('Materials writer busy; do not resend a robot action')
    try: yield
    finally:os.close(fd);p.unlink()

def active():
    ptr=ROOT/'active_task.json'
    if not ptr.exists():return None,None
    folder=safe(read(ptr)['folder']); state=read(folder/'task.json')
    return folder,state

def begin():
    folder,state=active()
    if state and state['status']=='open':raise ValueError('An open task exists; continue it or finish it first')
    if (ROOT/'ds_control/session.lock').exists():raise ValueError('Begin before starting hardware; an existing session needs review')
    ident='TASK_'+time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    folder=ROOT/'logs/tasks'/ident
    state={'task_id':ident,'object_id':'OBJECT_'+uuid.uuid4().hex[:12],'status':'open',
           'created_at_unix':time.time(),'runs':[],'excluded_runs':[p.name for p in run_root().glob('DS_*') if p.is_dir()],
           'assessment':'Model reports and raw evidence; not independent success certification'}
    write(folder/'task.json',state)
    for src,dst in [('devices/current.json','device_profile.json'),('source-release.json','source_release.json')]:
        write(folder/dst,read(ROOT/src))
    template={'physical_result':'unknown','autonomy_result':'unknown','object_description':'',
              'human_interventions':[],'failures':[],'recoveries':[],'unresolved':[],
              'evidence':{'support':[],'release':[],'stability':[]},'artifacts':[]}
    write(folder/'summary-draft.json',template)
    write(ROOT/'active_task.json',{'folder':str(folder.relative_to(ROOT)),'task_id':ident})
    return {'task':str(folder),'summary_template':str(folder/'summary-draft.json'),'hardware_started':False}

def collect():
    folder,state=active()
    if not state or state['status']!='open':return {'materials':'no open task'}
    # Runs created after begin belong to this single-owner workspace task. No old run is inherited.
    known={x['run_id'] for x in state['runs']}
    for p in sorted(run_root().glob('DS_*')):
        if p.name in state['excluded_runs'] or p.name in known or not (p/'manifest.json').exists():continue
        p=safe(p);m=read(p/'manifest.json')
        if m.get('run_id')!=p.name:raise ValueError('Run/manifest identity mismatch')
        state['runs'].append({'run_id':p.name,'path':str(p.relative_to(ROOT)),'backend':m.get('backend','unknown')})
    write(folder/'task.json',state)
    return {'task_id':state['task_id'],'runs':state['runs']}

def note(text, artifacts):
    collect();folder,state=active()
    if not state or state['status']!='open':raise ValueError('Begin a task first')
    refs=[]
    for value in artifacts:
        p=safe(value)
        if not p.is_file():raise ValueError('Missing artifact: '+value)
        refs.append(str(p.relative_to(ROOT)))
    with (folder/'notes.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'time_unix':time.time(),'text':text,'artifacts':refs},ensure_ascii=False)+'\n')
    return {'recorded':True}

def finish(summary):
    collect();folder,state=active()
    if not state or state['status']!='open':raise ValueError('No open task')
    if (ROOT/'ds_control/session.lock').exists():raise ValueError('End the hardware session before finishing materials')
    data=read(safe(summary))
    allowed={'success','failure','incomplete','unknown','stopped'}
    for key in ('physical_result','autonomy_result'):
        if data.get(key) not in allowed:raise ValueError('Invalid '+key)
    for key in ('human_interventions','failures','recoveries','unresolved','artifacts'):
        if not isinstance(data.get(key),list):raise ValueError(key+' must be a list')
    if data['autonomy_result']=='success' and (data['physical_result']!='success' or data['human_interventions']):
        raise ValueError('Autonomous success conflicts with outcome/interventions')
    evidence=data.get('evidence',{})
    for key in ('support','release','stability'):
        if not isinstance(evidence.get(key),list):raise ValueError('Missing evidence list: '+key)
        if data['physical_result']=='success' and not evidence[key]:raise ValueError('Success needs '+key+' evidence; otherwise report unknown')
    paths=set()
    for value in data['artifacts']+sum([evidence[k] for k in ('support','release','stability')],[]):
        p=safe(value)
        if not p.is_file():raise ValueError('Missing evidence: '+value)
        paths.add(p)
    # Include the complete raw runs, not just selected successful frames.
    for run in state['runs']:
        d=safe(run['path']);m=read(d/'manifest.json')
        if m.get('status')=='running':raise ValueError('Run still marked running; report interruption before finishing')
        for p in d.rglob('*'):
            p=safe(p)
            if p.is_file():paths.add(p)
    if data['physical_result']=='success' and not state['runs']:raise ValueError('Success needs a recorded run')
    if data['physical_result']=='success' and any(x['backend']!='hardware' for x in state['runs']):
        raise ValueError('Simulated results cannot certify a hardware success')
    notes=folder/'notes.jsonl'
    if notes.exists():
        for line in notes.read_text(encoding='utf-8').splitlines():
            for value in json.loads(line)['artifacts']:paths.add(safe(value))
    for p in paths:
        if not p.is_file():raise ValueError('Missing material: '+str(p))
    data.update(task_id=state['task_id'],object_id=state['object_id'],runs=state['runs'],
                elapsed_s=time.time()-state['created_at_unix'],assessment=state['assessment'])
    write(folder/'summary.json',data)
    paths.update([folder/'summary.json',folder/'device_profile.json',folder/'source_release.json'])
    if notes.exists():paths.add(notes)
    index=[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)]
    write(folder/'materials.json',{'task_id':state['task_id'],'files':index,'purpose':'maintainer review; no skill proposal or promotion'})
    state.update(status='closed',finished_at_unix=time.time());write(folder/'task.json',state)
    return {'materials':str(folder/'materials.json'),'summary':str(folder/'summary.json'),'files':len(index),'skills_changed':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('begin');sub.add_parser('collect')
    n=sub.add_parser('note');n.add_argument('text');n.add_argument('--artifact',action='append',default=[])
    f=sub.add_parser('finish');f.add_argument('--summary',required=True)
    a=p.parse_args()
    with lock():
        result=begin() if a.cmd=='begin' else collect() if a.cmd=='collect' else note(a.text,a.artifact) if a.cmd=='note' else finish(a.summary)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
    try:main()
    except Exception as e:raise SystemExit(type(e).__name__+': '+str(e))
