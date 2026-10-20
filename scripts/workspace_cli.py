"""Offline setup and experiment bookkeeping. Never opens hardware."""
import argparse,importlib,json,re,sys,time,uuid,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from skill_harness.runtime import new_episode,read,write
from skill_harness.catalog import load_catalog

def inside(path):
    result=Path(path).resolve()
    if not result.is_relative_to(ROOT.resolve()):raise ValueError('Path outside student workspace: '+str(result))
    return result

def identity(value,prefix):
    value=value or prefix+'_'+time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}',value):raise ValueError('Invalid experiment identity')
    return value

def new_experiment(plan='pick_invert_place',task_id=None,object_id=None):
    # Validate all inputs before creating any episode state.
    load_catalog()
    release=read(ROOT/'source-release.json')
    active=read(ROOT/'ds_control/active.json');run_id=identity(active['run'],'DS')
    cfg_path=ROOT/'ds_control/configs/platform.json'
    cfg=read(cfg_path);run=inside(cfg_path.parent/cfg['recording']['root']/run_id)
    manifest=read(run/'manifest.json')
    if manifest.get('run_id')!=run_id or manifest.get('status')!='running' or (run/'STOP').exists():
        raise ValueError('Start a new hardware run before creating an episode')
    if manifest.get('backend')!='hardware':raise ValueError('This entrypoint requires a real hardware run')
    for old in (ROOT/'logs/episodes').glob('*/state.json'):
        if read(old).get('run_id')==run_id:raise ValueError('Run already has an episode: '+str(old.parent))
    if bool(task_id)!=bool(object_id):raise ValueError('To continue the same object after a run restart, supply both TaskId and ObjectId')
    if task_id:
        matches=[read(p) for p in (ROOT/'logs/episodes').glob('*/state.json')]
        if not any(s.get('task_id')==task_id and s.get('object_instance_id')==object_id for s in matches):
            raise ValueError('TaskId/ObjectId must identify an existing task in this workspace')
    task_id=identity(task_id,'TASK');object_id=identity(object_id,'OBJECT');episode_id=identity(None,'EP')
    folder=ROOT/'logs/episodes'/episode_id
    device=ROOT/'devices/current.json'
    state=new_episode(folder,run_id,plan,device)
    write(folder/'source_release.json',release)
    state.update(task_id=task_id,object_instance_id=object_id,source_release='source_release.json')
    write(folder/'state.json',state)
    scene=read(folder/'scene.json');scene.update(object_ref=object_id,task_id=task_id,run_id=run_id)
    write(folder/'scene.json',scene);write(ROOT/'ds_control/scene/current.json',scene)
    analysis=ROOT/'workspace'/episode_id;analysis.mkdir(parents=True,exist_ok=False)
    profile=read(folder/'device_profile.json')
    summary={'schema_version':1,'task_id':task_id,'object_instance_id':object_id,'plan_id':plan,'run_id':run_id,
             'object_description':None,'physical_result':'unknown','autonomy_result':'unknown',
             'human_interventions':[],'failures':[],'recoveries':[],'unresolved':[],
             'evidence':{'support':[],'release':[],'stability':[]},'elapsed_s':None,
             'device_profile_id':profile['id'],'device_profile_version':profile['version'],
             'device_profile_sha256':hashlib.sha256((folder/'device_profile.json').read_bytes()).hexdigest(),
             'catalog_snapshot':'catalog_snapshot.json','source_release':'source_release.json','release_id':release['release']}
    write(folder/'summary.json',summary)
    result={'episode':str(folder),'analysis_directory':str(analysis),'task_id':task_id,'object_instance_id':object_id,
            'run_id':run_id,'plan_id':plan,'hardware_started_by_this_command':False}
    write(ROOT/'active_episode.json',result)
    return result

def check():
    prefix=inside(sys.prefix)
    if prefix!=ROOT/'.venv':raise ValueError('Use the student workspace Python')
    origins={}
    for name in ('flip_anything','skill_harness','numpy','cv2','xarm.wrapper','pyorbbecsdk'):
        module=importlib.import_module(name);origins[name]=str(inside(module.__file__))
    from flip_anything.config import load_config
    cfg=load_config(ROOT/'ds_control/configs/platform.json')
    for path in (cfg['recording']['root'],cfg['camera']['log_dir'],cfg['calibration']['file']):inside(path)
    load_catalog()
    release=read(ROOT/'source-release.json')
    for rel,sha in release['sha256'].items():
        if hashlib.sha256(inside(ROOT/rel).read_bytes()).hexdigest()!=sha:
            raise ValueError('Frozen runtime/library changed: '+rel)
    return {'offline_check':'passed','workspace':str(ROOT),'python':sys.executable,'prefix':str(prefix),
            'module_origins':origins,'recording_root':cfg['recording']['root'],'sdk_logs':cfg['camera']['log_dir'],
            'device_profile_status':read(ROOT/'devices/current.json')['status'],
            'release':release['release'],'hardware_connected':False}

def main():
    parser=argparse.ArgumentParser();subs=parser.add_subparsers(dest='command',required=True)
    subs.add_parser('check');new=subs.add_parser('new-experiment')
    new.add_argument('--plan',default='pick_invert_place');new.add_argument('--task-id');new.add_argument('--object-id')
    a=parser.parse_args()
    result=check() if a.command=='check' else new_experiment(a.plan,a.task_id,a.object_id)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
    try:main()
    except Exception as exc:raise SystemExit(type(exc).__name__+': '+str(exc))
