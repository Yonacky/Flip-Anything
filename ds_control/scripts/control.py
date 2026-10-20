import argparse,json,os,runpy,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT.parent))
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))

def atomic(p,obj):
    t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    t.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False),encoding='utf8');t.replace(p)

def send(run,payload,timeout):
    if not (run/'commands').is_dir(): raise RuntimeError('Run not ready')
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf8'))
    if manifest['status']!='running' or (run/'STOP').exists(): raise RuntimeError('Run stopped')
    ident=time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex
    atomic(run/'commands'/(ident+'.json'),payload)
    result=run/('result_'+ident+'.json')
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if result.exists():
            data=json.loads(result.read_text(encoding='utf8'))
            print(json.dumps({'command_id':ident,'result_file':str(result),'result':data},ensure_ascii=False))
            return 0 if data.get('ok') else 1
        time.sleep(.2)
    raise TimeoutError(f'Outcome unknown; do not resend. Inspect {result}')

def main():
    p=argparse.ArgumentParser(description='DS robot interface; start owns hardware, other commands use files')
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('doctor');sub.add_parser('start')
    for a in ('observe','status','shutdown','stop','gripper-check','activate-empty'):sub.add_parser(a)
    q=sub.add_parser('execute');q.add_argument('file');q.add_argument('--from-result',required=True)
    args=p.parse_args()
    if args.action in ('doctor','start'):
        from flip_pipeline.release import verify_release
        verify_release(ROOT.parent)
    if args.action=='doctor':
        import importlib
        for mod in ('numpy','cv2','xarm.wrapper','pyorbbecsdk','flip_anything.camera'): importlib.import_module(mod)
        from flip_anything.config import load_config
        cfg=load_config(ROOT/'configs/platform.json')
        print(json.dumps({'offline_check':'passed','python':sys.executable,'config':cfg,'hardware_connected':False},ensure_ascii=False));return 0
    if args.action=='start':
        lock=ROOT/'session.lock'
        fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        os.write(fd,str(os.getpid()).encode());os.close(fd)
        try:
            from flip_anything.config import load_config
            from hardware_guard import device_lease
            with device_lease(load_config(ROOT/'configs/platform.json')):
                rid='DS_'+time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
                atomic(ROOT/'active.json',{'run':rid,'pid':os.getpid()})
                sys.argv=['session.py','--run-id',rid]
                runpy.run_path(str(ROOT/'scripts/session.py'),run_name='__main__')
        finally:lock.unlink(missing_ok=True)
        return 0
    from flip_anything.config import load_config
    rid=json.loads((ROOT/'active.json').read_text(encoding='utf8'))['run']
    run=Path(load_config(ROOT/'configs/platform.json')['recording']['root'])/rid
    if args.action=='stop':
        (run/'STOP').touch();print('Stop requested; verify controller feedback. This is not a hardware emergency stop.');return 0
    payload={'action':args.action.replace('-','_')}
    if args.action=='activate-empty': payload={'action':'gripper_activate','object_held':False}
    if args.action=='execute':
        payload=json.loads(Path(args.file).read_text(encoding='utf-8-sig'))
        if payload.get('action') not in ('move','grip'):raise ValueError('execute supports move/grip only')
        ref=Path(args.from_result).resolve()
        if ref.parent != run.resolve() or not ref.name.startswith('result_'): raise ValueError('Use a result from the active run')
        prior=json.loads(ref.read_text(encoding='utf8'))
        if not prior.get('ok') or 'state' not in prior:raise ValueError('Reference must be a successful state result')
        if 'pipeline_ticket' not in payload:raise ValueError('Use pipeline.ps1 authorize or authorize-acquisition first')
        if payload.get('expected_pose')!=prior['state']['pose']:raise ValueError('Use exactly the receipt bound at authorization')
        if not isinstance(payload.get('grip_required'),bool):raise ValueError('Declare grip_required true/false explicitly')
    return send(run,payload,120)

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:print(f'{type(e).__name__}: {e}',file=sys.stderr);sys.exit(1)

