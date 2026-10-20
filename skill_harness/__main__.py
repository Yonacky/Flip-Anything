import argparse,json,os,sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from contextlib import contextmanager
from pathlib import Path
from .runtime import ROOT,read,write,library,retrieve,new_episode,run_skill,assess,reconcile,ExistingControl
from .learning import propose,promote
from .catalog import load_catalog

@contextmanager
def locked(folder):
    lock=Path(folder)/'.harness.lock'
    fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:yield
    finally:lock.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser(description='Skill retrieval, guarded execution and evidence-based experience curation')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('catalog');q.add_argument('--layer',choices=('all','goals','plans','capabilities'),default='all')
    q=sub.add_parser('retrieve');q.add_argument('facts');q.add_argument('--include-candidates',action='store_true')
    q=sub.add_parser('init');q.add_argument('episode');q.add_argument('--run-id',required=True);q.add_argument('--plan',default='pick_invert_place');q.add_argument('--device-profile')
    q=sub.add_parser('run');q.add_argument('episode');q.add_argument('skill');q.add_argument('--input',required=True);q.add_argument('--hardware',action='store_true');q.add_argument('--allow-candidate',action='store_true');q.add_argument('--supervised',action='store_true')
    q=sub.add_parser('assess');q.add_argument('episode');q.add_argument('--facts',required=True);q.add_argument('--supervised',action='store_true')
    q=sub.add_parser('reconcile');q.add_argument('episode');q.add_argument('--result',required=True);q.add_argument('--reason',required=True)
    q=sub.add_parser('propose');q.add_argument('episode');q.add_argument('--lesson',required=True)
    q=sub.add_parser('promote');q.add_argument('candidate');q.add_argument('--validation',required=True)
    a=p.parse_args()
    if a.command=='catalog':
        layers=load_catalog();r=layers if a.layer=='all' else list(layers[a.layer].values())
    elif a.command=='retrieve':
        facts=read(a.facts);r=retrieve([k for k,v in facts.items() if v.get('value') is True],a.include_candidates)
    elif a.command=='init':r=new_episode(a.episode,a.run_id,a.plan,a.device_profile)
    elif a.command in ('run','assess','reconcile'):
        with locked(a.episode):
            if a.command=='run':
                if not a.hardware:raise ValueError('No hardware dispatched. Use --hardware for explicitly authorized execution')
                data=read(a.input);state=read(Path(a.episode)/'state.json')
                r=run_skill(a.episode,a.skill,data['payload'],data.get('facts',{}),data.get('reference'),ExistingControl(state['run_id']),a.allow_candidate,a.supervised)
            elif a.command=='assess':r=assess(a.episode,read(a.facts),a.supervised)
            else:r=reconcile(a.episode,a.result,a.reason)
    elif a.command=='propose':r=propose(a.episode,read(a.lesson))
    else:r=promote(a.candidate,a.validation)
    print(json.dumps(r,ensure_ascii=False,indent=2))

if __name__=='__main__':
    try:main()
    except Exception as e:raise SystemExit(f'{type(e).__name__}: {e}')
