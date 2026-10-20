"""One-time project relocation normalizer. Text-only, with per-file backup and audit."""
from pathlib import Path
import json,re,zipfile,hashlib
ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'logs/maintenance'
EXT={'.py','.ps1','.md','.txt','.json','.jsonl','.toml','.yaml','.yml','.csv','.pth','.egg-link'}
pairs=[('ds_control/runs','logs/experiments/ds_control'),('runs','logs/experiments/legacy'),('docs/日志','logs/conversations'),('ds_control/Log','logs/sdk/ds_control'),('Log','logs/sdk/root'),('output/dsh_review','logs/reviews/dsh_review'),('skill_harness/episodes','logs/episodes'),('session','archive/legacy_session'),('ds','archive/camera_diagnostics'),('acl_recovery','archive/acl_recovery')]

def normalize(text,rel):
    # Match complete directory components, including nested JSON escaping.
    for old,new in pairs:
        pattern = r"(?:F:([/\\]+)2610[/\\]+|G:([/\\]+))Flip(?: |%20)Anything[/\\]+" + r"[/\\]+".join(map(re.escape,old.split('/'))) + r"(?=$|[ /\\\"'`<>),;\]\r\n])"
        def full_path(m):
            sep=m.group(1) or m.group(2)
            base='G:/Flip%20Anything/' if '%20' in m.group(0) else 'G:/Flip Anything/'
            return (base+new).replace('/',sep)
        text=re.sub(pattern,full_path,text)
    text=re.sub(r"F:([/\\]+)2610[/\\]+Flip(?P<space> |%20)Anything(?=$|[ /\\\"'`<>),;\]\r\n])",lambda m:'G:'+m.group(1)+'Flip'+m.group('space')+'Anything',text)
    # Relative references in records and editor state get an unambiguous target.
    for old,new in [('docs/日志','logs/conversations'),('skill_harness/episodes','logs/episodes'),('ds_control/runs','logs/experiments/ds_control')]:
        pattern=r'(?<![\w/\\])'+r'[/\\]+'.join(map(re.escape,old.split('/')))+r'(?=$|[ /\\\"\'`<>),;\]\r\n])'
        text=re.sub(pattern,lambda m:new,text)
    kind='ds_control' if (rel.startswith('ds_control/') or rel.startswith('logs/experiments/ds_control/')) else 'legacy'
    def run_path(m):
        tail=m.group(1);destination='ds_control' if tail.startswith('DS_') else kind
        return 'G:/Flip Anything/logs/experiments/'+destination+'/'+tail
    text=re.sub(r'(?<![\w/\\])runs[/\\]+([A-Za-z0-9_.-]+)',run_path,text)
    # Root-derived expressions remain portable. Parenthesize to preserve method calls.
    if rel.startswith('ds_control/') and rel.endswith('.py'):
        text=re.sub(r'ROOT\s*/\s*[\'\"]runs[\'\"]',"(ROOT.parent / 'logs' / 'experiments' / 'ds_control')",text)
        text=re.sub(r'Path\([\'\"]runs[\'\"]\)',"(Path(__file__).resolve().parents[2] / 'logs' / 'experiments' / 'ds_control')",text)
    elif rel.startswith('scripts/') and rel.endswith('.py'):
        text=re.sub(r'ROOT\s*/\s*[\'\"]runs[\'\"]',"(ROOT / 'logs' / 'experiments' / 'legacy')",text)
        text=re.sub(r'Path\([\'\"]runs[\'\"]\)',"Path('logs/experiments/legacy')",text)
    return text



def main():
    AUDIT.mkdir(parents=True,exist_ok=True)
    backup=AUDIT/'paths-before-normalization.zip'
    # Resume keeps the first original for every file; never overwrite a backup.
    files=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in EXT:continue
        rel=p.relative_to(ROOT).as_posix()
        if rel.startswith(('logs/maintenance/','.git/','tests/','skill_harness/tests/')) or p.resolve()==Path(__file__).resolve():continue
        if '.venv/' in rel and not (p.suffix in ('.pth','.egg-link') or p.name=='direct_url.json'):continue
        files.append((p,rel))
    changes=[]
    with zipfile.ZipFile(backup,'a',compression=zipfile.ZIP_DEFLATED) as z:
        saved=set(z.namelist())
        for p,rel in files:
            raw=p.read_bytes()
            try:text=raw.decode('utf-8-sig')
            except UnicodeDecodeError:continue
            updated=normalize(text,rel)
            if text==updated:continue
            if rel not in saved:
                z.writestr(rel,raw);saved.add(rel)
            p.write_text(updated,encoding='utf8')
            changes.append({'file':rel,'before_sha256':hashlib.sha256(raw).hexdigest(),'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    with zipfile.ZipFile(backup) as z:
        renamed={'ds_control/verification.txt':'logs/maintenance/ds_control_initial_verification.txt'}
        changes=[{'file':renamed.get(rel,rel),'backup_entry':rel,'before_sha256':hashlib.sha256(z.read(rel)).hexdigest(),'after_sha256':hashlib.sha256((ROOT/renamed.get(rel,rel)).read_bytes()).hexdigest()} for rel in z.namelist()]
    (AUDIT/'path-normalization.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'changed_files':len(changes),'backup':str(backup)},ensure_ascii=False))

if __name__=='__main__':main()
