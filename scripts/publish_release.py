"""Maintainer: publish an exact-byte source manifest with an explicit evidence allowlist."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
CODE_DIRS=('src','ds_control','flip_pipeline','pipeline_tests','skill_harness','skills','scripts','tests','docs','projects','environments','knowledge','devices','configs')
EXTENSIONS={'.py','.ps1','.md','.json','.toml','.txt','.yaml','.yml','.png','.jpg','.jpeg'}
TRANSIENT={'devices/active_calibration_task.json','ds_control/active.json','ds_control/scene/current.json'}

def files_for_release(root):
    from ds_context import path_in
    selected=set()
    for name in CODE_DIRS:
        for path in (root/name).rglob('*'):
            rel=path.relative_to(root).as_posix()
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:continue
            if any(part in {'__pycache__','.venv','.venv-pipeline','logs','Log','runs'} or part.endswith('.egg-info') for part in path.relative_to(root).parts):continue
            if rel in TRANSIENT or path.name in {'local.json','session.lock'}:continue
            path_in(root,rel)
            selected.add(rel)
    selected.update(p.name for p in root.iterdir() if p.is_file() and (p.suffix in {'.ps1','.md','.txt','.toml'} or p.name in {'.gitignore','.gitattributes'}))
    if (root/'.github').exists():
        selected.update(p.relative_to(root).as_posix() for p in (root/'.github').rglob('*') if p.is_file())
    for manifest_name in ['context-release.json','devices/calibration_release.json']:
        manifest=json.loads((root/manifest_name).read_text(encoding='utf-8-sig'))
        selected.add(manifest_name)
        for item in manifest.get('files',manifest.get('evidence',[])):
            path=path_in(root,item['path'])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Evidence changed or missing: '+item['path'])
            selected.add(item['path'])
    # Historical scripts remain available under calibration; exclude live scene/session state.
    for path in (root/'calibration').glob('*'):
        if path.is_file() and path.suffix in {'.json','.md','.py'} and path.name not in {'local.json','scene_current.json','session_state.json'}:
            selected.add(path.relative_to(root).as_posix())
    return sorted(selected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',required=True)
    args=parser.parse_args()
    paths=files_for_release(ROOT)
    manifest={'schema_version':2,'release':args.release,'workspace_variant':'canonical',
              'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
              'evidence_policy':'Only hashed calibration/context references under ignored logs/workspace are part of the source release.',
              'hardware_validation':'No physical experiment performed by this repository audit.'}
    (ROOT/'source-release.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'release':args.release,'files':len(paths),'hardware_contacted':False}))

if __name__=='__main__':main()
