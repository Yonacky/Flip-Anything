"""Maintainer-only offline context manifest builder. Does not publish device calibration."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',required=True)
    args=parser.parse_args()
    paths=set()
    for directory in ['projects','environments','knowledge']:
        for path in (ROOT/directory).rglob('*'):
            if path.is_file():
                if not path.resolve().is_relative_to(ROOT):raise ValueError('External context file')
                paths.add(path.relative_to(ROOT).as_posix())
    paths.update(['DS_START.md','START_PROMPT.txt','AGENTS.md','SKILL.md','README.md',
                  'docs/PROJECT_LAYOUT.md','scripts/ds_context.py','scripts/publish_context.py',
                  'context.ps1','pipeline_tests/test_ds_context.py'])
    manifest={'schema_version':1,'release_id':args.release,
              'purpose':'project data and read-only loader; not device calibration release',
              'files':[{'path':p,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in sorted(paths)]}
    (ROOT/'context-release.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'release':args.release,'files':len(paths),'hardware_contacted':False}))

if __name__=='__main__':main()
