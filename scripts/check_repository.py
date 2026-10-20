"""Offline repository acceptance: manifests, published calibration, syntax and test suites."""
import argparse
import ast
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]


def verify_files():
    from flip_pipeline.release import verify_release
    from flip_pipeline.storage import read, verify_evidence
    from ds_context import load_context, path_in
    release=read(ROOT/'source-release.json')
    for name in release['sha256']:
        path_in(ROOT,name)
    verify_release(ROOT)
    context=load_context(ROOT,'flip_anything')
    index=read(ROOT/'devices/calibration_release.json')
    verify_evidence(ROOT,index['evidence'])
    from flip_pipeline.calibration import validate_device_profile
    validate_device_profile(read(path_in(ROOT,index['profile_path'])),read(ROOT/'devices/installation.json'),task_profile='supervised_rigid_flip')
    for name in release['sha256']:
        if name.endswith('.py'):
            ast.parse((ROOT/name).read_text(encoding='utf-8-sig'),filename=name)
    return {'source_release':release['release'],'source_files':len(release['sha256']),
            'calibration_status':context['pipeline_calibration_release_status'],'hardware_contacted':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--files-only',action='store_true')
    args=parser.parse_args()
    report=verify_files()
    if not args.files_only:
        total=0
        for directory in ['tests','pipeline_tests','skill_harness/tests']:
            print('Testing '+directory,flush=True)
            suite=unittest.TestLoader().discover(str(ROOT/directory))
            result=unittest.TextTestRunner(verbosity=1).run(suite)
            total+=result.testsRun
            if not result.wasSuccessful():raise SystemExit(1)
        report['tests_passed']=total
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
