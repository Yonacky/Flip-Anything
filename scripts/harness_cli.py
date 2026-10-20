import runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if len(sys.argv)>1 and sys.argv[1] in ('promote','propose'):
    raise SystemExit('DS prepares experiment materials only. Skill extraction and maintenance belong to the maintainer. Use experiment.ps1 to record materials.')
runpy.run_module('skill_harness',run_name='__main__')
