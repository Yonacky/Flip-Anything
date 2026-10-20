import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flip_pipeline.cli import main
if __name__=="__main__":
    try:main(ROOT)
    except Exception as exc:raise SystemExit(type(exc).__name__+": "+str(exc))
