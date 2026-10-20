"""Verify published workspace sources before opening a hardware connection."""
from pathlib import Path
import hashlib
from .storage import read,inside

def verify_release(root):
    root=Path(root);release=read(root/"source-release.json")
    if not release.get("sha256"):raise ValueError("Published source manifest missing")
    for name,expected in release["sha256"].items():
        p=inside(root,name)
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError("Published file changed: "+name)
    return release["release"]
