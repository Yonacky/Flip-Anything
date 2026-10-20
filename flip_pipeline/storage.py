"""Content-addressed evidence and atomic state storage (not a security boundary)."""
import hashlib, json, math, os, re, uuid
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone

def utcnow(): return datetime.now(timezone.utc).isoformat()
def canonical(value): return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf8")
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    tmp.write_bytes(canonical(value)); tmp.replace(path)
def identity(value):
    if not isinstance(value,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}",value): raise ValueError("Invalid identity")
    return value

def inside(root,path):
    root=Path(root).resolve(); p=Path(path)
    p=(root/p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(root): raise ValueError("Path outside workspace")
    return p

def evidence(root,paths):
    out=[]
    for p in paths:
        p=inside(root,p)
        if not p.is_file(): raise ValueError("Missing evidence: "+str(p))
        out.append({"path":p.relative_to(Path(root).resolve()).as_posix(),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    if not out: raise ValueError("Evidence cannot be empty")
    return out

def verify_evidence(root,refs):
    if not refs: raise ValueError("Missing evidence")
    for ref in refs:
        p=inside(root,ref["path"])
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=ref["sha256"]: raise ValueError("Evidence changed/missing: "+str(p))

@contextmanager
def locked(path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(str(path),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd,str(os.getpid()).encode());os.close(fd);yield
    finally:path.unlink(missing_ok=True)
