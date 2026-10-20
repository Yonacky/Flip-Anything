import hashlib
import importlib.metadata
import json
import platform
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def stamp():
    return {"utc": datetime.now(timezone.utc).isoformat(), "host_monotonic_ns": time.monotonic_ns()}


def software_versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("flip-anything", "xarm-python-sdk", "pyorbbecsdk2", "numpy", "opencv-python"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    source = Path(__file__).parent
    digest = hashlib.sha256()
    for file in sorted(source.glob("*.py")):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    result["source_sha256"] = digest.hexdigest()
    return result


class Trial:
    def __init__(self, root, config, backend, run_id=None, context=None):
        run_id = run_id or (datetime.now(timezone.utc).strftime("RUN_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id):
            raise ValueError("run_id must be 1..96 letters, digits, underscores or hyphens")
        self.path = Path(root) / run_id
        if self.path.exists() and any(self.path.iterdir()):
            raise FileExistsError(f"Run directory already contains files: {self.path}")
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest = {"schema_version": 1, "run_id": run_id, "backend": backend,
                         "started": stamp(), "config": config, "software": software_versions(),
                         "context": context or {}, "status": "running", "task_outcome": "not_evaluated"}
        self._write_manifest()
        self.events = (self.path / "events.jsonl").open("x", encoding="utf-8")
        self.event("trial_started", backend=backend)

    def _write_manifest(self):
        temporary = self.path / "manifest.tmp"
        temporary.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.path / "manifest.json")

    def event(self, kind, **payload):
        row = {"schema_version": 1, "kind": kind, **stamp(), "payload": payload}
        self.events.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.events.flush()

    def finish(self, status, error=None):
        self.event("trial_finished", status=status, error=error)
        self.manifest.update(status=status, finished=stamp(), error=error)
        self._write_manifest()
        self.events.close()
