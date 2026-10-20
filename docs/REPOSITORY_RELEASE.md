# GitHub checkout and maintenance

This repository contains the execution code, project briefs, versioned environment data, knowledge cards and the exact evidence files required by the published calibration. Full experimental recordings, virtual environments, maintenance backups and downloaded papers stay local.

Use Python 3.12 for the compute environment:

```powershell
python -m venv .venv-pipeline
.\.venv-pipeline\Scripts\python.exe -m pip install -r requirements-pipeline-lock.txt
.\.venv-pipeline\Scripts\python.exe scripts/check_repository.py
.\context.ps1 context --project flip_anything
```

The checker runs without a robot SDK or hardware connection. It verifies source/context hashes, published calibration geometry and all three offline test suites. GitHub Actions runs the same command. Hardware dependencies remain in the separate `.venv` environment; installing compute dependencies does not start devices.

`source-release.json` identifies the canonical runtime. `context-release.json` identifies project data. Rebuild the context manifest after reviewed context edits, then rebuild the source manifest after reviewed source edits:

```powershell
.\.venv-pipeline\Scripts\python.exe scripts/publish_context.py --release <context-version>
.\.venv-pipeline\Scripts\python.exe scripts/publish_release.py --release <source-version>
.\.venv-pipeline\Scripts\python.exe scripts/check_repository.py
```

A manifest is an integrity record, not proof that unperformed physical tests passed. Current measured geometry remains accepted for supervised_rigid_flip; extra lift/rotation calibration tests are not prerequisites.

Only the paths named by the source manifest are required in a clean checkout. A small set under logs/ and workspace/ is intentionally tracked as immutable published evidence; do not force-add those directories wholesale. Historical absolute paths inside evidence are provenance, not runtime configuration. `.gitattributes` preserves exact bytes so clone/checkout does not invalidate hashes through line-ending conversion.

The former ORA agent wrapper is explicitly retired; current entrypoints are context.ps1, pipeline.ps1 and robot.ps1. The old skill harness remains available for historical interpretation and offline tests. devices/current.json is a derived compatibility snapshot for its material bookkeeping, with the current calibration release as its source.
