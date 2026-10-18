"""
Traceability. Every stage records WHO produced an artefact from WHAT:
  - git commit / branch / dirty flag of the code
  - the exact command line and parsed arguments
  - a fingerprint of every input file (size, mtime, hash of the first and last MiB: fast even for a 50 GB bag)
  - library versions, host, UTC time

Checkpoints carry the whole chain in ck['provenance'] (build -> train -> calibrate). Every CSV / JSON output gets a
sidecar '<output>.meta.json' with the run record plus the checkpoint's chain, so any score can be traced back to the
code, data and settings that produced it.
"""
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys

def _find_repo() -> str:
    """Nearest ancestor of this file that contains a .git directory, else the two-levels-up default."""
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(d, '.git')):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        d = parent


_REPO = _find_repo()


def _git(*args) -> str:
    try:
        return subprocess.run(['git', *args], cwd=_REPO, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ''


def git_info() -> dict:
    status = _git('status', '--porcelain', '--untracked-files=no')
    return dict(commit=_git('rev-parse', 'HEAD') or 'unknown', branch=_git('rev-parse', '--abbrev-ref', 'HEAD'),
                describe=_git('describe', '--tags', '--always', '--dirty'), dirty=bool(status),
                dirty_files=status.splitlines()[:50])


def fingerprint(path: str, chunk: int = 1 << 20) -> dict:
    if not path or not os.path.exists(path):
        return dict(path=path, exists=False)
    st = os.stat(path)
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        h.update(f.read(chunk))
        if st.st_size > 2 * chunk:
            f.seek(-chunk, os.SEEK_END)
            h.update(f.read(chunk))
    return dict(path=os.path.abspath(path), exists=True, size=st.st_size,
                mtime=datetime.datetime.utcfromtimestamp(st.st_mtime).isoformat() + 'Z', sha1_head_tail=h.hexdigest())


def _versions() -> dict:
    out = dict(python=platform.python_version())
    for name in ('numpy', 'torch', 'torchvision', 'cv2', 'scipy', 'h5py', 'pandas'):
        mod = sys.modules.get(name)
        if mod is not None:
            out[name] = getattr(mod, '__version__', '?')
    return out


def run_record(stage: str, args=None, inputs=None, **extra) -> dict:
    """Description of this run. inputs: {'role': path}."""
    rec = dict(stage=stage, time_utc=datetime.datetime.utcnow().isoformat() + 'Z', argv=list(sys.argv),
               args={k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v))
                     for k, v in (vars(args) if args is not None else {}).items()},
               git=git_info(), host=platform.node(), versions=_versions(),
               inputs={k: fingerprint(v) for k, v in (inputs or {}).items()})
    rec.update(extra)
    return rec


def append(ck: dict, record: dict) -> dict:
    """Add a stage record to a checkpoint's provenance chain (in place)."""
    ck['provenance'] = list(ck.get('provenance') or []) + [record]
    return ck


def write_meta(output_path: str, record: dict, chain=None) -> str:
    path = output_path + '.meta.json'
    with open(path, 'w') as f:
        json.dump(dict(run=record, checkpoint_provenance=list(chain or [])), f, indent=2, default=str)
    return path


def read_meta(output_path: str):
    path = output_path + '.meta.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def banner(record: dict) -> str:
    g = record.get('git', {})
    return (f"[PROVENANCE] {record.get('stage')} | git {g.get('describe') or g.get('commit', '')[:10]} "
            f"on {g.get('branch')}{' (DIRTY: uncommitted changes)' if g.get('dirty') else ''} | {record.get('time_utc')}")
