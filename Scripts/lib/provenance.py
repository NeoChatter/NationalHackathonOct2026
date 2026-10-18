"""
Traceability. Every stage records WHO produced an artefact from WHAT:
  - git commit / branch / dirty flag of the code
  - the exact command line, parsed arguments and the resolved config.toml values
  - a fingerprint of every input file: size, mtime, SHA-1 of the first and last MiB (fast), and - with
    [provenance].full_sha256 - the full SHA-256, cached per (path, size, mtime) so a 7 GB bag is hashed once
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

_HASH = dict(full_sha256=False, cache_path='')


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


def configure(cfg) -> None:
    """Take the hashing options from config.toml ([provenance].full_sha256, [paths].hash_cache)."""
    _HASH.update(full_sha256=bool(cfg.get('provenance', 'full_sha256', False)), cache_path=cfg.path('hash_cache'))


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')


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


def _sha256_full(path: str, st) -> str:
    key = f'{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}'
    cache_path = _HASH['cache_path']
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except (OSError, ValueError):
            cache = {}
    if key in cache:
        return cache[key]
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    digest = h.hexdigest()
    if cache_path:
        cache[key] = digest
        try:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            tmp = cache_path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(cache, f, indent=1)
            os.replace(tmp, cache_path)
        except OSError:
            pass
    return digest


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
    out = dict(path=os.path.abspath(path), exists=True, size=st.st_size,
               mtime=datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat().replace('+00:00', 'Z'),
               sha1_head_tail=h.hexdigest())
    if _HASH['full_sha256']:
        out['sha256'] = _sha256_full(path, st)
    return out


def _versions() -> dict:
    out = dict(python=platform.python_version())
    for name in ('numpy', 'torch', 'torchvision', 'cv2', 'scipy', 'h5py', 'pandas'):
        mod = sys.modules.get(name)
        if mod is not None:
            out[name] = getattr(mod, '__version__', '?')
    return out


def run_record(stage: str, args=None, inputs=None, cfg=None, **extra) -> dict:
    """Description of this run. inputs: {'role': path}."""
    rec = dict(stage=stage, time_utc=_now(), argv=list(sys.argv),
               args={k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v))
                     for k, v in (vars(args) if args is not None else {}).items()},
               git=git_info(), host=platform.node(), versions=_versions(),
               inputs={k: fingerprint(v) for k, v in (inputs or {}).items()})
    if cfg is not None:
        rec['config'] = cfg.record()
    rec.update(extra)
    return rec


def append(ck: dict, record: dict) -> dict:
    """Add a stage record to a checkpoint's provenance chain (in place)."""
    ck['provenance'] = list(ck.get('provenance') or []) + [json.loads(json.dumps(record, default=str))]
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
