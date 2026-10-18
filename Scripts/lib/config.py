"""
The single configuration file (config.toml) shared by every stage, tool and the run script.

  cfg = config.load()              # --config / $RACING_CONFIG / <repo>/config.toml
  cfg.path('cache')                # a [paths] entry as an absolute path
  cfg['train']['epochs']           # any other value

Scripts build their argparse defaults from it (config.parser), so a command-line flag still overrides one value.
A file may start with  extends = "base.toml"  (relative to itself): the base is loaded first and overridden key by key.
Relative paths are resolved against [paths].root ("" = repository root); $VAR / ${VAR} and ~ are expanded.
Standard library only (tomllib, Python >= 3.11), so the tools can use it without torch.
"""
import argparse
import copy
import os
import tomllib

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DEFAULT_FILE = os.path.join(REPO, 'config.toml')
ENV_VAR = 'RACING_CONFIG'


def _expand(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(str(p)))


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(path: str, seen=()) -> dict:
    path = os.path.abspath(path)
    if path in seen:
        raise SystemExit(f'[CONFIG] "extends" loop at {path}')
    with open(path, 'rb') as f:
        data = tomllib.load(f)
    base = data.pop('extends', None)
    if base:
        data = _merge(_read(os.path.join(os.path.dirname(path), _expand(base)), seen + (path,)), data)
    return data


class Config:
    def __init__(self, data: dict, file: str):
        self.data, self.file = data, file
        root = self.data.get('paths', {}).get('root') or ''
        self.root = os.path.normpath(os.path.join(REPO, _expand(root))) if root else REPO

    def __getitem__(self, section: str) -> dict:
        return self.data.get(section, {})

    def get(self, section: str, key: str, default=None):
        return self.data.get(section, {}).get(key, default)

    def resolve(self, p) -> str:
        """Path relative to [paths].root -> absolute ('' stays '')."""
        if not p:
            return ''
        p = _expand(p)
        return os.path.normpath(p if os.path.isabs(p) else os.path.join(self.root, p))

    def path(self, key: str) -> str:
        return self.resolve(self.data.get('paths', {}).get(key, ''))

    def record(self) -> dict:
        """What goes into provenance records: the file and every resolved value."""
        return dict(file=self.file, root=self.root, values=copy.deepcopy(self.data))


def load(path: str = None) -> Config:
    path = path or os.environ.get(ENV_VAR) or DEFAULT_FILE
    if not os.path.exists(path):
        raise SystemExit(f'[CONFIG] config file not found: {path}')
    return Config(_read(path), os.path.abspath(path))


def bootstrap(argv=None) -> Config:
    """Load the config named by --config (pre-parsed, so the real parser can take its defaults from it)."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    ns, _ = pre.parse_known_args(argv)
    return load(ns.config)


def parser(doc: str, cfg: Config) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=cfg.file,
                    help=f'configuration file (default: ${ENV_VAR} or config.toml; now {cfg.file})')
    return ap


def require_file(path: str, what: str, hint: str = '') -> str:
    if not path or not os.path.exists(path):
        raise SystemExit(f'[ERROR] {what} not found: {path!r}' + (f' ({hint})' if hint else ''))
    return path
