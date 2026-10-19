"""Experiment files: a matrix of runs.

    name: approaches-europeana
    defaults:                     # merged into every run
      train: {epochs: 2, lr: 1e-4}
      approach: {shuffle_options: true}
    matrix:                       # cartesian product; dotted keys set nested fields
      model: [gliner-decide, kev-4b]
      train.seed: [42, 7]
      variant:                    # keys starting with "variant": each value is a dict deep-merged into the run
        - {datasets: [boolq], approach: {noul: statement}}
        - {datasets: [sst5], approach: {score: as_choice}}
    include:                      # extra one-off runs (merged over defaults)
      - {model: plumb-4b, datasets: [boolq], approach: {noul: as_choice}}
    exclude:                      # drop matrix runs matching all of these keys
      - {model: gliner-decide, approach.score: as_choice}

Each run gets a stable name (``<model>__<datasets>__<approach>__s<seed>``) and ``out = runs/<sweep>/<name>``.
A run config file without ``matrix``/``include`` is a single run.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _set(d: dict, dotted: str, value) -> None:
    *path, last = dotted.split(".")
    for k in path:
        d = d.setdefault(k, {})
    d[last] = value


def _get(d: dict, dotted: str):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _short(v) -> str:
    if isinstance(v, list):
        return "+".join(_short(x) for x in v)
    if isinstance(v, dict):
        return v.get("name", "x")
    return str(v)


def run_name(rc: dict) -> str:
    model = rc["model"] if isinstance(rc["model"], str) else rc["model"].get("name", "model")
    ds = _short(rc["datasets"])
    ap = rc.get("approach") or {}
    ap_s = "-".join(f"{t[0]}{_short(ap[t])}" for t in ("choice", "score", "noul") if t in ap) or "default"
    seed = (rc.get("train") or {}).get("seed", 42)
    tag = f"__{rc['tag']}" if rc.get("tag") else ""
    return f"{model}__{ds}__{ap_s}{tag}__s{seed}"


def expand(path: str | Path) -> list[dict]:
    path = Path(path)
    spec = yaml.safe_load(path.read_text())
    sweep = spec.get("name", path.stem)
    defaults = spec.get("defaults") or {}
    runs: list[dict] = []
    if "matrix" in spec:
        keys = list(spec["matrix"])
        for combo in itertools.product(*(spec["matrix"][k] for k in keys)):
            rc = copy.deepcopy(defaults)
            for k, v in zip(keys, combo):
                if k.startswith("variant"):  # a bundle of settings, deep-merged into the run
                    rc = merge(rc, v)
                else:
                    _set(rc, k, copy.deepcopy(v))
            if any(all(_get(rc, k) == v for k, v in ex.items()) for ex in spec.get("exclude") or []):
                continue
            runs.append(rc)
    for extra in spec.get("include") or []:
        runs.append(merge(defaults, extra))
    if not runs and "model" in spec:
        runs.append({k: v for k, v in spec.items() if k not in ("defaults", "slurm")})
    seen = set()
    for rc in runs:
        rc.setdefault("name", run_name(rc))
        if rc["name"] in seen:
            raise ValueError(f"{path}: duplicate run name {rc['name']}")
        seen.add(rc["name"])
        rc.setdefault("out", str(ROOT / "runs" / sweep / rc["name"]))
        rc["sweep"] = sweep
        rc["slurm"] = merge(spec.get("slurm") or {}, rc.get("slurm") or {})
    return runs
