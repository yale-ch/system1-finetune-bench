"""Datasets: anything that can be turned into canonical :class:`~s1bench.schema.Example` rows.

A dataset is a YAML file in ``configs/datasets/`` (or a dict in a run config). Three ways to define one:

1. **Canonical JSONL already on disk** -- ``source: {canonical: {train: path, test: path}}``.
2. **Field mapping** over a Hugging Face dataset, a JSONL/CSV/Parquet file -- see ``mapping.py``.
3. **A converter function** -- ``converter: s1bench.data.builtin:europeana_persons`` (any importable
   ``fn(cfg) -> {split: [Example]}``), for sources that need real code.

``prepare(cfg)`` materialises ``train/dev/test.jsonl`` under ``data/<name>/`` once (dev carved out of
train by group hash when the source has none), so every run of every model sees identical rows.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import random
from pathlib import Path

import yaml

from ..schema import Example, read_jsonl, write_jsonl

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "datasets"


def load_config(name_or_cfg: str | dict) -> dict:
    if isinstance(name_or_cfg, dict):
        return name_or_cfg
    p = Path(name_or_cfg)
    if not p.exists():
        p = CONFIG_DIR / f"{name_or_cfg}.yaml"
    cfg = yaml.safe_load(p.read_text())
    cfg.setdefault("name", p.stem)
    return cfg


def bucket(group: str, salt: str = "") -> float:
    return int(hashlib.sha1(f"{salt}{group}".encode()).hexdigest(), 16) % 10_000 / 10_000


def hash_split(rows: list[Example], frac: float, salt: str = "") -> tuple[list[Example], list[Example]]:
    keep, held = [], []
    for ex in rows:
        (held if bucket(ex.group, salt) < frac else keep).append(ex)
    return keep, held


def build(cfg: dict) -> dict[str, list[Example]]:
    """Run the dataset's source and return {split: examples} (train/test at least)."""
    if "converter" in cfg:
        mod, _, fn = cfg["converter"].partition(":")
        splits = getattr(importlib.import_module(mod), fn)(cfg)
    elif "canonical" in cfg.get("source", {}):
        splits = {s: read_jsonl(p) for s, p in cfg["source"]["canonical"].items()}
    else:
        from .mapping import build_mapped

        splits = build_mapped(cfg)
    if "test" not in splits:
        frac = float(cfg.get("test_frac", 0.1))
        splits["train"], splits["test"] = hash_split(splits["train"], frac, salt="test:")
    if "dev" not in splits:
        frac = float(cfg.get("dev_frac", 0.1))
        splits["train"], splits["dev"] = hash_split(splits["train"], frac, salt="dev:")
    rng = random.Random(cfg.get("seed", 0))
    for split, cap in (cfg.get("limit") or {}).items():
        if split in splits and len(splits[split]) > cap:
            splits[split] = sorted(rng.sample(splits[split], cap), key=lambda e: e.id)
    for split, rows in splits.items():
        for ex in rows:
            ex.split = split
            ex.validate()
    leaks = {g for g in (e.group for e in splits["train"])} & {e.group for e in splits["test"]}
    if leaks and not cfg.get("allow_group_leak"):
        raise ValueError(f"{cfg['name']}: {len(leaks)} groups appear in both train and test (e.g. {sorted(leaks)[:3]})")
    return splits


def data_dir(cfg: dict, root: Path | None = None) -> Path:
    return (root or ROOT / "data") / cfg["name"]


def prepare(name_or_cfg: str | dict, root: Path | None = None, force: bool = False) -> Path:
    cfg = load_config(name_or_cfg)
    out = data_dir(cfg, root)
    if (out / "manifest.json").exists() and not force:
        return out
    splits = build(cfg)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"name": cfg["name"], "config": cfg, "splits": {}}
    for split, rows in splits.items():
        write_jsonl(rows, out / f"{split}.jsonl")
        types: dict[str, int] = {}
        for ex in rows:
            types[ex.type] = types.get(ex.type, 0) + 1
        manifest["splits"][split] = {"n": len(rows), "types": types}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info(f"prepared {cfg['name']}: {json.dumps(manifest['splits'])}")
    return out


def load(name_or_cfg: str | dict, split: str, root: Path | None = None) -> list[Example]:
    cfg = load_config(name_or_cfg)
    d = prepare(cfg, root)
    return read_jsonl(d / f"{split}.jsonl")
