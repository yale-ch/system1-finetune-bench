"""The interface every model adapter implements, and the registry that maps a model config to one.

An adapter wraps one *architecture family* (how a model reads a decision and where its probabilities come
from). A *model* is a YAML file in ``configs/models/`` that names the family plus the checkpoint and any
family settings, so adding another checkpoint of a known family is a config file, and adding a new
family is one class here.

The trainer only ever calls:

* ``load()`` -- build the model on ``self.device`` (and make the trainable parameters trainable);
* ``param_groups(cfg)`` -- optimizer parameter groups;
* ``loss(batch)`` -- mean training loss over a list of (rendered) examples, a scalar tensor;
* ``predict(batch)`` -- one probability output per example: ``{key: p}`` for choice/score, ``float``
  P(true) for noul -- in the *rendered* example's key space (the trainer maps it back);
* ``save(path)`` / ``load_checkpoint(path)`` -- write / restore what was trained (a LoRA adapter, a head,
  or the full weights), so the best checkpoint can be reloaded and shipped.

Adapters may override ``fit`` entirely (GLiNER uses its own trainer).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "models"

FAMILIES = {
    # family name -> "module:Class"
    "gliner2": "s1bench.models.gliner:GLiNER2Adapter",
    "letters": "s1bench.models.causal_lm:LetterAdapter",
    "pointer": "s1bench.models.causal_lm:PointerAdapter",
    "candidate": "s1bench.models.causal_lm:CandidateAdapter",
    "nli": "s1bench.models.causal_lm:NLIAdapter",
    "slots": "s1bench.models.causal_lm:SlotHeadAdapter",
    "laya": "s1bench.models.encoders:LayaAdapter",
    "von": "s1bench.models.encoders:VonAdapter",
    "bekko": "s1bench.models.encoders:BekkoAdapter",
}


def load_config(name_or_cfg: str | dict) -> dict:
    if isinstance(name_or_cfg, dict):
        return dict(name_or_cfg)
    p = Path(name_or_cfg)
    if not p.exists():
        p = CONFIG_DIR / f"{name_or_cfg}.yaml"
    cfg = yaml.safe_load(p.read_text())
    cfg.setdefault("name", p.stem)
    return cfg


def build(name_or_cfg: str | dict, device: str = "cuda", **overrides) -> "Adapter":
    cfg = {**load_config(name_or_cfg), **overrides}
    target = FAMILIES.get(cfg["family"], cfg["family"])
    mod, _, cls = target.partition(":")
    return getattr(importlib.import_module(mod), cls)(cfg, device)


class Adapter:
    #: question types the family answers natively; others need an approach that rewrites them (as_choice)
    supports: tuple[str, ...] = ("choice", "score", "noul")

    def __init__(self, cfg: dict, device: str):
        self.cfg = cfg
        self.device = device
        self.model = None

    # -- lifecycle ----------------------------------------------------------------------------------------
    def load(self) -> None:
        raise NotImplementedError

    def param_groups(self, train_cfg: dict) -> list[dict]:
        params = [p for p in self.model.parameters() if p.requires_grad]
        return [{"params": params, "lr": float(train_cfg.get("lr", 1e-4))}]

    def train_mode(self, on: bool) -> None:
        self.model.train(on)

    # -- the task -----------------------------------------------------------------------------------------
    def loss(self, batch: list):
        raise NotImplementedError

    def predict(self, batch: list) -> list:
        raise NotImplementedError

    # -- checkpoints --------------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load_checkpoint(self, path: Path) -> None:
        raise NotImplementedError

    # -- optional -----------------------------------------------------------------------------------------
    fit = None  # set to a method ``fit(train, dev, run_cfg, out_dir) -> best_path`` to bypass the shared loop

    def trainable_summary(self) -> dict:
        total = sum(p.numel() for p in self.model.parameters())
        train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"params_total": total, "params_trainable": train, "trainable_frac": train / max(total, 1)}

    def check_types(self, types: set[str]) -> None:
        missing = set(types) - set(self.supports)
        if missing:
            raise ValueError(f"{self.cfg['name']} ({self.cfg['family']}) does not answer {sorted(missing)} natively; "
                             f"use an approach that rewrites them, e.g. approach: {{score: as_choice, noul: as_choice}}")
