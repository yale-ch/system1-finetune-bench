"""GLiNER2 decision models (``fastino/GLiNER2.5-Decide``, ``-1B``, ``-multi``): a DeBERTa-v3 encoder that
scores every label (with its description) against the text in one pass and softmaxes over them.

All three question types are single-label classifications:

* choice -- labels = option keys, descriptions = option texts;
* score  -- labels = level keys, descriptions = level meanings;
* noul   -- labels ``true`` / ``false``; the statement goes into the text.

The text is ``text_template`` (model config) over ``question`` and ``state``. Training uses gliner2's own
trainer, with the fixes that made our Linked Art and Europeana fine-tunes work (see ``configure_processor``):
its default augmentation drops descriptions, drops labels (including ``none``) and renames labels, none of
which happens at inference. It also silently skips batches that fail (OOM included); here any failed
micro-batch is fatal.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

from ..schema import Example
from .base import Adapter

log = logging.getLogger(__name__)
TASK = "decision"
_MARKERS = re.compile(r"\[(?:P|L|C|E|R|DESCRIPTION|EXAMPLE|OUTPUT)\]")
NOUL_LABELS = {"true": "the statement is true", "false": "the statement is false"}


def sanitize(value: str) -> str:
    """gliner2 reserves its marker tokens and bare parentheses. Strip markers again after the bracket swap:
    "(C)" becomes "[C]", which is a marker."""
    value = _MARKERS.sub(" ", str(value))
    value = value.replace("(", "[").replace(")", "]")
    value = _MARKERS.sub(" ", value)
    return " ".join(value.split()) or "-"


class GLiNER2Adapter(Adapter):
    supports = ("choice", "score", "noul")

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.templates = {"choice": "{question}\n{state}", "score": "{question}\n{state}", "noul": "Statement: {question}\n{state}",
                          **(cfg.get("text_template") or {})}
        self.path = cfg["checkpoint"]

    # ----------------------------------------------------------------------------------------- encoding
    def labels(self, ex: Example) -> tuple[dict[str, str], dict[str, str]]:
        """(label -> description) as gliner2 sees it, and (label -> original key)."""
        if ex.type == "noul":
            return dict(NOUL_LABELS), {"true": "true", "false": "false"}
        spec, back = {}, {}
        for o in ex.options:
            name = sanitize(o.key)
            while name in spec:
                name += "_"
            spec[name] = sanitize(o.text) if o.text else ""
            back[name] = o.key
        return spec, back

    def text(self, ex: Example) -> str:
        return self.templates[ex.type].format(question=ex.question, state=ex.state).strip()

    def record(self, ex: Example) -> dict:
        spec, back = self.labels(ex)
        if ex.type == "noul":
            gold = "true" if float(ex.answer) >= 0.5 else "false"
        else:
            gold = next(n for n, k in back.items() if k == ex.answer)
        cls = {"task": TASK, "labels": list(spec), "true_label": [gold]}
        if any(spec.values()):
            cls["label_descriptions"] = {k: v for k, v in spec.items() if v}
        return {"input": self.text(ex), "output": {"classifications": [cls]}}

    # ------------------------------------------------------------------------------------------ inference
    def load(self) -> None:
        self._load_classifier(self.path)

    def _load_classifier(self, path) -> None:
        import torch
        from gliner2.classification import Classifier

        self.model = Classifier.from_pretrained(str(path)).to(self.device).eval()
        if self.cfg.get("bf16_inference"):
            self.model = self.model.to(torch.bfloat16)

    def train_mode(self, on: bool) -> None:
        pass  # inference object only; training happens in fit()

    def predict(self, batch: list[Example]) -> list:
        from gliner2.classification import ClassificationSchema

        out = []
        for ex in batch:
            spec, back = self.labels(ex)
            schema = ClassificationSchema().single(TASK, spec if any(spec.values()) else list(spec))
            res = self.model.classify(self.text(ex), schema)
            probs = {back[k]: float(v) for k, v in res.probabilities(TASK).items()}
            out.append(probs["true"] if ex.type == "noul" else probs)
        return out

    def loss(self, batch: list[Example]):
        """Cross-entropy of the gold label under the inference path (used for dev loss reporting only)."""
        import math

        import torch

        total = 0.0
        for ex, p in zip(batch, self.predict(batch)):
            if ex.type == "noul":
                y = float(ex.answer)
                total -= y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9))
            else:
                total -= math.log(max(p.get(ex.answer, 0.0), 1e-9))
        return torch.tensor(total / max(len(batch), 1))

    def trainable_summary(self) -> dict:
        n = sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0
        return {"params_total": n, "params_trainable": n, "trainable_frac": 1.0}

    # ------------------------------------------------------------------------------------------- training
    def fit(self, train, dev, spec, cfg, out: Path) -> Path:
        import random

        import torch
        from gliner2 import AutoExtractor
        from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

        from ..train import render

        rng = random.Random(cfg["seed"])
        # gliner2 shuffles candidate order itself; render once without our shuffle
        tr = [self.record(ex) for ex in render(train, spec, train=True, rng=rng)]
        dv = [self.record(ex) for ex in render(dev, spec)]
        micro, accum = cfg["micro_batch"], max(1, cfg["batch_size"] // cfg["micro_batch"])
        del self.model
        torch.cuda.empty_cache()
        model = AutoExtractor.from_pretrained(self.path)
        configure_processor(model.processor)
        if self.cfg.get("gradient_checkpointing", True):
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        ckpts = out / "checkpoints"
        tcfg = TrainingConfig(
            output_dir=str(ckpts), experiment_name=self.cfg["name"], num_epochs=cfg["epochs"], max_steps=cfg["max_steps"],
            batch_size=micro, eval_batch_size=micro, gradient_accumulation_steps=accum,
            encoder_lr=float(cfg["lr"]), task_lr=float(cfg.get("head_lr") or self.cfg.get("head_lr", 1e-4)),
            warmup_ratio=cfg["warmup_ratio"], scheduler_type=cfg["scheduler"], eval_strategy="steps",
            eval_steps=cfg["eval_every"], save_best=True, metric_for_best="eval_loss", greater_is_better=False,
            save_total_limit=2, logging_steps=cfg["log_every"], num_workers=4, seed=cfg["seed"], bf16=cfg["bf16"],
            fp16=False, max_len=self.cfg.get("max_len"), gradient_checkpointing=False,
        )
        trainer = GLiNER2Trainer(model, tcfg)
        original = trainer._record_failed_batch

        def record_failed(batch, *a, **kw):  # gliner2 would skip a failing batch (OOM included) and carry on
            original(batch, *a, **kw)
            raise RuntimeError(f"gliner2 training micro-batch failed: {kw.get('error')!r}")

        trainer._record_failed_batch = record_failed
        summary = trainer.train(train_data=tr, eval_data=dv)
        (out / "gliner-train-summary.json").write_text(json.dumps({k: v for k, v in (summary or {}).items()
                                                                     if isinstance(v, (int, float, str))}, indent=2))
        best = ckpts / "best" if (ckpts / "best").exists() else ckpts / "final"
        dest = out / "best"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(best), dest)
        shutil.rmtree(ckpts, ignore_errors=True)
        del trainer, model
        torch.cuda.empty_cache()
        return dest

    def save(self, path: Path) -> None:
        self.model.save_pretrained(str(path))

    def load_checkpoint(self, path: Path) -> None:
        self._load_classifier(path)


# ------------------------------------------------------------------------------------- gliner2 training fixes


def configure_processor(proc):
    """Train on the prompts we infer with: descriptions always on, every label kept, real label names;
    label order still shuffled so the model cannot learn positions."""
    from gliner2.processor import SamplingConfig

    proc.sampling_config = SamplingConfig(
        remove_classification_prob=0.0,
        shuffle_classification_labels=True,
        remove_classification_label_prob=0.0,
        synthetic_label_prob=0.0,
        include_true_label_prob=1.0,
    )
    proc.__class__ = _describing_class(type(proc))
    return proc


_DESCRIBING: dict = {}


def _describing_class(base):
    if base.__name__.startswith("Describing"):
        return base
    if base not in _DESCRIBING:
        def _transform_schema(self, parent, fields, child_prefix, prompt=None, examples=None, label_descriptions=None, example_mode="both"):
            if child_prefix == self.L_TOKEN and label_descriptions:
                example_mode = "both"
            return base._transform_schema(self, parent, fields, child_prefix, prompt=prompt, examples=examples,
                                          label_descriptions=label_descriptions, example_mode=example_mode)

        _DESCRIBING[base] = type(f"Describing{base.__name__}", (base,), {"_transform_schema": _transform_schema, "__module__": __name__})
        globals()[_DESCRIBING[base].__name__] = _DESCRIBING[base]
    return _DESCRIBING[base]
