"""One run = one model x one or more datasets x one prompt approach x one seed.

    python -m s1bench.train configs/experiments/<file>.yaml --run <name>     # a run from a sweep file
    python -m s1bench.train run.yaml                                       # a single run config

Steps: prepare the datasets -> (optionally) score the untouched model on every test split -> fine-tune,
choosing the best checkpoint on the dev splits (never on test) -> reload it and score every test split.
Outputs in ``<out>/``: ``config.json``, ``metrics.json`` (zero-shot and tuned, per dataset and type),
``predictions-<stage>@<dataset>.jsonl``, ``train-log.jsonl``, and ``best/`` (what the adapter saves).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path

from . import approaches, data
from .metrics import evaluate
from .models import base as models
from .schema import Example

log = logging.getLogger("s1bench")

TRAIN_DEFAULTS = dict(
    epochs=2, max_steps=-1, batch_size=16, micro_batch=4, lr=1e-4, head_lr=None, weight_decay=0.0,
    warmup_ratio=0.05, scheduler="linear", max_grad_norm=1.0, eval_every=200, eval_batch=16, patience=0,
    select_on="dev_loss", bf16=True, seed=42, zero_shot=True, preflight=True, max_dev=2000, log_every=10, calibrate=True,
)


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def render(exs: list[Example], spec: dict, *, train: bool = False, rng: random.Random | None = None) -> list[Example]:
    return [approaches.apply(ex, spec, train=train, rng=rng) for ex in exs]


def length(ex: Example) -> int:
    return len(ex.state) + len(ex.question) + sum(len(o.key) + len(o.text) for o in ex.options)


# ---------------------------------------------------------------------------------------------- evaluation


def predict_all(adapter, exs: list[Example], spec: dict, batch: int) -> list[dict]:
    import torch

    adapter.train_mode(False)
    rendered = render(exs, spec)
    order = sorted(range(len(rendered)), key=lambda i: length(rendered[i]))  # similar lengths together
    out: list[dict | None] = [None] * len(exs)
    with torch.inference_mode():
        for s in range(0, len(order), batch):
            idx = order[s : s + batch]
            probs = adapter.predict([rendered[i] for i in idx])
            for i, p in zip(idx, probs):
                out[i] = {"id": exs[i].id, "type": exs[i].type, "probs": approaches.back_probs(rendered[i], p)}
    return out


def dev_loss(adapter, exs: list[Example], spec: dict, batch: int) -> float:
    import torch

    adapter.train_mode(False)
    rendered = render(exs, spec)
    total, n = 0.0, 0
    with torch.inference_mode():
        for s in range(0, len(rendered), batch):
            chunk = rendered[s : s + batch]
            total += float(adapter.loss(chunk)) * len(chunk)
            n += len(chunk)
    return total / max(n, 1)


def score_tests(adapter, tests: dict[str, list[Example]], spec: dict, cfg: dict, out: Path, stage: str) -> dict:
    res = {}
    for name, exs in tests.items():
        t0 = time.time()
        preds = predict_all(adapter, exs, spec, cfg["eval_batch"])
        m = evaluate(exs, preds)
        m["seconds"] = round(time.time() - t0, 1)
        res[name] = m
        with open(out / f"predictions-{stage}@{name}.jsonl", "w") as fh:
            for ex, p in zip(exs, preds):
                fh.write(json.dumps({**p, "answer": ex.answer, "approach": spec.get(ex.type)}, ensure_ascii=False) + "\n")
        log.info(f"{stage} @ {name}: " + json.dumps({t: {k: round(v, 4) for k, v in mm.items() if isinstance(v, float)} if isinstance(mm, dict) else mm
                                                     for t, mm in m.items()}))
    return res


# ------------------------------------------------------------------------------------------------ training


def make_scheduler(opt, cfg: dict, total: int):
    from torch.optim.lr_scheduler import LambdaLR

    warm = max(1, int(cfg["warmup_ratio"] * total))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        frac = (step - warm) / max(1, total - warm)
        if cfg["scheduler"] == "constant":
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * frac)) if cfg["scheduler"] == "cosine" else max(0.0, 1 - frac)

    return LambdaLR(opt, f)


def fit(adapter, train: list[Example], dev: list[Example], spec: dict, cfg: dict, out: Path) -> Path:
    import torch

    rng = random.Random(cfg["seed"])
    micro, accum = cfg["micro_batch"], max(1, cfg["batch_size"] // cfg["micro_batch"])
    steps_per_epoch = math.ceil(len(train) / (micro * accum))
    total = cfg["max_steps"] if cfg["max_steps"] > 0 else steps_per_epoch * cfg["epochs"]
    opt = torch.optim.AdamW(adapter.param_groups(cfg), weight_decay=cfg["weight_decay"])
    sched = make_scheduler(opt, cfg, total)
    params = [p for g in opt.param_groups for p in g["params"]]
    best_dir, best_val, bad, step = out / "best", None, 0, 0
    logf = open(out / "train-log.jsonl", "a")
    log.info(f"training: {len(train)} examples, micro {micro} x accum {accum}, {total} optimizer steps, "
             + json.dumps(adapter.trainable_summary()))

    if cfg["preflight"]:
        longest = sorted(train, key=length, reverse=True)[:micro]
        adapter.train_mode(True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        adapter.loss(render(longest, spec, train=True, rng=rng)).backward()
        opt.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
        log.info(f"preflight OK on the {micro} longest examples: peak {peak:.1f} GiB")

    def evaluate_dev():
        nonlocal best_val, bad
        dl = dev_loss(adapter, dev, spec, cfg["eval_batch"])
        row = {"step": step, "dev_loss": dl}
        if cfg["select_on"] != "dev_loss":
            m = evaluate(dev, predict_all(adapter, dev, spec, cfg["eval_batch"]))
            row["dev_adjusted"] = m["adjusted_mean"]
        val = row["dev_loss"] if cfg["select_on"] == "dev_loss" else -row["dev_adjusted"]
        improved = best_val is None or val < best_val
        if improved:
            best_val, bad = val, 0
            adapter.save(best_dir)
        else:
            bad += 1
        row["best"] = improved
        logf.write(json.dumps(row) + "\n")
        logf.flush()
        log.info(f"eval step {step}: {row}")
        adapter.train_mode(True)
        return cfg["patience"] and bad >= cfg["patience"]

    t0, running, seen = time.time(), 0.0, 0
    stop = False
    for epoch in range(10**6):
        order = train[:]
        rng.shuffle(order)
        adapter.train_mode(True)
        for s in range(0, len(order), micro * accum):
            chunk = order[s : s + micro * accum]
            for m in range(0, len(chunk), micro):
                mb = render(chunk[m : m + micro], spec, train=True, rng=rng)
                loss = adapter.loss(mb)
                (loss * (len(mb) / len(chunk))).backward()
                running += float(loss.detach()) * len(mb)
                seen += len(mb)
            gn = torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg["log_every"] == 0:
                row = {"step": step, "epoch": epoch, "loss": running / max(seen, 1), "grad_norm": float(gn),
                       "lr": sched.get_last_lr()[0], "minutes": round((time.time() - t0) / 60, 2)}
                logf.write(json.dumps(row) + "\n")
                logf.flush()
                log.info(json.dumps(row))
                running, seen = 0.0, 0
            if step % cfg["eval_every"] == 0 and evaluate_dev():
                stop = True
            if stop or step >= total:
                break
        if stop or step >= total:
            break
    if step % cfg["eval_every"]:
        evaluate_dev()
    logf.close()
    return best_dir


# ----------------------------------------------------------------------------------------------------- run


def run(rc: dict) -> dict:
    import torch

    mcfg = models.load_config(rc["model"])
    cfg = {**TRAIN_DEFAULTS, **(mcfg.get("train") or {}), **(rc.get("train") or {})}
    out = Path(rc["out"])
    out.mkdir(parents=True, exist_ok=True)
    spec = rc.get("approach") or {}
    seed_everything(cfg["seed"])
    device = rc.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    names = rc["datasets"] if isinstance(rc["datasets"], list) else [rc["datasets"]]
    train, dev, tests = [], [], {}
    for n in names:
        train += data.load(n, "train")
        d = data.load(n, "dev")
        dev += random.Random(0).sample(d, min(len(d), cfg["max_dev"] // len(names)))
        tests[data.load_config(n)["name"]] = data.load(n, "test")
    for n in rc.get("eval_datasets") or []:
        tests[data.load_config(n)["name"]] = data.load(n, "test")
    if cfg.get("max_train"):
        train = random.Random(cfg["seed"]).sample(train, min(cfg["max_train"], len(train)))
    if cfg.get("max_test"):
        tests = {k: v[: cfg["max_test"]] for k, v in tests.items()}

    adapter = models.build(rc["model"], device=device)
    adapter.check_types({approaches.apply(ex, spec).type for ex in train[:2000] + [e for t in tests.values() for e in t[:500]]})
    t0 = time.time()
    adapter.load()
    log.info(f"loaded {adapter.cfg['name']} in {time.time() - t0:.0f}s on {device}")
    summary = {"run": rc.get("name"), "model": adapter.cfg, "datasets": names, "approach": spec, "train": cfg,
               "n": {"train": len(train), "dev": len(dev), **{f"test@{k}": len(v) for k, v in tests.items()}},
               "host": os.uname().nodename, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (out / "config.json").write_text(json.dumps(summary, indent=2, default=str))

    metrics = {"zero_shot": None, "tuned": None}
    if cfg["zero_shot"]:
        metrics["zero_shot"] = score_tests(adapter, tests, spec, cfg, out, "zeroshot")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if cfg.get("eval_only"):  # score the model as configured (e.g. an exported fine-tune); no training
        log.info(f"eval_only: done -> {out / 'metrics.json'}")
        return metrics
    t0 = time.time()
    if adapter.fit is not None:
        best = adapter.fit(train, dev, spec, cfg, out)
    else:
        best = fit(adapter, train, dev, spec, cfg, out)
    minutes = (time.time() - t0) / 60
    adapter.load_checkpoint(best)
    if cfg.get("calibrate", True) and hasattr(adapter, "calibrate"):
        metrics["temperature"] = adapter.calibrate(dev, cfg["eval_batch"])
        adapter.save(best)  # keep the refit temperature with the weights
    metrics["tuned"] = score_tests(adapter, tests, spec, cfg, out, "tuned")
    metrics["train_minutes"] = round(minutes, 1)
    metrics["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1) if torch.cuda.is_available() else None
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info(f"done -> {out / 'metrics.json'}")
    return metrics


def main() -> None:
    from .sweep import expand

    ap = argparse.ArgumentParser(description="Fine-tune and score one System One model run.")
    ap.add_argument("config", help="a run config, or a sweep file with --run / --index")
    ap.add_argument("--run", help="run name inside a sweep file")
    ap.add_argument("--index", type=int, help="run index inside a sweep file (e.g. $SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. train.max_steps=20")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runs = expand(args.config)
    if args.run:
        rc = next(r for r in runs if r["name"] == args.run)
    elif args.index is not None:
        rc = runs[args.index]
    elif len(runs) == 1:
        rc = runs[0]
    else:
        raise SystemExit(f"{args.config} has {len(runs)} runs; pass --run or --index")
    for kv in args.set:
        key, _, val = kv.partition("=")
        node = rc
        *path, last = key.split(".")
        for k in path:
            node = node.setdefault(k, {})
        import yaml

        node[last] = yaml.safe_load(val)
    run(rc)


if __name__ == "__main__":
    main()
