"""Every decoder family on a tiny random Qwen3.5 (real tokenizer, 4 layers, d=64) on CPU: loss, backward,
predict, checkpoint round trip, calibration, and one end-to-end run of the trainer.

Needs the network once (the Qwen3.5 tokenizer and JEV-9B's judge_config.json). Run with ``pytest -m tiny``.
"""

import json
import math

import pytest

from s1bench.models import base as models
from s1bench.schema import Example, Option, write_jsonl

pytestmark = pytest.mark.tiny
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    import torch
    import transformers as tf

    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("tiny")
    tok = tf.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
    cfg = tf.Qwen3_5TextConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                               num_attention_heads=2, num_key_value_heads=1, head_dim=32, linear_num_key_heads=2,
                               linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
                               layer_types=["linear_attention"] * 3 + ["full_attention"], tie_word_embeddings=True)
    tf.Qwen3_5ForCausalLM(cfg).save_pretrained(d / "lm")
    tok.save_pretrained(d / "lm")
    cls_cfg = tf.Qwen3_5TextConfig(**{**cfg.to_dict(), "num_labels": 3, "id2label": {0: "contradiction", 1: "entailment", 2: "neutral"},
                                      "label2id": {"contradiction": 0, "entailment": 1, "neutral": 2}, "pad_token_id": tok.pad_token_id})
    tf.Qwen3_5TextForSequenceClassification(cls_cfg).save_pretrained(d / "nli")
    tok.save_pretrained(d / "nli")
    return d


def examples():
    return [
        Example(id="c", type="choice", state="Ada Lovelace wrote the first program.", question="Who is this?",
                options=[Option("wikidata:Q7259", "Ada Lovelace, mathematician"), Option("wikidata:Q1", "the universe"),
                         Option("none", "none of these")], answer="wikidata:Q7259"),
        Example(id="s", type="score", state="Loved every minute.", question="How positive?",
                options=[Option("1", "negative"), Option("2", "neutral"), Option("3", "positive")], answer="3"),
        Example(id="n", type="noul", state="The sky is green.", question="The sky is blue.", answer=0.0),
    ]


def family_cfgs(d):
    lora = {"r": 4, "alpha": 8, "dropout": 0.0, "targets": TARGETS}
    lm = str(d / "lm")
    return {
        "letters-jevk5": {"family": "letters", "style": "jevk5", "checkpoint": lm, "lora": lora},
        "letters-decider": {"family": "letters", "style": "decider", "checkpoint": lm, "lora": lora},
        "letters-tev1": {"family": "letters", "style": "tev1", "checkpoint": lm, "lora": lora},
        "pointer": {"family": "pointer", "checkpoint": lm, "lora": lora, "head_dim": 16},
        "candidate": {"family": "candidate", "checkpoint": lm, "lora": lora, "brier_weight": 0.1},
        "nli": {"family": "nli", "checkpoint": str(d / "nli"), "lora": {**lora, "scope": None}},
        "slots": {"family": "slots", "checkpoint": lm, "lora": lora, "head": {"repo": "autotrust/JEV-9B", "pretrained": False}},
    }


@pytest.mark.parametrize("name", ["letters-jevk5", "letters-decider", "letters-tev1", "pointer", "candidate", "nli", "slots"])
def test_family(tiny, tmp_path, name):
    import torch

    cfg = {"name": name, "gradient_checkpointing": True, **family_cfgs(tiny)[name]}
    a = models.build(cfg, device="cpu")
    a.load()
    exs = examples()
    a.train_mode(True)
    loss = a.loss(exs)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for g in a.param_groups({"lr": 1e-3}) for p in g["params"] if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads), "no gradient reached the trainable parameters"
    a.train_mode(False)
    with torch.no_grad():
        preds = a.predict(exs)
    assert math.isclose(sum(preds[0].values()), 1.0, rel_tol=1e-4) and set(preds[0]) == set(exs[0].keys)
    assert set(preds[1]) == {"1", "2", "3"} and 0.0 <= preds[2] <= 1.0
    a.save(tmp_path / "ck")
    with torch.no_grad():
        for p in (p for g in a.param_groups({"lr": 1}) for p in g["params"]):
            p.add_(1.0)
    a.load_checkpoint(tmp_path / "ck")
    with torch.no_grad():
        again = a.predict(exs)
    assert all(abs(again[0][k] - preds[0][k]) < 1e-5 for k in preds[0])
    t = a.calibrate(exs * 2, batch=3)
    assert all(0.05 <= v <= 20 for v in t.values())


def test_knockout(tiny):
    cfg = {"name": "ko", **family_cfgs(tiny)["letters-decider"], "max_options": 3}
    a = models.build(cfg, device="cpu")
    a.load()
    ex = Example(id="k", type="choice", state="s", question="q", options=[Option(str(i), f"opt {i}") for i in range(7)], answer="3")
    import torch

    with torch.no_grad():
        p = a.predict([ex])[0]
    assert set(p) == set(ex.keys) and math.isclose(sum(p.values()), 1.0, rel_tol=1e-4)
    assert len(a.fit_options(ex).options) == 3 and "3" in a.fit_options(ex).keys


def test_end_to_end_run(tiny, tmp_path):
    from s1bench.train import run

    exs = examples()
    rows = {s: [e.copy(id=f"{e.id}{i}", group=f"{e.id}{i}") for i in range(4) for e in exs] for s in ("train", "dev", "test")}
    paths = {}
    for s, r in rows.items():
        paths[s] = str(tmp_path / f"{s}.jsonl")
        write_jsonl(r, paths[s])
    ds = {"name": "toy", "source": {"canonical": paths}, "allow_group_leak": True}
    rc = {"name": "e2e", "model": {"name": "tiny", **family_cfgs(tiny)["letters-jevk5"]}, "datasets": [ds],
          "out": str(tmp_path / "run"), "device": "cpu",
          "train": {"max_steps": 4, "micro_batch": 2, "batch_size": 4, "eval_every": 2, "eval_batch": 4, "log_every": 1}}
    import s1bench.data as data

    data_root = tmp_path / "data"
    orig = data.prepare
    data.prepare = lambda cfg, root=None, force=False: orig(cfg, data_root, force)
    try:
        m = run(rc)
    finally:
        data.prepare = orig
    assert m["tuned"]["toy"]["adjusted_mean"] >= 0
    assert (tmp_path / "run" / "best" / "trainable.safetensors").exists()
    log = [json.loads(l) for l in open(tmp_path / "run" / "train-log.jsonl")]
    assert any("dev_loss" in r for r in log)
