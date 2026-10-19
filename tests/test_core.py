"""Model-free tests: schema, approaches, metrics, mapping datasets, sweeps."""

import json
import random

import pytest

from s1bench import approaches
from s1bench.data import build, hash_split
from s1bench.metrics import evaluate
from s1bench.schema import Example, Option
from s1bench.sweep import expand


def choice_ex(answer="wikidata:Q2"):
    return Example(id="c1", type="choice", state="Text about Ada.", question="Which record?",
                   options=[Option("wikidata:Q1", "Ada Lovelace"), Option("wikidata:Q2", "Ada Yonath"), Option("none", "None")],
                   answer=answer)


def score_ex():
    return Example(id="s1", type="score", state="Great film", question="How positive?",
                   options=[Option(str(i), t) for i, t in enumerate(["bad", "ok", "good"], 1)], answer="3")


def noul_ex(y=1.0):
    return Example(id="n1", type="noul", state="The sky is blue.", question="The sky is blue.", answer=y)


def test_validate_rejects_bad_answer():
    with pytest.raises(ValueError):
        Example(id="x", type="choice", state="s", question="q", options=[Option("a"), Option("b")], answer="c").validate()
    with pytest.raises(ValueError):
        Example(id="x", type="noul", state="s", question="q", answer=2.0).validate()


@pytest.mark.parametrize("name", ["described", "in_state", "bare", "lettered", "text_as_key"])
def test_choice_approaches_round_trip(name):
    ex = choice_ex()
    r = approaches.apply(ex, {"choice": name})
    r.validate()
    probs = {k: 0.1 for k in r.keys}
    probs[r.answer] = 0.8
    back = approaches.back_probs(r, probs)
    assert max(back, key=back.get) == ex.answer
    assert abs(sum(back.values()) - 1.0) < 1e-9


def test_in_state_moves_descriptions():
    r = approaches.apply(choice_ex(), {"choice": "in_state"})
    assert "Ada Lovelace" in r.state and all(not o.text for o in r.options)


def test_score_as_choice_and_noul_as_choice():
    r = approaches.apply(score_ex(), {"score": "as_choice"})
    assert r.type == "choice" and r.answer == "3"
    assert approaches.back_probs(r, {"1": 0.1, "2": 0.2, "3": 0.7}) == {"1": 0.1, "2": 0.2, "3": 0.7}
    n = approaches.apply(noul_ex(0.0), {"noul": "as_choice"})
    assert n.type == "choice" and n.answer == "false"
    assert approaches.back_probs(n, {"true": 0.3, "false": 0.7}) == 0.3


def test_shuffle_keeps_none_last():
    rng = random.Random(1)
    for _ in range(20):
        r = approaches.apply(choice_ex(), {"shuffle_options": True}, train=True, rng=rng)
        assert r.options[-1].key == "none"


def test_metrics():
    exs = [choice_ex(), choice_ex("none"), score_ex(), noul_ex(1.0), noul_ex(0.0)]
    preds = [{"probs": {"wikidata:Q1": 0.1, "wikidata:Q2": 0.8, "none": 0.1}},
             {"probs": {"wikidata:Q1": 0.6, "wikidata:Q2": 0.1, "none": 0.3}},
             {"probs": {"1": 0.0, "2": 0.2, "3": 0.8}},
             {"probs": 0.9}, {"probs": 0.2}]
    m = evaluate(exs, preds)
    assert m["choice"]["accuracy"] == 0.5 and m["choice"]["false_link_rate"] == 1.0
    assert m["score"]["accuracy"] == 1.0 and abs(m["score"]["mae_expected"] - 0.2) < 1e-9
    assert m["noul"]["balanced_accuracy"] == 1.0 and m["noul"]["auroc"] == 1.0


def test_mapping_dataset(tmp_path):
    rows = [{"q": f"question {i}", "p": f"passage {i}", "a": i % 2 == 0, "t": f"title{i % 7}"} for i in range(200)]
    src = tmp_path / "rows.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows))
    cfg = {"name": "toy", "type": "noul", "source": {"jsonl": str(src)}, "splits": {"train": "train"},
           "state": "{p}", "question": "{q}", "answer": {"field": "a", "map": {True: 1.0, False: 0.0}}, "group": "{t}"}
    splits = build(cfg)
    assert set(splits) == {"train", "dev", "test"}
    groups = {s: {e.group for e in v} for s, v in splits.items()}
    assert not (groups["train"] & groups["test"]) and not (groups["train"] & groups["dev"])
    assert sum(len(v) for v in splits.values()) == 200


def test_mapping_options_dotted(tmp_path):
    row = {"id": "x", "question": "Q?", "choices": {"label": ["A", "B"], "text": ["yes", "no"]}, "answerKey": "B"}
    src = tmp_path / "arc.jsonl"
    src.write_text(json.dumps(row))
    cfg = {"name": "arc", "type": "choice", "source": {"jsonl": {"train": str(src), "test": str(src)}},
           "splits": {"train": "train", "test": "test"}, "id": "{id}", "state": "{question}", "question": "Pick",
           "options": {"keys": "choices.label", "texts": "choices.text"}, "answer": "{answerKey}",
           "dev_frac": 0.0, "allow_group_leak": True}
    ex = build(cfg)["test"][0]
    assert ex.keys == ["A", "B"] and ex.options[1].text == "no" and ex.answer == "B"


def test_hash_split_is_stable():
    exs = [Example(id=str(i), type="noul", state="s", question="q", answer=1.0) for i in range(100)]
    a, b = hash_split(exs, 0.2), hash_split(exs, 0.2)
    assert [e.id for e in a[1]] == [e.id for e in b[1]]


def test_sweep(tmp_path):
    spec = {"name": "t", "defaults": {"train": {"epochs": 1}},
            "matrix": {"model": ["gliner-decide", "kev-4b"], "datasets": [["boolq"]], "approach.noul": ["statement", "as_choice"],
                       "train.seed": [1, 2]},
            "exclude": [{"model": "kev-4b", "approach.noul": "as_choice"}]}
    p = tmp_path / "t.yaml"
    import yaml

    p.write_text(yaml.safe_dump(spec))
    runs = expand(p)
    assert len(runs) == 6
    assert len({r["name"] for r in runs}) == 6
    assert all(r["train"]["epochs"] == 1 for r in runs)
