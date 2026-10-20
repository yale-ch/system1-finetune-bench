"""Converters for sources that need code. Each is ``fn(cfg) -> {split: [Example]}``.

``recon``: entity-reconciliation decisions in the GLiNER layout we used for Linked Art and Europeana
(``text`` = "Mention: ...\\nEntity type: ...\\nText: ...", ``labels`` = candidate key -> description incl.
``none``, ``gold``). Read from a Hugging Face dataset (the private
``yale-cultural-heritage/europeana-persons-wikidata-labelled``, config ``gliner``) or local JSONL.

``as`` picks how each decision is asked:

* ``choice`` -- one Choice over all candidates + none (the task as deployed);
* ``noul``   -- one Noul per candidate: "<mention> is <candidate>" (true for the gold candidate only).
  Records whose gold is ``none`` give all-false rows, so the model learns to say no;
* ``score``  -- one Score per candidate, how well it matches on a 1-3 scale: 3 = the gold candidate,
  1 = every other one. (A coarse, derived scale -- useful to exercise Score heads, not a real rubric.)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..schema import Example, Option

NONE = "none"
_HEAD = re.compile(r"^Mention: (?P<mention>.*)\nEntity type: (?P<etype>.*)\nText: (?P<text>.*)$", re.S)


def _rows(cfg: dict, split: str):
    src = cfg["source"]
    if "hf" in src:
        import datasets

        yield from datasets.load_dataset(src["hf"], src.get("name"), split=src.get("splits", {}).get(split, split))
        return
    path = Path(src["jsonl"][split])
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _parse(row: dict) -> tuple[str, str, str, list[Option], str, str]:
    m = _HEAD.match(row["text"])
    mention, etype, text = (m["mention"], m["etype"], m["text"]) if m else (row.get("term") or row.get("name", ""), row.get("kind", "entity"), row["text"])
    labels = row["labels"]
    opts = [Option(l["key"], l.get("description", "")) for l in labels] if isinstance(labels, list) else [Option(k, v) for k, v in labels.items()]
    group = str(row.get("pid") or row.get("lux_id") or row.get("id"))
    return mention, etype, text, opts, row["gold"], group


def recon(cfg: dict) -> dict[str, list[Example]]:
    mode = cfg.get("as", "choice")
    q_choice = cfg.get("question", "Which record is the {etype} “{mention}” referred to in the text?")
    q_noul = cfg.get("statement", "The {etype} “{mention}” in the text is the record: {candidate}")
    q_score = cfg.get("score_question", "How well does this record match the {etype} “{mention}” in the text? Record: {candidate}")
    levels = [Option("1", "a different entity"), Option("2", "possibly the same entity"), Option("3", "the same entity")]
    out = {}
    for split in cfg.get("splits", ["train", "test"]):
        rows = []
        for i, row in enumerate(_rows(cfg, split)):
            mention, etype, text, opts, gold, group = _parse(row)
            base = f"{cfg['name']}:{group}:{mention}"
            meta = {"mention": mention, "kind": row.get("kind", "")}
            if mode == "choice":
                rows.append(Example(id=base, group=group, type="choice", state=text, question=q_choice.format(etype=etype, mention=mention),
                                    options=opts, answer=gold, meta=meta))
                continue
            for j, o in enumerate(o for o in opts if o.key != NONE):
                cand = f"{o.key} — {o.text}" if o.text else o.key
                if mode == "noul":
                    rows.append(Example(id=f"{base}#{j}", group=group, type="noul", state=text,
                                        question=q_noul.format(etype=etype, mention=mention, candidate=cand),
                                        answer=1.0 if o.key == gold else 0.0, meta={**meta, "candidate": o.key}))
                elif mode == "score":
                    rows.append(Example(id=f"{base}#{j}", group=group, type="score", state=text,
                                        question=q_score.format(etype=etype, mention=mention, candidate=cand),
                                        options=levels, answer="3" if o.key == gold else "1", meta={**meta, "candidate": o.key}))
                else:
                    raise ValueError(f"recon: unknown 'as' {mode!r}")
        out[split] = rows
    return out
