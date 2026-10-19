"""Prompt approaches: different ways to put the same decision in front of a model.

An approach rewrites a canonical :class:`~s1bench.schema.Example` into the example the model actually sees,
possibly as a different question type (a Score asked as a Choice over the levels; a Noul asked as a
true/false Choice). The model adapter then renders that rewritten example in its own native template.
Every rewrite records how to map the model's answer back (``meta["_back"]``), so metrics are always
computed on the original question.

Pick one approach per question type in a run config::

    approach: {choice: in_state, score: rubric, noul: statement}

Add an approach by decorating a function ``(Example, **params) -> Example`` with ``@approach(type, name)``.
"""

from __future__ import annotations

import random
from typing import Callable

from .schema import Example, Option

APPROACHES: dict[tuple[str, str], Callable[..., Example]] = {}
DEFAULT = {"choice": "described", "score": "rubric", "noul": "statement"}


def approach(qtype: str, name: str):
    def deco(fn):
        APPROACHES[(qtype, name)] = fn
        fn.__doc__ = fn.__doc__ or ""
        return fn

    return deco


def apply(ex: Example, spec: dict | None, *, train: bool = False, rng: random.Random | None = None) -> Example:
    """Rewrite ``ex`` with the approach configured for its type. ``spec`` maps type -> name or
    type -> {name: ..., <params>}; a top-level ``shuffle_options`` (training only) permutes options."""
    spec = spec or {}
    conf = spec.get(ex.type, DEFAULT[ex.type])
    if isinstance(conf, str):
        conf = {"name": conf}
    conf = dict(conf)
    name = conf.pop("name")
    fn = APPROACHES.get((ex.type, name))
    if fn is None:
        raise KeyError(f"no approach {name!r} for {ex.type}; have {sorted(n for t, n in APPROACHES if t == ex.type)}")
    out = fn(ex.copy(), **conf)
    out.meta.setdefault("_back", {"type": ex.type, "map": None})
    out.meta["_approach"] = f"{ex.type}:{name}"
    if train and spec.get("shuffle_options") and out.type == "choice" and rng is not None:
        keep_last = [o for o in out.options if o.key in NULL_KEYS]
        rest = [o for o in out.options if o.key not in NULL_KEYS]
        rng.shuffle(rest)
        out.options = rest + keep_last
    return out


NULL_KEYS = {"none", "null", "None"}


def back_probs(rendered: Example, probs: dict[str, float] | float) -> dict[str, float] | float:
    """Map a model's output on the rendered example back to the original question's answer space."""
    back = rendered.meta.get("_back") or {}
    orig = back.get("type", rendered.type)
    m = back.get("map")
    if orig == "noul" and rendered.type == "choice":
        return float(probs.get(m["true"], 0.0))
    if m and isinstance(probs, dict):
        out: dict[str, float] = {}
        for k, p in probs.items():
            out[m.get(k, k)] = out.get(m.get(k, k), 0.0) + p
        return out
    return probs


# ------------------------------------------------------------------------------------------------ choice


@approach("choice", "described")
def choice_described(ex: Example) -> Example:
    """Options carry their descriptions (each option = key + text). The models' default layout."""
    return ex


@approach("choice", "in_state")
def choice_in_state(ex: Example, heading: str = "Options:") -> Example:
    """Move option descriptions into the state as a list; the options become bare keys.
    (Laya's best layout on reconciliation; GLiNER's worst.)"""
    lines = [f"- {o.key}: {o.text}" if o.text else f"- {o.key}" for o in ex.options]
    ex.state = f"{ex.state}\n\n{heading}\n" + "\n".join(lines)
    ex.options = [Option(o.key, "") for o in ex.options]
    return ex


@approach("choice", "bare")
def choice_bare(ex: Example) -> Example:
    """Keys only, no descriptions anywhere (a control: how much do the descriptions carry?)."""
    ex.options = [Option(o.key, "") for o in ex.options]
    return ex


@approach("choice", "text_as_key")
def choice_text_as_key(ex: Example, max_chars: int = 200) -> Example:
    """Use the description itself as the option label (no opaque ids); mapped back to the original keys."""
    seen, opts, m = set(), [], {}
    for o in ex.options:
        label = (o.text or o.key)[:max_chars]
        while label in seen:
            label += " "
        seen.add(label)
        opts.append(Option(label, ""))
        m[label] = o.key
    ex.options = opts
    if ex.answer is not None:
        ex.answer = next(k for k, v in m.items() if v == ex.answer)
    ex.meta["_back"] = {"type": "choice", "map": m}
    return ex


@approach("choice", "lettered")
def choice_lettered(ex: Example) -> Example:
    """Relabel options A, B, C, ... (descriptions kept); mapped back to the original keys."""
    letters = [chr(65 + i) if i < 26 else f"A{i}" for i in range(len(ex.options))]
    m = {letter: o.key for letter, o in zip(letters, ex.options)}
    ex.options = [Option(letter, o.text) for letter, o in zip(letters, ex.options)]
    if ex.answer is not None:
        ex.answer = next(k for k, v in m.items() if v == ex.answer)
    ex.meta["_back"] = {"type": "choice", "map": m}
    return ex


# ------------------------------------------------------------------------------------------------- score


@approach("score", "rubric")
def score_rubric(ex: Example) -> Example:
    """Native Score: the scale levels with what each level means."""
    return ex


@approach("score", "numeric")
def score_numeric(ex: Example) -> Example:
    """Native Score with bare level numbers (no level descriptions)."""
    ex.options = [Option(o.key, "") for o in ex.options]
    return ex


@approach("score", "as_choice")
def score_as_choice(ex: Example, template: str = "{key}: {text}") -> Example:
    """Ask the Score as a Choice over the levels (for models without a Score head, or to compare)."""
    ex.options = [Option(o.key, template.format(key=o.key, text=o.text).rstrip(": ")) for o in ex.options]
    ex.meta["_back"] = {"type": "score", "map": None}
    ex.type = "choice"
    return ex


# -------------------------------------------------------------------------------------------------- noul


@approach("noul", "statement")
def noul_statement(ex: Example) -> Example:
    """Native Noul: the statement as given."""
    return ex


@approach("noul", "question")
def noul_question(ex: Example, prefix: str = "Is it true that ") -> Example:
    """Rephrase the statement as a yes/no question."""
    s = ex.question.rstrip(".?! ")
    ex.question = f"{prefix}{s[:1].lower()}{s[1:]}?"
    return ex


@approach("noul", "as_choice")
def noul_as_choice(ex: Example, true_key: str = "true", false_key: str = "false") -> Example:
    """Ask the Noul as a two-way Choice; P(true) is read back from the true option."""
    ex.options = [Option(true_key, "The statement is true"), Option(false_key, "The statement is false")]
    if ex.answer is not None:
        ex.meta["_soft"] = float(ex.answer)
        ex.answer = true_key if float(ex.answer) >= 0.5 else false_key
    ex.meta["_back"] = {"type": "noul", "map": {"true": true_key}}
    ex.type = "choice"
    return ex
