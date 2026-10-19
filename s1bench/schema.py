"""The one data format every dataset is converted to and every model adapter consumes.

A System One *decision* asks one typed question about a ``state`` (text):

* ``choice`` -- pick exactly one of ``options`` (``answer`` = the option key);
* ``score``  -- rate the state on an ordinal scale; ``options`` are the scale levels in order, low to high
  (keys like ``"1".."5"``, ``text`` = what the level means); ``answer`` = the level key;
* ``noul``   -- is ``question`` (a statement about the state) true? ``answer`` = 1.0 / 0.0 (a float in
  between is a soft label). ``options`` is empty.

One JSON object per line::

    {"id": "p1#0", "group": "p1", "type": "choice", "state": "...", "question": "Which record ...?",
     "options": [{"key": "wikidata:Q42", "text": "Douglas Adams, English writer (1952-2001)"},
                 {"key": "none", "text": "None of the candidates"}],
     "answer": "wikidata:Q42", "split": "train", "meta": {...}}

``group`` keeps related decisions (several questions about one record) on the same side of every split.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable, Iterator

TYPES = ("choice", "score", "noul")


@dataclass(frozen=True)
class Option:
    key: str
    text: str = ""


@dataclass
class Example:
    id: str
    type: str
    state: str
    question: str
    options: list[Option] = field(default_factory=list)
    answer: str | float | None = None
    group: str = ""
    split: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.options = [o if isinstance(o, Option) else Option(**o) for o in self.options]
        if not self.group:
            self.group = self.id

    # ------------------------------------------------------------------------------------------ checks

    def validate(self) -> None:
        if self.type not in TYPES:
            raise ValueError(f"{self.id}: type {self.type!r} not in {TYPES}")
        if self.type == "noul":
            if self.options:
                raise ValueError(f"{self.id}: noul questions take no options")
            if self.answer is not None and not 0.0 <= float(self.answer) <= 1.0:
                raise ValueError(f"{self.id}: noul answer must be in [0, 1], got {self.answer!r}")
            return
        keys = [o.key for o in self.options]
        if len(keys) < 2:
            raise ValueError(f"{self.id}: {self.type} needs at least two options")
        if len(set(keys)) != len(keys):
            raise ValueError(f"{self.id}: duplicate option keys {keys}")
        if self.answer is not None and self.answer not in keys:
            raise ValueError(f"{self.id}: answer {self.answer!r} is not an option key")

    @property
    def keys(self) -> list[str]:
        return [o.key for o in self.options]

    def gold_index(self) -> int:
        return self.keys.index(self.answer)

    def copy(self, **changes) -> "Example":
        return replace(self, meta=dict(self.meta), **changes)

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d["meta"]:
            d.pop("meta")
        return d


def read_jsonl(path: str | Path) -> list[Example]:
    out = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    ex = Example(**json.loads(line))
                    ex.validate()
                except Exception as e:  # noqa: BLE001 -- name the line, then fail
                    raise ValueError(f"{path}:{n}: {e}") from e
                out.append(ex)
    return out


def write_jsonl(examples: Iterable[Example], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_dicts(path: str | Path) -> Iterator[dict]:
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)
