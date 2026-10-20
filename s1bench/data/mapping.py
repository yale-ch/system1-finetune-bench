"""Field-mapping datasets: describe a source table and how its columns become a decision, no code needed.

    name: boolq
    type: noul
    source: {hf: google/boolq}              # or {jsonl: path} / {csv: path} / {parquet: path}; per split:
    splits: {train: train, test: validation} #   source: {jsonl: {train: a.jsonl, test: b.jsonl}}
    state: "{passage}"                      # str.format template over the row
    question: "{question}"                  # for noul: the statement; for choice/score: the question
    answer: {field: answer, map: {true: 1.0, false: 0.0}}
    id: "{idx}"                             # optional; default <name>:<split>:<row number>
    group: "{title}"                        # optional; keeps related rows on one side of the split

Options, for choice and score:

    options:                                # fixed options (e.g. the levels of a score scale) ...
      - {key: "1", text: very negative}
      - {key: "2", text: negative}
    options: {field: choices, key: "{label}", text: "{text}"}   # ... or a list column, templated per item
    options: {keys: choices.label, texts: choices.text}           # ... or parallel list columns (dotted paths)

``answer`` is either a template string (``"{label}"``) or ``{field, map}``; for choice/score it must yield an
option key; for noul a float in [0, 1].
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import Example, Option


class _Row(dict):
    def __missing__(self, key):
        raise KeyError(f"row has no field {key!r}; fields: {sorted(self)}")


def _fmt(template, row: dict):
    if template is None:
        return None
    if not isinstance(template, str):
        return template
    return template.format_map(_Row(row))


def _rows(source: dict, split_name: str):
    if "hf" in source:
        import datasets

        kw = {k: source[k] for k in ("name", "revision", "data_files") if k in source}
        ds = datasets.load_dataset(source["hf"], split=split_name, **kw)
        yield from ds
        return
    for kind in ("jsonl", "csv", "parquet"):
        if kind in source:
            path = source[kind]
            if isinstance(path, dict):
                path = path[split_name]
            path = Path(path)
            if kind == "jsonl":
                with open(path) as fh:
                    for line in fh:
                        if line.strip():
                            yield json.loads(line)
            elif kind == "csv":
                import csv

                with open(path, newline="") as fh:
                    yield from csv.DictReader(fh)
            else:
                import pyarrow.parquet as pq

                yield from pq.read_table(path).to_pylist()
            return
    raise ValueError(f"unknown source {source}")


def _answer(spec, row):
    if isinstance(spec, dict):
        v = _get(row, spec["field"])
        m = spec.get("map")
        if m is not None:
            if v not in m and str(v) in {str(k) for k in m}:
                v = next(val for k, val in m.items() if str(k) == str(v))
            else:
                v = m[v]
        return v
    return _fmt(spec, row)


def _get(row: dict, path: str):
    """``a.b`` reads row["a"]["b"]."""
    v = row
    for part in path.split("."):
        v = v[part]
    return v


def _options(spec, row) -> list[Option]:
    if spec is None:
        return []
    if isinstance(spec, list):
        return [Option(str(o["key"]), _fmt(o.get("text", ""), row)) for o in spec]
    if "field" in spec:
        items = _get(row, spec["field"])
        return [Option(str(_fmt(spec.get("key", "{_item}"), {**row, **(it if isinstance(it, dict) else {}), "_item": it, "_i": i})),
                       _fmt(spec.get("text", ""), {**row, **(it if isinstance(it, dict) else {}), "_item": it, "_i": i}) or "")
                for i, it in enumerate(items)]
    keys = _get(row, spec["keys"]) if isinstance(spec["keys"], str) else spec["keys"]
    texts = _get(row, spec["texts"]) if isinstance(spec.get("texts"), str) else spec.get("texts") or [""] * len(keys)
    return [Option(str(k), str(t)) for k, t in zip(keys, texts)]


def build_mapped(cfg: dict) -> dict[str, list[Example]]:
    qtype = cfg["type"]
    splits = cfg.get("splits") or {"train": "train"}
    out = {}
    for split, src_split in splits.items():
        rows = []
        for i, row in enumerate(_rows(cfg["source"], src_split)):
            row = dict(row, idx=i)
            if cfg.get("filter") and not eval(cfg["filter"], {}, {"row": row}):  # noqa: S307 -- trusted local config
                continue
            ans = _answer(cfg["answer"], row)
            if qtype == "noul":
                ans = float(ans)
            else:
                ans = str(ans)
            ex = Example(
                id=_fmt(cfg.get("id"), row) or f"{cfg['name']}:{split}:{i}",
                type=qtype,
                state=_fmt(cfg["state"], row),
                question=_fmt(cfg.get("question", ""), row),
                options=_options(cfg.get("options"), row),
                answer=ans,
                group=_fmt(cfg.get("group"), row) or "",
            )
            rows.append(ex)
        out[split] = rows
    return out
