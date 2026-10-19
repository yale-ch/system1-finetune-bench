"""Summarise a sweep: one row per run plus mean ± sd across seeds.

    python -m s1bench.report runs/ten-models            # markdown to stdout, also writes report.md + report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

HEADLINE = {"choice": "accuracy", "score": "mae_expected", "noul": "balanced_accuracy"}


def load_runs(root: Path) -> list[dict]:
    rows = []
    for m in sorted(root.glob("*/metrics.json")):
        cfg = json.loads((m.parent / "config.json").read_text()) if (m.parent / "config.json").exists() else {}
        met = json.loads(m.read_text())
        for stage in ("zero_shot", "tuned"):
            for ds, res in (met.get(stage) or {}).items():
                for qtype, r in res.items():
                    if not isinstance(r, dict):
                        continue
                    rows.append({"run": m.parent.name, "model": (cfg.get("model") or {}).get("name", "?"), "dataset": ds,
                                 "approach": json.dumps({k: v for k, v in (cfg.get("approach") or {}).items() if k in ("choice", "score", "noul")}),
                                 "seed": (cfg.get("train") or {}).get("seed"), "stage": stage, "type": qtype,
                                 "headline": r.get(HEADLINE[qtype]), "adjusted": r.get("adjusted"), "n": r.get("n"),
                                 "false_link_rate": r.get("false_link_rate"), "minutes": met.get("train_minutes"),
                                 "peak_gib": met.get("peak_gib")})
    return rows


def fmt(xs: list[float]) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    return f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    args = ap.parse_args()
    root = Path(args.root)
    rows = load_runs(root)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["type"], r["model"], r["approach"], r["stage"])].append(r)
    lines = ["| dataset | type | model | approach | stage | seeds | headline | adjusted | false links | train min | peak GiB |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (ds, t, model, ap_, stage), rs in sorted(groups.items()):
        lines.append(f"| {ds} | {t} ({HEADLINE[t]}) | {model} | {ap_} | {stage} | {len(rs)} | {fmt([r['headline'] for r in rs])} | "
                     f"{fmt([r['adjusted'] for r in rs])} | {fmt([r['false_link_rate'] for r in rs])} | "
                     f"{fmt([r['minutes'] for r in rs if stage == 'tuned'])} | {fmt([r['peak_gib'] for r in rs if stage == 'tuned'])} |")
    md = "\n".join(lines)
    print(md)
    (root / "report.md").write_text(md + "\n")
    (root / "report.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
