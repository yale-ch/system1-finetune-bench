"""Metrics, always on the original question type.

A prediction is ``{"id", "type", "probs"}`` where ``probs`` is ``{option_key: p}`` for choice/score and
``p_true`` (float) for noul.

* choice: accuracy, log loss, ECE; if a ``none``-like option exists, also linking accuracy / abstention
  recall / false-link rate (the reconciliation view).
* score: exact accuracy, MAE of the argmax level and of the expected level, within-one accuracy.
* noul: balanced accuracy at 0.5, Brier, log loss, AUROC.

``adjusted`` follows S1MB's spirit: 0 = the best trivial baseline (majority class / constant score),
100 = perfect, clipped to [0, 100], so numbers are comparable across datasets.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .approaches import NULL_KEYS
from .schema import Example

EPS = 1e-9


def _argmax(probs: dict[str, float]) -> str:
    return max(probs.items(), key=lambda kv: kv[1])[0]


def ece(conf_correct: list[tuple[float, bool]], bins: int = 10) -> float:
    if not conf_correct:
        return 0.0
    buckets = defaultdict(list)
    for c, ok in conf_correct:
        buckets[min(int(c * bins), bins - 1)].append((c, ok))
    total = len(conf_correct)
    return sum(len(b) / total * abs(sum(c for c, _ in b) / len(b) - sum(ok for _, ok in b) / len(b)) for b in buckets.values())


def choice_metrics(exs: list[Example], preds: list[dict]) -> dict:
    n = len(exs)
    right, nll, cc = 0, 0.0, []
    null_gold = null_right = false_links = ans = ans_right = 0
    for ex, p in zip(exs, preds):
        pick = _argmax(p["probs"])
        ok = pick == ex.answer
        right += ok
        nll -= math.log(max(p["probs"].get(ex.answer, 0.0), EPS))
        cc.append((p["probs"][pick], ok))
        if any(k in NULL_KEYS for k in ex.keys):
            if ex.answer in NULL_KEYS:
                null_gold += 1
                null_right += pick in NULL_KEYS
                false_links += pick not in NULL_KEYS
            else:
                ans += 1
                ans_right += ok
    majority = Counter(ex.answer for ex in exs).most_common(1)[0][1] / n if n else 0
    uniform = sum(1 / len(ex.options) for ex in exs) / n if n else 0
    base = max(majority, uniform)
    acc = right / n if n else 0
    out = {"n": n, "accuracy": acc, "log_loss": nll / max(n, 1), "ece": ece(cc),
           "adjusted": _adjust(acc, base)}
    if null_gold or ans:
        out.update({"link_accuracy": ans_right / max(ans, 1), "abstention_recall": null_right / max(null_gold, 1),
                    "false_link_rate": false_links / max(null_gold, 1), "null_gold": null_gold})
    return out


def score_metrics(exs: list[Example], preds: list[dict]) -> dict:
    n = len(exs)
    exact = within1 = 0
    mae_arg = mae_exp = 0.0
    for ex, p in zip(exs, preds):
        levels = ex.keys
        gold = levels.index(ex.answer)
        pick = levels.index(_argmax(p["probs"]))
        expect = sum(i * p["probs"].get(k, 0.0) for i, k in enumerate(levels)) / max(sum(p["probs"].get(k, 0.0) for k in levels), EPS)
        exact += pick == gold
        within1 += abs(pick - gold) <= 1
        mae_arg += abs(pick - gold)
        mae_exp += abs(expect - gold)
    golds = sorted(ex.keys.index(ex.answer) for ex in exs)
    median = golds[len(golds) // 2] if golds else 0
    base_mae = sum(abs(g - median) for g in golds) / max(n, 1)
    mae = mae_exp / max(n, 1)
    return {"n": n, "accuracy": exact / max(n, 1), "within_one": within1 / max(n, 1), "mae_argmax": mae_arg / max(n, 1),
            "mae_expected": mae, "baseline_mae": base_mae,
            "adjusted": max(0.0, min(100.0, 100 * (1 - mae / base_mae))) if base_mae else 0.0}


def noul_metrics(exs: list[Example], preds: list[dict]) -> dict:
    pairs = [(float(p["probs"]), float(ex.answer)) for ex, p in zip(exs, preds)]
    pos = [(p, y) for p, y in pairs if y >= 0.5]
    neg = [(p, y) for p, y in pairs if y < 0.5]
    tpr = sum(p >= 0.5 for p, _ in pos) / max(len(pos), 1)
    tnr = sum(p < 0.5 for p, _ in neg) / max(len(neg), 1)
    bal = (tpr + tnr) / 2 if pos and neg else (tpr if pos else tnr)
    brier = sum((p - y) ** 2 for p, y in pairs) / max(len(pairs), 1)
    ll = -sum(y * math.log(max(p, EPS)) + (1 - y) * math.log(max(1 - p, EPS)) for p, y in pairs) / max(len(pairs), 1)
    auc = _auroc([p for p, _ in pos], [p for p, _ in neg])
    return {"n": len(pairs), "balanced_accuracy": bal, "brier": brier, "log_loss": ll, "auroc": auc,
            "positives": len(pos), "adjusted": max(0.0, min(100.0, 100 * (bal - 0.5) / 0.5))}


def _auroc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    ranked = sorted([(p, 1) for p in pos] + [(p, 0) for p in neg])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if ranked[k][1])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _adjust(acc: float, base: float) -> float:
    return max(0.0, min(100.0, 100 * (acc - base) / (1 - base))) if base < 1 else 0.0


def evaluate(exs: list[Example], preds: list[dict]) -> dict:
    """Metrics per question type plus the mean adjusted score over the types present."""
    by: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for ex, p in zip(exs, preds):
        by[ex.type][0].append(ex)
        by[ex.type][1].append(p)
    fn = {"choice": choice_metrics, "score": score_metrics, "noul": noul_metrics}
    out = {t: fn[t](e, p) for t, (e, p) in sorted(by.items())}
    out["adjusted_mean"] = sum(m["adjusted"] for m in out.values()) / max(len(out), 1)
    return out
