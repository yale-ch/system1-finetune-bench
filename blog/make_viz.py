"""Animated SVG diagrams for the blog post comparing eighteen fine-tuned System One models.

    python blog/make_viz.py OUT_DIR

Same visual language as the earlier Europeana post (cream ground, Source Sans 3, brown accent, "STEP n / N"
captions, CSS keyframe loops); ``render_video.py`` turns each SVG into an MP4 and a GIF. Every number comes from
``blog/results.json`` (built from runs/ten-models and runs/larger-models) or, for the prompt chart, from
runs/prompt-approaches (copied below with its source).
"""

from __future__ import annotations

import json
import math
import sys
from html import escape
from pathlib import Path

W, H = 1200, 760
BG, INK, ACCENT, MUTED, RULE, CARD = "#fffdf6", "#1b1a17", "#7d5a2a", "#867c68", "#e6dcc6", "#fbf6ea"
GOOD, WARN, BAD, BLUE, PURPLE = "#4f7a4a", "#c8862a", "#a8432e", "#46607a", "#7b5a8c"
LIGHT = "#cfc5ae"
SANS = "'Source Sans 3', 'Helvetica Neue', Helvetica, Arial, sans-serif"
HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())

# display name, family, total parameters (billions), note
MODELS = {
    "gliner-decide": ("GLiNER2.5-Decide", "label", 0.34, ""),
    "laya": ("Laya", "marker", 0.42, ""),
    "von": ("von", "marker", 0.40, ""),
    "bekko-400m": ("bekko-400m", "marker", 0.40, ""),
    "decider-2b": ("decider-2b", "letters", 1.9, ""),
    "plumb-4b": ("plumb-4b", "letters", 4.2, ""),
    "decider-4b": ("decider-4b", "letters", 4.2, ""),
    "kev-4b": ("kev-4b", "pointer", 4.2, ""),
    "openjev-nli-4b": ("openjev-NLI-4B", "candidate", 4.5, ""),
    "jevk5-9b": ("JevK5-9B", "letters", 9, ""),
    "jev-9b": ("JEV-9B", "slots", 9, ""),
    "open-jev-9b": ("Open-Jev-9B", "candidate", 9, ""),
    "kev-9b": ("kev-9b", "pointer", 9, ""),
    "gev-26b": ("GEV-26B", "slots", 26, "MoE"),
    "jev-27b": ("JEV-27B", "slots", 27, ""),
    "open-jev-27b": ("Open-Jev-27B", "candidate", 27, ""),
    "kev-27b": ("kev-27b", "pointer", 27, ""),
    "decider-35b-a3b": ("decider-35B-A3B", "letters", 35, "MoE"),
}
FAMILY = {  # name, colour
    "label": ("label classifier", BLUE),
    "marker": ("option markers", PURPLE),
    "letters": ("lettered options", WARN),
    "slots": ("slot head", ACCENT),
    "pointer": ("pointer head", GOOD),
    "candidate": ("one prompt per option", BAD),
}


class Svg:
    """Collects elements and their (start, end) visibility windows inside one looping timeline."""

    def __init__(self, duration: float, label: str):
        self.d, self.label, self.parts, self.css, self.n = duration, label, [], [], 0

    def _pct(self, t: float) -> str:
        return f"{100 * max(0.0, min(t, self.d)) / self.d:.3f}%"

    def anim(self, start: float, end: float | None = None, rise: int = 6, fade: float = 0.5, kind: str = "fade") -> str:
        self.n += 1
        k = f"k{self.n}"
        end = self.d - 0.9 if end is None else end
        a, b, c, e = start, start + fade, end, end + fade
        if kind == "grow":
            hidden, shown = "transform:scaleX(0)", "transform:scaleX(1)"
            self.css.append(f"@keyframes {k}{{0%,{self._pct(a)}{{opacity:1;{hidden}}}{self._pct(b + 0.6)},{self._pct(c)}"
                            f"{{opacity:1;{shown}}}{self._pct(e)},100%{{opacity:0;{shown}}}}}"
                            f".{k}{{animation:{k} {self.d}s linear infinite both;transform-box:fill-box;transform-origin:left center}}")
        else:
            off = f"transform:translateY({rise}px)" if rise else "transform:none"
            self.css.append(f"@keyframes {k}{{0%,{self._pct(a)}{{opacity:0;{off}}}{self._pct(b)},{self._pct(c)}"
                            f"{{opacity:1;transform:none}}{self._pct(e)},100%{{opacity:0;transform:none}}}}"
                            f".{k}{{animation:{k} {self.d}s linear infinite both}}")
        return k

    def group(self, cls: str, inner: str) -> None:
        self.parts.append(f'<g class="{cls}">{inner}</g>')

    def write(self, path: Path) -> None:
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
               f'aria-label="{escape(self.label)}"><style>{"".join(self.css)}</style>'
               f'<rect width="{W}" height="{H}" fill="{BG}"/>' + "".join(self.parts) + "</svg>")
        path.write_text(svg, encoding="utf-8")


def text(x, y, s, size=18, fill=INK, weight=400, anchor="start", spacing=0, italic=False) -> str:
    st = ' font-style="italic"' if italic else ""
    sp = f' letter-spacing="{spacing}"' if spacing else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{SANS}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}"{sp}{st}>{escape(s)}</text>')


def rect(x, y, w, h, fill="none", stroke="none", r=6, sw=1.5, extra="") -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'


def line(x1, y1, x2, y2, stroke=RULE, sw=1.5, dash="") -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"{d}/>'


def circle(x, y, r, fill, stroke="none", sw=0) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def captions(svg: Svg, steps: list[tuple[float, float, str]]) -> None:
    n = len(steps)
    for i, (a, b, body) in enumerate(steps, 1):
        svg.group(svg.anim(a, b), text(64, 44, f"STEP {i} / {n}", 14, ACCENT, 700, spacing=2) + text(64, 74, body, 21))


def m(key, field):
    return R[key][field][0]


# ------------------------------------------------------------------------------------------------- 1 readers


def readers(out: Path) -> None:
    """The same decision, read five ways."""
    s = Svg(31, "Animated diagram. One decision, a record naming 'Boom BK' with two candidates and 'none', shown to five "
                "kinds of decision model: a label classifier that scores each option label; option-marker encoders that "
                "score a marker token placed before each option; decoders that list lettered options and read the "
                "next-token scores of the letters; a pointer head that compares each option's closing token with a final "
                "decide token; and models that write one prompt per option and ask whether it is correct.")
    T = [(0.2, 5.0), (5.3, 10.2), (10.5, 15.4), (15.7, 20.6), (20.9, 25.8), (26.1, 30.1)]
    captions(s, [
        (*T[0], "Every model gets the same decision: a record, its candidates, and the option ‘none’."),
        (*T[1], "A label classifier reads the options as labels and scores each one against the record."),
        (*T[2], "Option-marker encoders put a marker before each option and score the markers in one pass."),
        (*T[3], "Lettered prompts list the options as A, B, C and read how strongly the model would answer each letter."),
        (*T[4], "A pointer head compares a final ‘decide’ token with the end of each option."),
        (*T[5], "Or ask once per option: ‘is this the right answer?’, and compare the yes-scores."),
    ])
    # the decision card (left, stays)
    card = s.anim(0.5)
    rows = [rect(64, 112, 360, 560, CARD, RULE, 10),
            text(88, 146, "THE DECISION", 13, MUTED, 700, spacing=1.5),
            text(88, 184, "Record", 15, MUTED, 600),
            text(88, 210, "Boom BK · collector", 19, INK, 700),
            text(88, 236, "Ceanothus, Boskoop, 31 Aug 1934", 16),
            text(88, 260, "Naturalis herbarium", 16, MUTED),
            text(88, 304, "Question", 15, MUTED, 600),
            text(88, 330, "Which candidate is the person?", 17),
            text(88, 374, "Options", 15, MUTED, 600)]
    opts = [("A", "Boudewijn Karel Boom", "Dutch botanist, 1903–1980"), ("B", "Bert Boom", "Dutch cyclist, born 1938"),
            ("C", "none of these", "the person is not listed")]
    for i, (k, a, b) in enumerate(opts):
        y = 392 + i * 86
        rows += [rect(88, y, 312, 72, "#f3ead6" if k == "C" else BG, RULE, 8),
                 text(104, y + 30, k, 18, ACCENT, 700), text(130, y + 30, a, 17, INK, 700, italic=k == "C"),
                 text(130, y + 54, b, 14, MUTED)]
    s.group(card, "".join(rows))

    X0, X1 = 470, 1136

    def strip(t0, t1, title, models, segs, readout, note):
        """segs: (label, width, kind) kind in {"text", "opt", "mark", "read"}; reads get an arrow to a score."""
        g = s.anim(t0 + 0.2, t1)
        parts = [text(X0, 150, title, 26, INK, 700), text(X0, 180, models, 16, MUTED, 600)]
        x = X0
        reads = []
        for lab, w, kind in segs:
            fill = {"text": CARD, "opt": "#f3ead6", "mark": "#efe4ff", "read": "#e9f1e6"}[kind]
            stroke = {"text": RULE, "opt": RULE, "mark": PURPLE, "read": GOOD}[kind]
            parts.append(rect(x, 300, w - 6, 56, fill, stroke, 7, 1.5 if kind in ("text", "opt") else 2))
            parts.append(text(x + (w - 6) / 2, 334, lab, 15, INK, 600, anchor="middle"))
            if kind in ("mark", "read"):
                reads.append(x + (w - 6) / 2)
            x += w
        parts.append(text(X0, 280, "WHAT THE MODEL READS", 13, MUTED, 700, spacing=1.5))
        s.group(g, "".join(parts))
        # arrows up to three score bars
        bars = s.anim(t0 + 1.4, t1)
        bp = [text(X0, 424, "WHERE IT SCORES THE OPTIONS", 13, MUTED, 700, spacing=1.5)]
        for i, cx in enumerate(reads[:3]):
            bp.append(line(cx, 362, cx, 398, GOOD, 2))
            bp.append(f'<path d="M{cx - 5:.1f} 392 L{cx:.1f} 400 L{cx + 5:.1f} 392" fill="none" stroke="{GOOD}" stroke-width="2"/>')
        for i, (k, w) in enumerate([("A", 250), ("B", 26), ("C", 70)]):
            y = 444 + i * 38
            bp.append(text(X0, y + 21, k, 17, ACCENT, 700))
            bp.append(rect(X0 + 28, y + 6, w, 20, GOOD if k == "A" else LIGHT, r=3))
        bp.append(text(X0 + 300, 470, "softmax over the options:", 15, MUTED))
        bp.append(text(X0 + 300, 492, "one probability each", 15, MUTED))
        s.group(bars, "".join(bp))
        s.group(s.anim(t0 + 2.2, t1), text(X0, 586, readout, 17, INK, 600) + text(X0, 612, note, 15, MUTED, italic=True))

    strip(*T[1], "Label classifier", "GLiNER2.5-Decide",
          [("A label", 112, "read"), ("B label", 112, "read"), ("C label", 112, "read"), ("the record text …", 330, "text")],
          "One encoder pass over labels + text; a small network scores each label.", "Options and their descriptions are part of the label list.")
    strip(*T[2], "Option markers", "Laya · von · bekko-400m",
          [("question + record …", 288, "text"), ("◆ A …", 126, "mark"), ("◆ B …", 126, "mark"), ("◆ C …", 126, "mark")],
          "One encoder pass; the score is read at each option's marker.", "von and bekko stop the options from seeing each other.")
    strip(*T[3], "Lettered options", "plumb · JevK5 · decider · JEV · GEV",
          [("record + question …", 250, "text"), ("A) …", 96, "opt"), ("B) …", 96, "opt"), ("C) …", 96, "opt"), ("answer:", 128, "read")],
          "Read once, at the end: how much the model wants to say A, B or C.", "JEV and GEV read a trained 24-slot head there instead of the letters.")
    strip(*T[4], "Pointer head", "kev-4b · kev-9b · kev-27b",
          [("record + question …", 210, "text"), ("⟨A …⟩", 110, "read"), ("⟨B …⟩", 110, "read"), ("⟨C …⟩", 110, "read"), ("decide", 126, "mark")],
          "Each option's closing token is compared with the final decide token.", "Options are wrapped in reserved tokens the text can't contain.")
    # per-candidate: three prompts stacked
    t0, t1 = T[5]
    g = s.anim(t0 + 0.2, t1)
    parts = [text(X0, 150, "One prompt per option", 26, INK, 700), text(X0, 180, "Open-Jev-9B · Open-Jev-27B · openjev-NLI-4B", 16, MUTED, 600),
             text(X0, 228, "WHAT THE MODEL READS · THREE TIMES", 13, MUTED, 700, spacing=1.5)]
    for i, k in enumerate("ABC"):
        y = 246 + i * 64
        parts += [rect(X0, y, 400, 52, CARD, RULE, 7), text(X0 + 200, y + 32, "record + question …", 15, INK, 600, anchor="middle"),
                  rect(X0 + 406, y, 150, 52, "#f3ead6", RULE, 7), text(X0 + 481, y + 32, f"is {k} right?", 15, INK, 600, anchor="middle"),
                  rect(X0 + 562, y, 98, 52, "#e9f1e6", GOOD, 7, 2), text(X0 + 611, y + 32, "yes?", 15, INK, 600, anchor="middle")]
    s.group(g, "".join(parts))
    bars = s.anim(t0 + 1.4, t1)
    bp = [text(X0, 470, "COMPARED ACROSS THE THREE PROMPTS", 13, MUTED, 700, spacing=1.5)]
    for i, (k, w) in enumerate([("A", 250), ("B", 26), ("C", 70)]):
        y = 486 + i * 32
        bp += [text(X0, y + 19, k, 17, ACCENT, 700), rect(X0 + 28, y + 4, w, 18, GOOD if k == "A" else LIGHT, r=3)]
    s.group(bars, "".join(bp))
    s.group(s.anim(t0 + 2.2, t1), text(X0, 612, "Simple and accurate, but the record is read once per candidate.", 17, INK, 600)
            + text(X0, 638, "Ten candidates means ten passes: the slowest design to train.", 15, MUTED, italic=True))
    s.group(s.anim(1.0), text(X0, 712, "Bars are illustrative, not model output.", 14, MUTED, italic=True))
    s.write(out / "readers.svg")


# --------------------------------------------------------------------------------------------- 2 leaderboard


def leaderboard(out: Path) -> None:
    order = sorted(MODELS, key=lambda k: -m(k, "acc"))
    s = Svg(25, "Animated chart. Agreement with the labels on 364 held-out persons, before and after fine-tuning, for "
                "eighteen models (mean of three seeds). " + "; ".join(
                    f"{MODELS[k][0]} {m(k, 'zs'):.1%} to {m(k, 'acc'):.1%}" for k in order) + ".")
    captions(s, [
        (0.2, 5.6, "Before any training, the eighteen models agree with the labels between 40% and 92% of the time."),
        (5.9, 11.8, "Fine-tuned on the same 2,780 decisions, every one lands between 91% and 97%."),
        (12.1, 18.0, "GLiNER2.5-Decide, the model behind our last run, finishes last. kev-4b reaches 97.1%."),
        (18.3, 24.1, "The top six are within one point of each other: about three people out of 364."),
    ])
    X0, X1, lo, hi = 360, 1100, 0.35, 1.0
    xs = lambda v: X0 + (X1 - X0) * (v - lo) / (hi - lo)
    top, rowh = 132, 29.5
    axis = [text(64, 118, "PARAMETERS", 13, MUTED, 700, spacing=1.5), text(X0, 118, "AGREEMENT WITH THE LABELS · 364 HELD-OUT PERSONS · MEAN OF 3 SEEDS", 13, MUTED, 700, spacing=1.5)]
    for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        axis.append(line(xs(v), top, xs(v), top + rowh * 18, RULE, 1))
        axis.append(text(xs(v), top + rowh * 18 + 22, f"{v:.0%}", 15, MUTED, anchor="middle"))
    s.group(s.anim(0.3, rise=0), "".join(axis))
    for i, k in enumerate(order):
        y = top + rowh * i + rowh / 2
        name, fam, params, note = MODELS[k]
        col = FAMILY[fam][1]
        hl = k in ("kev-4b", "gliner-decide")
        lab = text(X0 - 18, y + 6, name, 16, INK, 700 if hl else 600, anchor="end")
        lab += text(64, y + 6, f"{params:g}B" + (" MoE" if note else ""), 13, MUTED)
        s.group(s.anim(0.6 + 0.08 * i, rise=0), lab + circle(xs(m(k, "zs")), y, 6, LIGHT))
        s.group(s.anim(6.2 + 0.12 * i, rise=0), line(xs(m(k, "zs")) + 7, y, xs(m(k, "acc")) - 7, y, col, 2, "2 4"))
        sd = R[k]["acc"][1]
        s.group(s.anim(6.6 + 0.12 * i, rise=0), line(xs(m(k, "acc") - sd), y, xs(m(k, "acc") + sd), y, col, 3)
                + circle(xs(m(k, "acc")), y, 7, col) + text(xs(m(k, "acc")) + 14, y + 5, f"{m(k, 'acc'):.1%}", 14, INK, 600))
        if hl:
            s.group(s.anim(12.4, 17.9), rect(40, y - 13, 1130, 26, "none", GOOD if k == "kev-4b" else BAD, 6, 2))
    # bracket the top six (to the right of the value labels)
    yb0, yb1, bx = top + 6, top + rowh * 6 - 6, 1158
    s.group(s.anim(18.6, rise=0), line(bx, yb0, bx, yb1, GOOD, 2.5) + line(bx - 8, yb0, bx, yb0, GOOD, 2.5) + line(bx - 8, yb1, bx, yb1, GOOD, 2.5)
            + f'<text x="{bx + 20}" y="{(yb0 + yb1) / 2:.1f}" font-family="{SANS}" font-size="15" font-weight="700" fill="{GOOD}" '
              f'text-anchor="middle" transform="rotate(90 {bx + 20} {(yb0 + yb1) / 2:.1f})">within 1 point</text>')
    # legend
    lg = []
    x = 64
    for fam, (lab, col) in FAMILY.items():
        lg += [circle(x + 6, 718, 6, col), text(x + 18, 723, lab, 14, MUTED, 600)]
        x += 30 + 7.4 * len(lab)
    lg += [circle(x + 6, 718, 6, LIGHT), text(x + 18, 723, "zero-shot", 14, MUTED, 600)]
    s.group(s.anim(0.4, rise=0), "".join(lg))
    s.write(out / "leaderboard.svg")


# --------------------------------------------------------------------------------------------- 3 false links


def false_links(out: Path) -> None:
    order = sorted(MODELS, key=lambda k: m(k, "fl"))
    g_fl, k_fl = m("gliner-decide", "fl"), m("kev-4b", "fl")
    s = Svg(24, f"Animated chart. 90 of the 364 held-out persons have no correct candidate. GLiNER2.5-Decide links about "
                f"{round(90 * g_fl)} of them to someone anyway ({g_fl:.1%}); kev-4b about {round(90 * k_fl)} ({k_fl:.1%}). "
                "False-link rates for all eighteen models: " + "; ".join(f"{MODELS[k][0]} {m(k, 'fl'):.1%}" for k in order) + ".")
    captions(s, [
        (0.2, 5.6, "90 of the 364 test people have no correct candidate. For them, the right answer is ‘none’."),
        (5.9, 11.4, f"GLiNER2.5-Decide links about {round(90 * g_fl)} of them to someone anyway."),
        (11.7, 17.4, f"kev-4b links about {round(90 * k_fl)}: one in twenty-five instead of one in five."),
        (17.7, 23.1, "Across all eighteen, the large decoders are the cautious ones; the small encoders link too readily."),
    ])

    def grid(x0, y0, n_red, t_base, t_red, title, color):
        s.group(s.anim(t_base, rise=0), text(x0, y0 - 18, title, 15, INK, 700))
        for i in range(90):
            cx, cy = x0 + 9 + (i % 15) * 26, y0 + 9 + (i // 15) * 26
            s.group(s.anim(t_base + 0.008 * i, rise=0), circle(cx, cy, 8, LIGHT))
        for i in range(n_red):
            j = [3, 17, 31, 44, 52, 60, 8, 23, 38, 71, 79, 86, 12, 27, 49, 57, 66, 74, 83, 5][i]
            cx, cy = x0 + 9 + (j % 15) * 26, y0 + 9 + (j // 15) * 26
            s.group(s.anim(t_red + 0.12 * i, rise=0), circle(cx, cy, 8, color))

    grid(64, 150, round(90 * g_fl), 0.6, 6.2, f"GLiNER2.5-Decide · {g_fl:.1%} false links", BAD)
    grid(64, 446, round(90 * k_fl), 0.6, 12.0, f"kev-4b · {k_fl:.1%} false links", BAD)
    s.group(s.anim(0.8), text(64, 650, "Each dot: one test person with no correct candidate.", 14, MUTED, italic=True)
            + text(64, 670, "Red: linked to a candidate anyway (mean of three seeds).", 14, MUTED, italic=True))
    # bars
    X0, top, rowh = 650, 136, 29.5
    xs = lambda v: X0 + 400 * v / 0.25
    s.group(s.anim(17.9, rise=0), text(X0, 118, "FALSE-LINK RATE · TUNED · MEAN OF 3 SEEDS", 13, MUTED, 700, spacing=1.5))
    for i, k in enumerate(order):
        y = top + rowh * i
        name, fam, *_ = MODELS[k]
        col = FAMILY[fam][1]
        t = 18.0 + 0.1 * i
        s.group(s.anim(t, rise=0), text(X0 - 12, y + 19, name, 15, INK, 700 if k in ("kev-4b", "gliner-decide") else 600, anchor="end"))
        s.group(s.anim(t + 0.1, kind="grow"), rect(X0, y + 6, max(xs(m(k, "fl")) - X0, 2), 18, col, r=3))
        s.group(s.anim(t + 0.5, rise=0), text(xs(m(k, "fl")) + 8, y + 20, f"{m(k, 'fl'):.1%}", 14, INK, 600))
    s.write(out / "false-links.svg")


# ---------------------------------------------------------------------------------------------------- 4 size


def size(out: Path) -> None:
    s = Svg(24, "Animated scatter of agreement after fine-tuning against model size. Encoders under half a billion "
                "parameters reach 91–93%; models of two to four billion 94–97%; nine to thirty-five billion up to 97.4%. "
                "kev-4b (97.1%) is within 0.3 points of kev-27b (97.4%) with a sixth of the parameters.")
    captions(s, [
        (0.2, 5.6, "The four encoders, all under half a billion parameters, land at 91–93%."),
        (5.9, 11.4, "Two to four billion parameters: 94–97%. Most of the gain is here."),
        (11.7, 17.4, "Nine to thirty-five billion: up to 97.4%, and no further. Bigger stops paying."),
        (17.7, 23.1, "kev-4b is within 0.3 points of kev-27b with a sixth of the parameters."),
    ])
    X0, X1, Y0, Y1 = 150, 1020, 650, 150
    lx0, lx1 = math.log10(0.25), math.log10(45)
    xs = lambda p: X0 + (X1 - X0) * (math.log10(p) - lx0) / (lx1 - lx0)
    ys = lambda v: Y0 - (Y0 - Y1) * (v - 0.90) / 0.08
    ax = [text(X0 - 10, Y1 - 26, "AGREEMENT AFTER FINE-TUNING · MEAN OF 3 SEEDS", 13, MUTED, 700, spacing=1.5)]
    for v in (0.90, 0.92, 0.94, 0.96, 0.98):
        ax += [line(X0, ys(v), X1, ys(v), RULE, 1), text(X0 - 12, ys(v) + 5, f"{v:.0%}", 15, MUTED, anchor="end")]
    for p, lab in ((0.3, "0.3B"), (1, "1B"), (3, "3B"), (10, "10B"), (30, "30B")):
        ax += [line(xs(p), Y0, xs(p), Y0 + 6, MUTED, 1), text(xs(p), Y0 + 26, lab, 15, MUTED, anchor="middle")]
    ax.append(text((X0 + X1) / 2, Y0 + 56, "total parameters (log scale)", 16, MUTED, anchor="middle"))
    s.group(s.anim(0.3, rise=0), "".join(ax))
    jitter = {"von": -10, "bekko-400m": 10, "plumb-4b": -9, "decider-4b": 0, "kev-4b": 9, "jevk5-9b": -13, "jev-9b": -4,
              "open-jev-9b": 4, "kev-9b": 13, "jev-27b": -8, "open-jev-27b": 0, "kev-27b": 8}
    labels = {  # dx, dy, anchor
        "gliner-decide": (-12, 5, "end"), "laya": (14, 10, "start"), "von": (-12, -6, "end"), "bekko-400m": (14, 5, "start"),
        "decider-2b": (12, 18, "start"), "plumb-4b": (-14, 5, "end"), "decider-4b": (-14, 5, "end"), "kev-4b": (-14, -8, "end"),
        "openjev-nli-4b": (14, 18, "start"), "jevk5-9b": (14, 20, "start"), "jev-9b": (-16, 0, "end"), "open-jev-9b": (16, 12, "start"),
        "kev-9b": (16, -2, "start"), "gev-26b": (-14, 5, "end"), "jev-27b": (-16, 4, "end"), "open-jev-27b": (14, 16, "start"),
        "kev-27b": (-16, -6, "end"), "decider-35b-a3b": (12, -8, "start")}
    pos = {k: (xs(MODELS[k][2]) + jitter.get(k, 0), ys(m(k, "acc"))) for k in MODELS}
    wave = lambda p: 0 if p < 1 else (1 if p < 6 else 2)
    starts = (0.6, 6.2, 12.0)
    for fam_keys, col in ((["kev-4b", "kev-9b", "kev-27b"], GOOD), (["decider-2b", "decider-4b", "decider-35b-a3b"], WARN),
                          (["jev-9b", "jev-27b"], ACCENT), (["open-jev-9b", "open-jev-27b"], BAD)):
        for a, b in zip(fam_keys, fam_keys[1:]):
            t = starts[wave(MODELS[b][2])] + 0.6
            s.group(s.anim(t, rise=0), line(*pos[a], *pos[b], col, 1.5, "3 4"))
    for k, (name, fam, p, note) in MODELS.items():
        t = starts[wave(p)] + 0.15 * list(MODELS).index(k) % 1.2
        x, y = pos[k]
        dx, dy, anc = labels[k]
        lab = name + (" (MoE)" if note else "")
        s.group(s.anim(t, rise=0), circle(x, y, 7, FAMILY[fam][1]) + text(x + dx, y + dy, lab, 14, INK, 600, anchor=anc))
    # highlight kev-4b vs kev-27b
    (ax4, ay4), (ax27, ay27) = pos["kev-4b"], pos["kev-27b"]
    s.group(s.anim(17.9, rise=0), circle(ax4, ay4, 14, "none", GOOD, 2.5) + circle(ax27, ay27, 14, "none", GOOD, 2.5)
            + rect(610, 560, 400, 70, "#eef3ea", GOOD, 10, 1.5)
            + text(630, 590, "kev-4b 97.1%  ·  kev-27b 97.4%", 17, GOOD, 700)
            + text(630, 615, "0.3 points for six times the parameters", 15, MUTED))
    s.group(s.anim(0.8), text(X1, Y0 + 84, "MoE: mixture of experts; only 3–4B parameters are active per token.", 13, MUTED, anchor="end", italic=True))
    s.write(out / "size.svg")


# ------------------------------------------------------------------------------------------------- 5 prompts

# runs/prompt-approaches, one seed per cell: accuracy on the 364 Europeana test persons (zero-shot, tuned)
PROMPTS = {  # model: (described zs, in_state zs, described tuned, in_state tuned, bare tuned)
    "kev-4b": (0.879, 0.896, 0.962, 0.970, 0.668), "jev-9b": (0.492, 0.777, 0.967, 0.953, 0.668),
    "open-jev-9b": (0.607, 0.887, 0.970, 0.962, 0.659), "openjev-nli-4b": (0.728, 0.874, 0.953, 0.967, 0.690),
    "plumb-4b": (0.849, 0.821, 0.953, 0.962, 0.626), "decider-2b": (0.739, 0.799, 0.904, 0.956, 0.734),
    "bekko-400m": (0.835, 0.393, 0.953, 0.942, 0.673), "von": (0.516, 0.266, 0.931, 0.904, 0.637),
    "laya": (0.467, 0.610, 0.918, 0.918, 0.615), "gliner-decide": (0.396, 0.247, 0.926, 0.912, 0.646),
}


def prompts(out: Path) -> None:
    s = Svg(24, "Animated slope chart for ten models. Before fine-tuning, moving the candidate descriptions from the "
                "options into the record raises the decoders (JEV-9B 49% to 78%, Open-Jev-9B 61% to 89%) and lowers the "
                "encoders (bekko 84% to 39%, von 52% to 27%, GLiNER 40% to 25%). After fine-tuning both layouts reach "
                "90–97%. With no descriptions at all, every model falls to 62–73% even after fine-tuning.")
    captions(s, [
        (0.2, 5.6, "Before training: each model, with the candidates’ descriptions written next to the options."),
        (5.9, 11.4, "Move the descriptions into the record instead: the decoders gain up to 29 points, the encoders lose up to 44."),
        (11.7, 17.4, "After fine-tuning, the layout hardly matters: both land at 90–97%."),
        (17.7, 23.1, "What does matter is the descriptions: without them, every model falls to 62–73%."),
    ])
    lo, hi = 0.2, 1.0

    def panel(x0, title, idx_a, idx_b, t0):
        X_a, X_b = x0 + 90, x0 + 330
        ys = lambda v: 640 - 470 * (v - lo) / (hi - lo)
        ax = [text(x0, 130, title, 15, INK, 700, spacing=1)]
        for v in (0.2, 0.4, 0.6, 0.8, 1.0):
            ax += [line(x0 + 50, ys(v), x0 + 380, ys(v), RULE, 1), text(x0 + 40, ys(v) + 5, f"{v:.0%}", 14, MUTED, anchor="end")]
        ax += [text(X_a, 672, "descriptions", 15, MUTED, 600, anchor="middle"), text(X_a, 690, "on the options", 15, MUTED, 600, anchor="middle"),
               text(X_b, 672, "descriptions", 15, MUTED, 600, anchor="middle"), text(X_b, 690, "in the record", 15, MUTED, 600, anchor="middle")]
        s.group(s.anim(t0, rise=0), "".join(ax))
        return X_a, X_b, ys

    Xa, Xb, ys = panel(64, "BEFORE FINE-TUNING", 0, 1, 0.3)
    show = {"jev-9b": "JEV-9B", "open-jev-9b": "Open-Jev-9B", "bekko-400m": "bekko-400m", "von": "von", "gliner-decide": "GLiNER", "kev-4b": "kev-4b"}
    enc = {"bekko-400m", "von", "laya", "gliner-decide"}
    for i, (k, (dz, iz, dt, it, bt)) in enumerate(PROMPTS.items()):
        col = BLUE if k in enc else WARN
        s.group(s.anim(0.6 + 0.08 * i, rise=0), circle(Xa, ys(dz), 6, col))
        s.group(s.anim(6.2 + 0.1 * i, rise=0), line(Xa, ys(dz), Xb, ys(iz), col, 2) + circle(Xb, ys(iz), 6, col))
        if k in show:
            dy = {"jev-9b": 10, "open-jev-9b": 8, "kev-4b": -6, "bekko-400m": 4, "von": -9, "gliner-decide": 12}[k]
            s.group(s.anim(6.8, rise=0), text(Xb + 12, ys(iz) + 5 + dy, show[k], 13, INK, 600))
    Xa2, Xb2, ys2 = panel(640, "AFTER FINE-TUNING", 2, 3, 11.9)
    for i, (k, (dz, iz, dt, it, bt)) in enumerate(PROMPTS.items()):
        col = BLUE if k in enc else WARN
        s.group(s.anim(12.2 + 0.08 * i, rise=0), line(Xa2, ys2(dt), Xb2, ys2(it), col, 2) + circle(Xa2, ys2(dt), 6, col) + circle(Xb2, ys2(it), 6, col))
    s.group(s.anim(17.9, rise=0), rect(Xa2 - 30, ys2(0.734) - 8, Xb2 - Xa2 + 60, ys2(0.615) - ys2(0.734) + 16, "#f6e5df", BAD, 8, 1.5)
            + text((Xa2 + Xb2) / 2, ys2(0.615) + 30, "no descriptions at all: 62–73%", 15, BAD, 700, anchor="middle")
            + "".join(circle(Xa2 + 20 + i * (Xb2 - Xa2 - 40) / 9, ys2(v[4]), 6, BAD) for i, v in enumerate(PROMPTS.values())))
    s.group(s.anim(0.5, rise=0), circle(64 + 6, 726, 6, WARN) + text(80, 731, "decoders", 14, MUTED, 600)
            + circle(170, 726, 6, BLUE) + text(184, 731, "encoders", 14, MUTED, 600)
            + text(1136, 731, "One run per cell: differences under ~3 points are noise.", 14, MUTED, anchor="end", italic=True))
    s.write(out / "prompts.svg")


# ---------------------------------------------------------------------------------------------------- 6 cost


GPU = {"NVIDIA RTX 5000 Ada Generation": ("RTX 5000 Ada · 32 GB", BLUE),
       "NVIDIA RTX PRO 6000 Blackwell Server Edition": ("RTX PRO 6000 · 96 GB", WARN),
       "NVIDIA B200": ("B200 · 180 GB", BAD)}


def cost(out: Path) -> None:
    s = Svg(24, "Animated scatter of training time against peak GPU memory for the eighteen fine-tunes, coloured by the "
                "GPU each ran on: RTX 5000 Ada (32 GB), RTX PRO 6000 (96 GB) or B200 (180 GB). kev-4b took 22 minutes "
                "and 12 GB on a B200; Open-Jev-27B 108 minutes and 94 GB. Times and memory depend on the GPU and on our "
                "batch settings, so they are rough guides, not benchmarks.")
    captions(s, [
        (0.2, 5.6, "Each run went to whichever GPU was free on Yale’s cluster. Three kinds were used."),
        (5.9, 11.4, "Training time against the most GPU memory each run needed (mean of three seeds)."),
        (11.7, 17.4, "Read these as rough guides: a B200 is several times faster than an RTX 5000."),
        (17.7, 23.1, "kev-4b: 22 minutes and 12 GB of memory. It would fit on a 32 GB card."),
    ])
    X0, X1, Y0, Y1 = 150, 1100, 640, 190
    xs = lambda g: X0 + (X1 - X0) * g / 100
    ys = lambda t: Y0 - (Y0 - Y1) * t / 120
    ax = [text(X0 - 10, Y1 - 26, "TRAINING TIME (MINUTES) · 2 EPOCHS, 2,780 DECISIONS", 13, MUTED, 700, spacing=1.5)]
    for t in (0, 30, 60, 90, 120):
        ax += [line(X0, ys(t), X1, ys(t), RULE, 1), text(X0 - 12, ys(t) + 5, str(t), 15, MUTED, anchor="end")]
    for g in (0, 20, 40, 60, 80, 100):
        ax += [text(xs(g), Y0 + 26, f"{g} GB", 15, MUTED, anchor="middle")]
    ax.append(text((X0 + X1) / 2, Y0 + 56, "peak GPU memory during training", 16, MUTED, anchor="middle"))
    s.group(s.anim(6.0, rise=0), "".join(ax))
    # legend first
    for i, (g, (lab, col)) in enumerate(GPU.items()):
        s.group(s.anim(0.6 + 0.5 * i), circle(X0 + 10 + i * 300, 118, 8, col) + text(X0 + 26 + i * 300, 124, lab, 17, INK, 700))
    off = {"von": (12, 16), "laya": (12, -6), "decider-2b": (-10, 18, "end"), "gliner-decide": (-12, 16, "end"), "bekko-400m": (12, 5),
           "plumb-4b": (12, 12), "kev-4b": (16, 4), "decider-4b": (0, -14, "middle"), "jev-9b": (12, 5), "jevk5-9b": (12, -6),
           "kev-9b": (12, 14), "open-jev-9b": (-12, -8, "end"), "openjev-nli-4b": (12, 16), "gev-26b": (12, 5), "jev-27b": (12, 16),
           "kev-27b": (12, -6), "open-jev-27b": (-12, 5, "end"), "decider-35b-a3b": (12, 5)}
    for i, k in enumerate(sorted(MODELS, key=lambda k: m(k, "gib"))):
        gname = R[k]["gpu"][0][0]
        col = GPU[gname][1]
        x, y = xs(m(k, "gib")), ys(m(k, "min"))
        o = off[k]
        s.group(s.anim(6.3 + 0.12 * i, rise=0), circle(x, y, 7, col) + text(x + o[0], y + o[1], MODELS[k][0], 13, INK, 600,
                                                                            anchor=o[2] if len(o) > 2 else "start"))
    s.group(s.anim(11.9), rect(620, 196, 470, 118, CARD, RULE, 10)
            + text(640, 226, "Not a speed benchmark", 16, INK, 700)
            + text(640, 252, "The fastest cards ran the largest models, the slowest the", 14, MUTED)
            + text(640, 272, "smallest. Memory also depends on the batch sizes we chose", 14, MUTED)
            + text(640, 294, "and on gradient checkpointing; neither is a minimum.", 14, MUTED))
    x, y = xs(m("kev-4b", "gib")), ys(m("kev-4b", "min"))
    s.group(s.anim(17.9, rise=0), circle(x, y, 15, "none", GOOD, 2.5) + line(xs(32), Y1, xs(32), Y0, GOOD, 1.5, "5 5")
            + text(xs(32) + 8, Y1 + 18, "32 GB card", 14, GOOD, 700))
    s.write(out / "cost.svg")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for f in (readers, leaderboard, false_links, size, prompts, cost):
        f(out)
        print("wrote", f.__name__)
