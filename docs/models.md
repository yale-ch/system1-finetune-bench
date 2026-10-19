# The ten model architectures

Each model is a YAML file in `configs/models/` naming a **family** (an adapter class in `s1bench/models/`) plus the
checkpoint, revision and family settings. The prompt layouts below are copied from each model's own code or
model card, so the untouched model is scored as its authors serve it. Revisions are pinned to the Hub commits
of 2026-10-07.

| Config | Hub checkpoint | Family | Backbone | What trains | Prompt / readout | Licence |
|---|---|---|---|---|---|---|
| `gliner-decide` | fastino/GLiNER2.5-Decide | `gliner2` | DeBERTa-v3-large span model | all weights (gliner2 trainer) | `[P] task [L] label: description …` then text; MLP on each `[L]`, softmax | Apache-2.0 |
| `kev-4b` | Qwen/Qwen3.5-4B-Base + jaredpalmer/kev-4b | `pointer` | Qwen3.5-4B-Base | released LoRA r16 + pointer head | `<\|fim_prefix\|>`state `<\|fim_middle\|>`question (`<\|box_start\|>`opt`<\|box_end\|>`)* `<\|fim_suffix\|>`; option-close states · decide state | Apache-2.0 |
| `plumb-4b` | crh225/plumb-4b | `letters` (style `jevk5`) | Qwen3.5-4B (JevK5 v0.2 fine-tune) | new LoRA r16 (attention + DeltaNet) | chat template, system prompt, JSON `{evidence, criterion, options:[{letter, description}]}`; LM-head rows of A..P | Apache-2.0 |
| `decider-2b` | Mapika/decider-2b | `letters` (style `decider`) | Qwen3.5-2B-Base | new LoRA r64 | `Context: … Question: … Options: (A) … Answer: (`; LM-head rows of A..J at the `(` | Apache-2.0 |
| `open-jev-9b` | Qwen/Qwen3.5-9B + ZefanCai/Open-Jev-9B | `candidate` | Qwen3.5-9B | released LoRA r8 + scalar head | one chat prompt per option, "Proposed answer: … Is this proposed answer correct?"; Linear(d,1) on the last token, softmax across options | Apache-2.0 |
| `openjev-nli-4b` | AlexWortega/openjev `qwen3.5-4b-nli-v5` | `nli` | Qwen3.5-4B sequence classifier | new LoRA r16 + `score` head | `Premise: state / Hypothesis: The answer to "q" is label: description` per option; log P(entailment), softmax across options | MIT |
| `jev-9b` | autotrust/JEV-9B | `slots` | Qwen3.5-9B text tower | released LoRA r16 + 24-slot head | `[kind] … [state] … [question] … [options] … [decision]:`; Linear(d,24) on the last token, slot range per type | Apache-2.0 |
| `laya` | convaiinnovations/laya `typed-decisions` | `laya` | ModernBERT-large | all weights | `[CLS] type question: q [SEP] [MASK] opt … [SEP] state [SEP]`; type embedding + 2-layer head, MLP on each `[MASK]` | Apache-2.0 |
| `von` | wfzyx/von | `von` | ModernBERT-large | all weights | `q state [SEP] [MASK] opt …`, each option attends only to the prefix and itself; MLP on each `[MASK]` | Apache-2.0 |
| `bekko-400m` | hotchpotch/bekko-system-one-v0-400m | `bekko` | ModernBERT (ettin-reranker-400m) | all weights | query `Instruction: q / State: …` once, each `Candidate: id: desc` a branch over it; mean-pool + per-type Linear | **none declared** |

Also configured, not in the ten: `jevk5-4b` (alibiserikbay/JevK5, same family as plumb) and `tev1-4b`
(togethercomputer/Tev1-4B-experimental, letters style `tev1`; licence "being finalized").

## Question types per family

All ten answer Choice, Score and Noul natively. Limits the adapters enforce:

- **Option caps**: `letters` jevk5 16, decider 10, tev1 24; `slots` 16. Larger choices are scored by a two-round
  knockout at inference (P(o) = P(o | its group) · P(its group's winner wins the final)); in training, wrong
  options are dropped at random down to the cap (never the gold or a `none` option).
- **Score**: `slots` (JEV-9B) reads 0–5 only: up to 6 levels, the level texts go into the question. Longer scales
  need `approach: {score: as_choice}`. decider's default "isolated levels" score mode is not implemented; scores are
  asked listwise (`0: level`, `1: level`, …).
- **Noul** order differs by model (A = true for jevk5/tev1/von/bekko, `[no, yes]` for kev/decider/laya/Open-Jev); the
  adapters handle it, the data always says `answer: 1.0` for true.

## Known gaps and judgement calls

- **Temperatures.** Zero-shot scoring uses each model's shipped temperature (kev 2.41 from `head.pt`, plumb 2.07,
  decider per type, Open-Jev 1.897, JEV-9B per kind, Laya per type + per-option-count map, von 1.169). After
  fine-tuning, one temperature per question type is refit on the dev split (`train.calibrate`, on by default) —
  the shipped ones describe the old weights.
- **von** is served with an input-dependent temperature and a Noul "band" rule; we use the plain posterior with a
  fitted scalar. Its docs say v1.2 was trained with digit splitting, but the SDK serves without it
  (`digit_split: false` here, as served).
- **Laya** `head_max_len` is raised from 256 to 512 so ~10 described candidates keep their text.
- **openjev-nli-4b** was trained by its author with 3-class NLI cross-entropy per option; here it is trained
  listwise (softmax over options of log P(entailment)), which matches how it is read. Its training mix contains the
  *test* splits of ARC, MMLU, BANKING77 and others, so do not report it on `arc-easy`.
- **bekko-400m** has no licence on the Hub and **tev1-4b**'s is "being finalized": fine for internal experiments,
  not for redistribution of fine-tuned weights.
- **Contamination**: several models list BoolQ, SST-5 and ARC train splits in their training data; those datasets
  test the harness, the Europeana sets are the real held-out comparison.
