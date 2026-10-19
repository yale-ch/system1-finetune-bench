# system1-finetune-bench

Fine-tune and compare open **System One decision models**, the models that answer typed questions about a
`state` with probabilities instead of text, on your own datasets, on Yale's Bouchet cluster. There are three
question types:

- **Choice** — pick one option;
- **Score** — rate the state on an ordinal rubric;
- **Noul** — is a statement true?

This repo is for training only: no hosted model API is called anywhere in the workflow. `bouchet/env.sh`
blanks the API keys in every job.

The abstraction has three parts, and each part can change on its own:

| Part | What it is | Where | To add one |
|---|---|---|---|
| **Dataset** | Any source turned into canonical decisions (`state`, `type`, `question`, `options`, `answer`) | `configs/datasets/*.yaml` | A YAML field mapping over a Hub dataset, JSONL, CSV or Parquet file, or a converter function |
| **Prompt approach** | How each question type is put to the model, e.g. option descriptions on the options or in the state; a Score asked as a Choice | `s1bench/approaches.py` | A decorated function |
| **Model** | A checkpoint plus its *family*, meaning how the model reads options and produces probabilities | `configs/models/*.yaml`, `s1bench/models/` | A YAML file if the family exists, otherwise one adapter class |

An **experiment** is a matrix over these parts plus seeds and hyperparameters (`configs/experiments/*.yaml`).
Every run becomes one Slurm job.

## Results so far

Eighteen models fine-tuned on Europeana person → Wikidata reconciliation (3 seeds each, same data and splits) are in
[docs/results.md](docs/results.md) and written up in the blog post
[Eighteen Models, One Decision](https://wjbmattingly.com/blog/eighteen-models-one-decision/). The best value,
kev-4b (97.1% agreement with the labels, 4.1% false links), is published as
[`yale-cultural-heritage/kev-4b-europeana-persons`](https://huggingface.co/yale-cultural-heritage/kev-4b-europeana-persons).

## The ten architectures

GLiNER2.5-Decide, kev-4b, plumb-4b, decider-2b, Open-Jev-9B, openjev-nli-4b, JEV-9B, Laya, von and bekko-400m.
They cover nine adapter families: an encoder label classifier, a pointer head, LM-head letter slots in two prompt
styles, per-candidate yes/no, an NLI cross-encoder, a slot head and three ModernBERT option-marker designs.
[docs/models.md](docs/models.md) has the details, licences and caveats for each model.

## Quick start (from a laptop)

```bash
uv sync                                   # local env (CPU is fine for tests)
uv run pytest -q                          # schema/approach/metric tests + every decoder family on a tiny model
bin/bouchet setup                         # copy the code to Bouchet and build its venv (once, ~5 min)
bin/bouchet push-data europeana-recon-choice boolq sst5   # prepare datasets locally, copy them up
bin/bouchet submit configs/experiments/ten-models.yaml --dry-run
bin/bouchet submit configs/experiments/ten-models.yaml
bin/bouchet status ten-models
bin/bouchet report ten-models             # markdown table, mean ± sd over seeds
bin/bouchet pull ten-models               # metrics, predictions and logs (not weights) into runs/
```

`bin/bouchet` needs `ssh yale-cluster` to work. On Bouchet:

- **Code:** `/nfs/roberts/pi/pi_rs2668/system1-finetune-bench/repo`, on the `pi` fileset. The project fileset is near its quota.
- **Hugging Face cache:** `…/system1-finetune-bench/hf_cache`.
- **Data:** `repo/data/<name>/`. Datasets are prepared on the laptop and copied up, so private Hub datasets need no token on the cluster.
- **Runs:** `repo/runs/<experiment>/<run>/`.

## Datasets

Seven datasets are configured:

| Config | Type | Source | Train / dev / test |
|---|---|---|---|
| `europeana-recon-choice` | choice | private `yale-cultural-heritage/europeana-persons-wikidata-labelled` (`gliner` config) | 2,780 / 312 / 364 |
| `europeana-recon-noul` | noul | the same, one "is this candidate the person?" per candidate | 14,370 / 1,593 / 1,720 |
| `europeana-recon-score` | score | the same, 3-level match score per candidate (derived) | 14,370 / 1,593 / 1,720 |
| `linkedart-recon-choice` | choice | the Linked Art term split, JSONL on Bouchet | — |
| `boolq` | noul | `google/boolq` (train capped at 6,000) | 6,000 / 931 / 3,270 |
| `sst5` | score | `SetFit/sst5` (5 levels) | 8,544 / 1,101 / 2,210 |
| `arc-easy` | choice | `allenai/ai2_arc` ARC-Easy | 2,251 / 570 / 2,376 |

When a source has no dev split, the dev split is carved from train by a hash of each example's `group`. Test is
carved the same way when needed. A group never lands on both sides of a split, and the build fails if one does.

### A new dataset without code

```yaml
# configs/datasets/my-tickets.yaml
name: my-tickets
type: choice
source: {jsonl: {train: /path/train.jsonl, test: /path/test.jsonl}}   # or {hf: org/name}, {csv: ...}, {parquet: ...}
splits: {train: train, test: test}
state: "Subject: {subject}\n\n{body}"
question: "Which team should handle this ticket?"
options:
  - {key: billing, text: "payments, invoices, refunds"}
  - {key: tech, text: "bugs, outages, errors"}
  - {key: none, text: "none of these teams"}
answer: "{team}"
group: "{customer_id}"        # optional: keep a customer's tickets on one side of every split
```

Per-row options come from a list column (`options: {field: candidates, key: "{id}", text: "{label}"}`) or from
parallel columns (`options: {keys: choices.label, texts: choices.text}`). A Noul answer is a float in `[0, 1]`,
and soft labels are allowed. A Score's `options` are its levels from low to high.

For a source that needs code, write `fn(cfg) -> {split: [Example]}` and point to it with
`converter: module:function` (see `s1bench/data/builtin.py:recon`).

## Prompt approaches

Choose one approach per question type in a run or experiment:
`approach: {choice: in_state, score: rubric, noul: statement, shuffle_options: true}`.

| Type | Approach | What the model sees |
|---|---|---|
| choice | `described` (default) | Option keys with their descriptions |
| choice | `in_state` | Descriptions listed in the state; options are bare keys |
| choice | `bare` | Keys only (a control for how much the descriptions carry) |
| choice | `lettered` | Options relabelled A, B, C… with descriptions kept |
| choice | `text_as_key` | The description itself as the option name, with no opaque IDs |
| score | `rubric` (default) | Levels with what each level means |
| score | `numeric` | Bare level numbers |
| score | `as_choice` | The Score asked as a Choice over the levels |
| noul | `statement` (default) | The statement as given |
| noul | `question` | The statement rephrased as a yes/no question |
| noul | `as_choice` | A two-option true/false Choice |

An approach rewrites the decision. The model's family then renders it in the model's native prompt template.
Answers are mapped back, so metrics are always computed on the original question.

## A run

1. **Zero-shot.** The untouched model, with its shipped temperature, is scored on every test split.
2. **Fine-tuning.** AdamW with warmup and a linear, cosine or constant schedule, separate learning rates for the backbone or LoRA and for the head, and gradient clipping.
3. **Model selection.** The best checkpoint is chosen by **dev** loss; test is never used for selection.
4. **Calibration.** One temperature per question type is refit on dev.
5. **Final score.** The tuned model is scored on the test splits.
6. **Preflight.** Before step 2, one forward and backward pass on the longest training examples checks that the micro-batch fits in GPU memory, so an out-of-memory error appears in the first minute rather than hours in.

Run outputs:

- `metrics.json`: zero-shot and tuned results, per dataset and question type.
- `predictions-{zeroshot,tuned}@<dataset>.jsonl`: per-option probabilities for every test example.
- `train-log.jsonl`.
- `best/`: the trainable tensors (LoRA and head, or full weights) plus the temperatures.

Metrics:

| Type | Metrics |
|---|---|
| Choice | accuracy, log loss, ECE; plus linking accuracy, abstention recall and false-link rate when a `none` option exists |
| Score | exact accuracy, within-one accuracy, MAE of the argmax level and of the expected level |
| Noul | balanced accuracy, Brier score, AUROC |

Each type also gets an S1MB-style `adjusted` score: 0 is the best trivial baseline and 100 is perfect.

## Adding a model

**A checkpoint from a known family** needs only a YAML file:

```yaml
# configs/models/kev-9b.yaml
family: pointer
checkpoint: Qwen/Qwen3.5-9B-Base
revision: 68c46c4b3498877f3ef123c856ecfde50c39f404
adapter: {repo: jaredpalmer/kev-9b}
head: {repo: jaredpalmer/kev-9b, file: head.pt}
train: {lr: 2.0e-5, micro_batch: 2, batch_size: 8}
resources: {partition: gpu_rtx6000, gpu: rtx_pro_6000_blackwell, mem: 96G, time: "16:00:00"}
```

**A new family** is a subclass of `LogitAdapter` (`s1bench/models/common.py`) with two methods.
`logits(batch)` returns one 1-D tensor per example. `slots(ex)` gives the order of those logits, which is the
option keys, or `true`/`false` for Noul. Register the class in `FAMILIES` (`s1bench/models/base.py`).

The base class supplies:
- the loss (soft cross-entropy, with optional Brier);
- prediction, the knockout for option caps, and temperature calibration;
- checkpoints of the trainable tensors.

`tests/test_tiny_decoders.py` shows how to test a new family on a tiny random model.
