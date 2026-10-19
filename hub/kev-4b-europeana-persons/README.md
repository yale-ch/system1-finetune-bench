---
license: apache-2.0
base_model:
- jaredpalmer/kev-4b
- Qwen/Qwen3.5-4B-Base
library_name: peft
pipeline_tag: text-classification
tags:
- system-one
- decision-model
- entity-linking
- reconciliation
- wikidata
- europeana
- cultural-heritage
- kev
datasets:
- yale-cultural-heritage/europeana-persons-wikidata-labelled
---

# kev-4b · Europeana persons → Wikidata

[`jaredpalmer/kev-4b`](https://huggingface.co/jaredpalmer/kev-4b) fine-tuned to decide which Wikidata person, if
any, a person named in a Europeana cultural-heritage record is (Yale Cultural Heritage AI). Given the record text
and up to ten Wikidata candidates, it picks one candidate or `none`.

It is the best of ten System One architectures we fine-tuned on this task with
system1-finetune-bench (Yale Cultural Heritage AI) (GLiNER2.5-Decide, kev-4b, plumb-4b,
decider-2b, Open-Jev-9B, openjev-nli-4b, JEV-9B, Laya, von, bekko-400m; three seeds each).

## Files

The same layout as `jaredpalmer/kev-4b`, so kev's own code loads it:

| File | |
|---|---|
| `adapter_config.json`, `adapter_model.safetensors` | PEFT LoRA r16 / α32 on `Qwen/Qwen3.5-4B-Base` @ `1001bb4d…` (attention, MLP and DeltaNet projections) |
| `head.pt` | kev checkpoint metadata: the fine-tuned pointer head and the refit temperature (T = 1.372) |
| `s1bench-config.json`, `s1bench-metrics.json` | the training run's configuration and its full zero-shot / tuned results |

## Results

Held-out test split of [`europeana-persons-wikidata-labelled`](https://huggingface.co/datasets/yale-cultural-heritage/europeana-persons-wikidata-labelled)
(364 persons with candidates; 90 have no correct candidate). The reference labels were produced with the help of a
larger language model and **were not checked by people**, so these figures measure agreement, not accuracy.

| | Agreement | Links matching the labels | Correct `none` | False links (labels say none) |
|---|---|---|---|---|
| kev-4b, untouched | 88.2% | 89.4% | 84.4% | 15.6% |
| **this model** (seed 42) | **97.3%** | **97.4%** | **96.7%** | **3.3%** |
| same recipe, mean ± sd of 3 seeds | 97.1% ± 0.3 | | | 4.1% ± 2.3 |

The released seed was chosen by loss on the development split, not by test score. Expected calibration error on the
test split is 0.014 with the refit temperature.

For comparison, the other architectures' 3-seed means on the same split ranged from 91.3% (GLiNER2.5-Decide, 20.7%
false links) to 96.1% (JEV-9B, 5.6% false links).

## Use

Serve it with the [kev](https://github.com/jaredpalmer/kev) repository, exactly as kev-4b:

```bash
git clone https://github.com/jaredpalmer/kev.git && cd kev && uv sync --extra serve
huggingface-cli login            # the repository is private
uv run --extra serve python -m kev.serve --run yale-cultural-heritage/kev-4b-europeana-persons --port 8008
```

Ask it the way it was trained: the state is the record text, the Choice criteria are the candidates
(`wikidata:Q…` → a one-line summary) plus `none`:

```python
from typesafe_sdk import Choice, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")
state = """Name as recorded: Zamoyski, Maurycy (1871-1939). (Wyd.)
Record 1: Goniec Wieczorny. 1916, no 488
  Type: text; periodical
  This person's role: creator
  Date: 1916
  Collection: Warsaw University Library"""
r = client.system_one(state=state, questions={"entity": Choice(
    instructions="Which record is the person “Zamoyski, Maurycy” referred to in the text?",
    criteria={
        "wikidata:Q1911719": "Maurycy Klemens Zamoyski — Polish noble/diplomat [1871-1939] [human · politician; diplomat · 1871–1939]",
        "none": "None of the listed records is the entity referred to in the text",
    })})
print(r.choices["entity"].choice, r.choices["entity"].probabilities)
```

Or with system1-finetune-bench: point a `pointer`-family model config's `adapter:` and `head:` at this repository
(see `configs/models/kev-4b.yaml`).

## Training

- **Start**: `jaredpalmer/kev-4b` (its LoRA adapter and pointer head), base `Qwen/Qwen3.5-4B-Base` @ `1001bb4d826a52d1f399e183466143f4da7b741b`.
- **Data**: the `gliner` config of `europeana-persons-wikidata-labelled`, as Choice decisions: 2,780 training
  decisions (10% of the training split, by person, held out as development: 312), candidate order shuffled with
  `none` last.
- **Recipe**: 2 epochs, AdamW, lr 2e-5 (LoRA and head), linear schedule with 5% warmup, effective batch 8 (2 × 4),
  bf16 weights, gradient checkpointing; cross-entropy over the options. Best checkpoint by development loss (step 300
  of 696). One temperature refit on the development split by NLL.
- **Compute**: one GPU, 25 minutes, 12 GiB peak memory.

## Limits

- Trained only on person reconciliation against Wikidata in Europeana records; other entity types and catalogues
  are out of distribution.
- It chooses only among the candidates it is shown; a person the candidate search misses is answered `none` or,
  less often, linked to a namesake.
- The reference labels were not reviewed by people.
- kev-4b's general-purpose abilities (other Choice, Score and Noul tasks) were not re-measured after fine-tuning.

## Licence

Apache-2.0, as kev-4b and Qwen3.5-4B-Base. The training data (Europeana metadata, Wikidata) is CC0.
