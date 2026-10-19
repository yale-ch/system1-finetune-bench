"""Shared machinery for adapters whose model produces one logit per option.

:class:`LogitAdapter` turns ``logits(batch)`` (one 1-D tensor per example, aligned with ``slots(ex)``) into
everything the trainer needs: soft-target cross-entropy (+ optional Brier) loss, probabilities with a
per-type temperature, temperature calibration on dev, a knockout for models that read at most
``max_options`` options per pass, and checkpoints that hold only the trainable tensors.

Noul slots are named ``"true"`` / ``"false"`` (in whatever order the model reads them); choice and score
slots are the option keys.

Also here: helpers for the Qwen3.5-family decoders most System One models are built on (loading text-only
weights out of VLM checkpoints, LoRA, right-padded batches read at chosen positions).
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
from pathlib import Path

from ..schema import Example
from .base import Adapter

log = logging.getLogger(__name__)
NULL_KEYS = {"none", "null", "None"}


class LogitAdapter(Adapter):
    #: noul slot order this model reads (``["false", "true"]`` = index 1 is P(true))
    noul_slots = ("false", "true")

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        t = cfg.get("temperature", 1.0)
        self.temperature = dict(t) if isinstance(t, dict) else {k: float(t) for k in ("choice", "score", "noul")}
        self.max_options = cfg.get("max_options")
        self.brier = float(cfg.get("brier_weight", 0.0))
        self.rng = random.Random(0)

    # ---------------------------------------------------------------------------------- to implement
    def logits(self, batch: list[Example]) -> list:
        """One 1-D float tensor per example, aligned with ``self.slots(ex)``."""
        raise NotImplementedError

    def slots(self, ex: Example) -> list[str]:
        return list(self.noul_slots) if ex.type == "noul" else ex.keys

    # ------------------------------------------------------------------------------------- training
    def target(self, ex: Example):
        import torch

        keys = self.slots(ex)
        if ex.type == "noul":
            y = float(ex.answer)
            return torch.tensor([y if k == "true" else 1 - y for k in keys], dtype=torch.float32)
        t = torch.zeros(len(keys), dtype=torch.float32)
        t[keys.index(ex.answer)] = 1.0
        return t

    def fit_options(self, ex: Example) -> Example:
        """Training: drop random wrong options (never the gold or a null option) down to ``max_options``."""
        if not self.max_options or ex.type == "noul" or len(ex.options) <= self.max_options:
            return ex
        keep = [o for o in ex.options if o.key == ex.answer or o.key in NULL_KEYS]
        rest = [o for o in ex.options if o not in keep]
        chosen = set(o.key for o in self.rng.sample(rest, self.max_options - len(keep))) | {o.key for o in keep}
        return ex.copy(options=[o for o in ex.options if o.key in chosen])

    def loss(self, batch: list[Example]):
        import torch

        batch = [self.fit_options(ex) for ex in batch]
        total = 0.0
        for ex, z in zip(batch, self.logits(batch)):
            t = self.target(ex).to(z.device)
            lp = torch.log_softmax(z.float(), -1)
            total = total - (t * lp).sum()
            if self.brier:
                total = total + self.brier * ((lp.exp() - t) ** 2).sum()
        return total / len(batch)

    # ------------------------------------------------------------------------------------ inference
    def _probs(self, ex: Example, z) -> list[float]:
        import torch

        return torch.softmax(z.float() / self.temperature.get(ex.type, 1.0), -1).tolist()

    def predict(self, batch: list[Example]) -> list:
        out: list = [None] * len(batch)
        simple = [i for i, ex in enumerate(batch) if not self.max_options or ex.type == "noul" or len(ex.options) <= self.max_options]
        if simple:
            for i, z in zip(simple, self.logits([batch[i] for i in simple])):
                out[i] = self._read(batch[i], self._probs(batch[i], z))
        for i, ex in enumerate(batch):
            if out[i] is None:
                out[i] = self.knockout(ex)
        return out

    def _read(self, ex: Example, p: list[float]):
        keys = self.slots(ex)
        if ex.type == "noul":
            return p[keys.index("true")]
        return dict(zip(keys, p))

    def knockout(self, ex: Example) -> dict:
        """More options than one pass reads: score groups, then the group winners against each other.
        P(o) = P(o | its group) * P(group winner wins the final), which sums to 1."""
        cap = self.max_options
        groups = [ex.options[i : i + cap] for i in range(0, len(ex.options), cap)]
        if len(groups) > cap:
            raise ValueError(f"{ex.id}: {len(ex.options)} options exceed a two-round knockout of {cap}")
        rounds = [ex.copy(options=g) for g in groups if len(g) > 1]
        singles = [g[0] for g in groups if len(g) == 1]
        within = {}
        winners = []
        for r, z in (zip(rounds, self.logits(rounds)) if rounds else []):
            p = dict(zip(self.slots(r), self._probs(r, z)))
            within.update({k: (v, r) for k, v in p.items()})
            winners.append(max(p, key=p.get))
        for o in singles:
            within[o.key] = (1.0, None)
            winners.append(o.key)
        final_ex = ex.copy(options=[o for o in ex.options if o.key in winners])
        pf = dict(zip(self.slots(final_ex), self._probs(final_ex, self.logits([final_ex])[0])))
        win_of = {}
        for k, (_, r) in within.items():
            win_of[k] = k if r is None else next(w for w in winners if w in r.keys)
        return {o.key: within[o.key][0] * pf[win_of[o.key]] for o in ex.options}

    # ----------------------------------------------------------------------------------- calibration
    def calibrate(self, dev: list[Example], batch: int = 8) -> dict:
        """Fit one temperature per question type by NLL on dev (golden-section search over log T)."""
        import torch

        self.train_mode(False)
        by: dict[str, list] = {}
        with torch.inference_mode():
            for s in range(0, len(dev), batch):
                chunk = [self.fit_options(ex) for ex in dev[s : s + batch]]
                for ex, z in zip(chunk, self.logits(chunk)):
                    by.setdefault(ex.type, []).append((z.float().cpu(), self.target(ex)))
        for qtype, rows in by.items():
            def nll(log_t):
                t = math.exp(log_t)
                return sum(float(-(y * torch.log_softmax(z / t, -1)).sum()) for z, y in rows) / len(rows)

            a, b = math.log(0.05), math.log(20.0)
            g = (math.sqrt(5) - 1) / 2
            c, d = b - g * (b - a), a + g * (b - a)
            for _ in range(40):
                if nll(c) < nll(d):
                    b, d = d, c
                    c = b - g * (b - a)
                else:
                    a, c = c, d
                    d = a + g * (b - a)
            self.temperature[qtype] = round(math.exp((a + b) / 2), 4)
        log.info(f"calibrated temperatures on {len(dev)} dev examples: {self.temperature}")
        return dict(self.temperature)

    # ----------------------------------------------------------------------------------- checkpoints
    def trainable_modules(self):
        """Modules whose trainable parameters make up a checkpoint (default: the whole model)."""
        return {"model": self.model}

    def save(self, path: Path) -> None:
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tensors = {}
        for prefix, mod in self.trainable_modules().items():
            for n, p in mod.named_parameters():
                if p.requires_grad:
                    tensors[f"{prefix}.{n}"] = p.detach().to("cpu").contiguous()
        save_file(tensors, str(path / "trainable.safetensors"))
        (path / "s1bench.json").write_text(json.dumps({"model": self.cfg, "temperature": self.temperature}, indent=2, default=str))

    def load_checkpoint(self, path: Path) -> None:
        import torch
        from safetensors.torch import load_file

        path = Path(path)
        state = load_file(str(path / "trainable.safetensors"))
        found = 0
        with torch.no_grad():
            for prefix, mod in self.trainable_modules().items():
                params = dict(mod.named_parameters())
                for k, v in state.items():
                    if k.startswith(prefix + "."):
                        params[k[len(prefix) + 1 :]].copy_(v.to(params[k[len(prefix) + 1 :]].device))
                        found += 1
        if found != len(state):
            raise RuntimeError(f"checkpoint {path}: restored {found} of {len(state)} tensors")
        meta = json.loads((path / "s1bench.json").read_text())
        self.temperature = meta.get("temperature", self.temperature)

    def train_mode(self, on: bool) -> None:
        for mod in self.trainable_modules().values():
            mod.train(on)


# =============================================================================================== decoders


def load_text_lm(repo: str, *, revision: str | None = None, dtype=None, device: str = "cuda", subfolder: str | None = None):
    """(backbone, lm_head_weight or None, tokenizer, full_model) for a decoder System One checkpoint.

    * text-only causal LMs (``Qwen3_5ForCausalLM``, ``Qwen3_5MoeForCausalLM``): backbone = ``.model``;
    * vision-language checkpoints (``Qwen3_5ForConditionalGeneration``, ``Gemma4ForConditionalGeneration``): backbone =
      the language model inside (``full`` is returned so a released adapter written against the full model's
      module paths can be attached to it; LoRA layers are injected in place, so the backbone then uses them);
    * bare text models with no LM head (``Qwen3_5TextModel``, e.g. kev-27b): backbone = the model, lm_head None.

    Weights are loaded straight onto ``device`` (27-35B checkpoints would otherwise need as much host RAM).
    ``backbone(input_ids, attention_mask).last_hidden_state`` is the final-norm hidden state.
    """
    import torch
    import transformers as tf

    dtype = dtype or torch.bfloat16
    kw = {"revision": revision, "dtype": dtype, "attn_implementation": "sdpa", "device_map": {"": device}}
    if subfolder:
        kw["subfolder"] = subfolder
    cfg = tf.AutoConfig.from_pretrained(repo, revision=revision, subfolder=subfolder or "")
    tok = tf.AutoTokenizer.from_pretrained(repo, revision=revision, subfolder=subfolder or "")
    arch = (getattr(cfg, "architectures", None) or [""])[0]
    head = None
    if getattr(cfg, "vision_config", None) is not None:
        full = tf.AutoModelForImageTextToText.from_pretrained(repo, **kw)
        backbone = full.model.language_model
        head = full.get_output_embeddings().weight
    elif arch.endswith("TextModel"):
        full = backbone = tf.AutoModel.from_pretrained(repo, **kw)
    else:
        full = tf.AutoModelForCausalLM.from_pretrained(repo, **kw)
        backbone = full.model
        head = full.get_output_embeddings().weight
    if head is not None:
        head = head.detach()
    full.requires_grad_(False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return backbone, head, tok, full


def add_lora(backbone, cfg: dict, modules_to_save: list[str] | None = None):
    """Wrap ``backbone`` in a PEFT LoRA. ``cfg``: {r, alpha, dropout, targets: [module-name suffixes],
    scope: optional substring a module path must contain (e.g. ``language_model`` to skip a vision tower)}.
    Targets missing from this architecture are skipped (Qwen3.5 vs Qwen3.8 vs Gemma names differ)."""
    from peft import LoraConfig, get_peft_model

    wanted = cfg.get("targets", ["q_proj", "k_proj", "v_proj", "o_proj"])
    scope = cfg.get("scope")
    paths = [n for n, _ in backbone.named_modules() if n.rsplit(".", 1)[-1] in wanted and (not scope or scope in n)]
    if not paths:
        raise ValueError(f"none of the LoRA targets {wanted} exist in this model" + (f" under {scope!r}" if scope else ""))
    targets = sorted({p.rsplit(".", 1)[-1] for p in paths})
    target_modules = f".*{re.escape(scope)}.*\\.({'|'.join(targets)})$" if scope else targets
    lc = LoraConfig(r=int(cfg.get("r", 16)), lora_alpha=int(cfg.get("alpha", 32)), lora_dropout=float(cfg.get("dropout", 0.05)),
                    target_modules=target_modules, bias="none", modules_to_save=modules_to_save)
    model = get_peft_model(backbone, lc)
    log.info(f"LoRA r={lc.r} alpha={lc.lora_alpha} on {targets} ({len(paths)} modules){' + ' + str(modules_to_save) if modules_to_save else ''}")
    return model


def enable_checkpointing(model) -> None:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg.use_cache = False


def hidden_at(backbone, rows: list[list[int]], positions: list[list[int]], pad_id: int, device: str):
    """Run right-padded rows through ``backbone`` and return, per row, the hidden states at ``positions``
    (a [len(positions[i]), d] tensor each). Rows are never packed: DeltaNet layers ignore 4D masks."""
    import torch

    n = max(len(r) for r in rows)
    ids = torch.full((len(rows), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), n), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = 1
    out = backbone(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
    h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return [h[i, torch.tensor(p, device=h.device)] for i, p in enumerate(positions)]


def chunked(xs: list, budget_tokens: int, length=len):
    """Split ``xs`` into consecutive chunks whose padded size (rows x longest) stays under ``budget_tokens``."""
    chunk, longest = [], 0
    for x in xs:
        n = length(x)
        if chunk and max(longest, n) * (len(chunk) + 1) > budget_tokens:
            yield chunk
            chunk, longest = [], 0
        chunk.append(x)
        longest = max(longest, n)
    if chunk:
        yield chunk


def render_value(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
