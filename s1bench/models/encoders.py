"""Bidirectional-encoder System One models (all ModernBERT-based, fully fine-tuned).

``laya``  -- convaiinnovations/laya(-typed-decisions): ``[CLS] "<type> question: q" [SEP] [MASK] opt0 [MASK] opt1 .. [SEP]
             state [SEP]``; a type embedding + 2-layer transformer head + MLP scorer on each [MASK]. Uses the repo's
             own ``rl_common.py`` (downloaded at load time) for the model and sequence layout.
``von``   -- wfzyx/von: ``"q state [SEP] [MASK] opt0 [MASK] opt1 .."`` with each option attending only to the prefix
             and itself (order-invariant); MLP scorer on each [MASK]. Vendored ``option_marker.py``.
``bekko`` -- hotchpotch/bekko-system-one-v0-*: query ``Instruction: q / State: <state as JSON>`` encoded once, each
             candidate ``Candidate: id: desc`` as its own branch over the shared prefix; mean-pool + per-type
             Linear head. Uses the repo's ``inference_v0.py`` (remote code, pinned revision).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

from ..schema import Example
from .common import LogitAdapter, render_value

log = logging.getLogger(__name__)


def import_file(path: str | Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def detach_inference_tensors(module) -> None:
    """Weights created under ``torch.inference_mode`` cannot take part in autograd; re-create them."""
    import torch

    with torch.no_grad():
        for p in module.parameters():
            p.data = p.data.clone()
        for name, b in list(module.named_buffers()):
            owner = module.get_submodule(name.rsplit(".", 1)[0]) if "." in name else module
            setattr(owner, name.rsplit(".", 1)[-1], b.clone())


class EncoderAdapter(LogitAdapter):
    def autocast(self):
        import torch

        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda") and self.cfg.get("bf16", True))

    def param_groups(self, train_cfg):
        lr = float(train_cfg.get("lr", 2e-5))
        head_lr = float(train_cfg.get("head_lr") or self.cfg.get("head_lr") or lr * 5)
        enc, head = [], []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                (enc if n.startswith(self.encoder_prefix) else head).append(p)
        return [{"params": enc, "lr": lr}, {"params": head, "lr": head_lr}]

    encoder_prefix = "encoder."


# ======================================================================================================== laya


class LayaAdapter(EncoderAdapter):
    noul_slots = ("false", "true")
    QTYPES = {"choice": 0, "score": 1, "noul": 2}

    def load(self) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        c = self.cfg
        sub = c.get("subfolder", "")
        root = Path(snapshot_download(c["checkpoint"], revision=c.get("revision"),
                                      allow_patterns=["rl_common.py", f"{sub}/*" if sub else "*"]))
        self.lib = import_file(root / "rl_common.py", "laya_rl_common")
        d = root / sub if sub else root
        self.acfg = json.loads((d / "rl_agent_config.json").read_text())
        self.tok = AutoTokenizer.from_pretrained(str(d / "tokenizer"))
        model = self.lib.build_model(self.acfg, encoder_dir=str(d / "encoder"))
        model.load_state_dict(load_file(str(d / "model.safetensors")), strict=True)
        model = model.float().to(self.device)
        model.requires_grad_(True)
        model.act_head.requires_grad_(False)  # "carries no usable signal yet"; not part of the decision
        if c.get("gradient_checkpointing", True):
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.head_checkpointing = True
        self.model = model
        self.max_len = int(c.get("max_len", self.acfg["max_len"]))
        self.head_max_len = int(c.get("head_max_len", self.acfg["head_max_len"]))
        if not c.get("temperature"):
            t = self.acfg.get("temperature", [1.0, 1.0, 1.0])
            self.temperature = {"choice": t[0], "score": t[1], "noul": t[2]}
            self.bucket_t = dict(self.acfg.get("temperature_by_options") or {}) if c.get("bucket_temperatures", True) else {}
        else:
            self.bucket_t = {}

    def question(self, ex: Example) -> dict:
        if ex.type == "choice":
            return {"t": "choice", "ins": ex.question, "crit": {o.key: o.text for o in ex.options}}
        if ex.type == "score":
            return {"t": "score", "ins": ex.question, "crit": [o.text or o.key for o in ex.options]}
        return {"t": "noul", "ins": ex.question, "crit": {}}

    def logits(self, batch):
        import torch

        items = []
        for ex in batch:
            ids, markers = self.lib.build_sequence(self.tok, render_value(ex.state), self.question(ex), self.max_len, self.head_max_len)
            if len(markers) != len(self.slots(ex)):
                raise ValueError(f"{ex.id}: {len(markers)} of {len(self.slots(ex))} option markers fit in head_max_len {self.head_max_len}")
            items.append((ids, markers, self.QTYPES[ex.type]))
        n, k = max(len(i) for i, _, _ in items), max(len(m) for _, m, _ in items)
        ids = torch.full((len(items), n), self.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(items), n), dtype=torch.long)
        pos = torch.full((len(items), k), -1, dtype=torch.long)
        for b, (i, m, _) in enumerate(items):
            ids[b, : len(i)] = torch.tensor(i)
            att[b, : len(i)] = 1
            pos[b, : len(m)] = torch.tensor(m)
        dev = self.device
        with self.autocast():
            z, _ = self.model(ids.to(dev), att.to(dev), pos.to(dev), (pos >= 0).to(dev),
                              torch.tensor([q for _, _, q in items], device=dev))
        return [z[b, : len(m)].float() for b, (_, m, _) in enumerate(items)]

    def _probs(self, ex, z):
        import torch

        k = len(z)
        bucket = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
        t = self.bucket_t.get(f"{ex.type}:{bucket}", self.temperature.get(ex.type, 1.0))
        return torch.softmax(z.float() / t, -1).tolist()

    def calibrate(self, dev, batch=8):
        self.bucket_t = {}  # the shipped per-option-count map would override the refit
        return super().calibrate(dev, batch)


# ========================================================================================================= von


class VonAdapter(EncoderAdapter):
    noul_slots = ("true", "false")

    def load(self) -> None:
        import torch
        from huggingface_hub import hf_hub_download, snapshot_download

        from .vendor.von_option_marker import OptionMarkerModel

        c = self.cfg
        root = snapshot_download(c["checkpoint"], revision=c.get("revision"),
                                 allow_patterns=["config.json", "model.safetensors", "tokenizer*", "calibration.json", "marker_calibration.json"])
        model = OptionMarkerModel(base_model_id=root, digit_split=bool(c.get("digit_split", False)))
        sd = torch.load(hf_hub_download(c["checkpoint"], "option_marker.pt", revision=c.get("revision")), map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
        del sd
        self.independent = bool(c.get("independent_options", True))
        if not c.get("temperature"):
            t = json.loads((Path(root) / "calibration.json").read_text())
            t = float(t.get("temperature", 1.0)) if isinstance(t, dict) else float(t)
            self.temperature = {k: t for k in ("choice", "score", "noul")}
        model.requires_grad_(True)
        if c.get("gradient_checkpointing", True):
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model = model.float().to(self.device)
        self.tok = model.tokenizer
        self.max_len = int(c.get("max_len", 4096))
        self.max_state = int(c.get("max_state_tokens", 2048))

    def options(self, ex) -> list[str]:
        if ex.type == "noul":
            return ["Yes, condition holds true.", "No, condition is false."]
        return [o.text or o.key for o in ex.options]

    def logits(self, batch):
        texts = []
        for ex in batch:
            ids = self.tok.encode(render_value(ex.state), add_special_tokens=False)
            state = render_value(ex.state) if len(ids) <= self.max_state else self.tok.decode(ids[: self.max_state])
            texts.append(self.model.pack_sequence(state, ex.question, self.options(ex)))
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len).to(self.device)
        pos = [(enc.input_ids[b] == self.model.mask_token_id).nonzero().flatten().tolist() for b in range(len(batch))]
        for ex, p in zip(batch, pos):
            if len(p) != len(self.slots(ex)):
                raise ValueError(f"{ex.id}: {len(p)} of {len(self.slots(ex))} option markers survived truncation")
        with self.autocast():
            zs = self.model(enc.input_ids, enc.attention_mask, pos, independent_options=self.independent)
        return [z.float() for z in zs]


# ======================================================================================================= bekko


class BekkoAdapter(EncoderAdapter):
    noul_slots = ("true", "false")
    encoder_prefix = "encoder."

    def load(self) -> None:
        from huggingface_hub import hf_hub_download

        c = self.cfg
        code = hf_hub_download(c["checkpoint"], "inference_v0.py", revision=c.get("revision"))
        self.lib = import_file(code, "bekko_inference_v0")
        mod = self.lib.BekkoInference.load(c["checkpoint"], subfolder="0_BekkoInference", revision=c.get("revision"))
        detach_inference_tensors(mod)
        mod.requires_grad_(True)
        self.model = mod.to(self.device)
        self.layout = c.get("prefix_layout", "instruction_state")
        self.qlen, self.dlen = int(c.get("query_length", 4096)), int(c.get("document_length", 512))

    def native(self, ex) -> dict:
        if ex.type == "noul":
            crit = [{"id": "true", "description_json": json.dumps("The statement is true."), "value": None},
                    {"id": "false", "description_json": json.dumps("The statement is false."), "value": None}]
        else:
            crit = [{"id": o.key, "description_json": json.dumps(o.text or o.key), "value": float(i) if ex.type == "score" else None}
                    for i, o in enumerate(ex.options)]
        return {"state_json": json.dumps(render_value(ex.state), ensure_ascii=False),
                "decisions": [{"id": "q", "kind": "judgment", "type": ex.type, "instructions_json": json.dumps(ex.question),
                               "system_prompt": "", "criteria": crit, "documents": [], "scoring": None}]}

    def logits(self, batch):
        groups = [g for ex in batch for g in self.lib.input_groups(self.native(ex), prefix_layout=self.layout)]
        prepared = self.model.prepare_groups(groups, query_length=self.qlen, document_length=self.dlen)
        feats = self.lib.to_device(self.lib.collate_groups(prepared, self.model), self.device)
        scores = self.model(feats)["scores"].flatten()
        out, i = [], 0
        for g in prepared:
            out.append(scores[i : i + len(g.documents)].float())
            i += len(g.documents)
        return out
