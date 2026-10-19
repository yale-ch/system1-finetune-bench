"""Export a run's best checkpoint in the format its model's own runtime loads.

    python -m s1bench.export runs/ten-models/<run> OUT_DIR

Supported:

* ``pointer`` (kev): ``adapter_config.json`` + ``adapter_model.safetensors`` (PEFT LoRA) + ``head.pt`` (kev's
  checkpoint metadata with the fine-tuned pointer head and the refit temperature), i.e. the same files as
  ``jaredpalmer/kev-4b``, so ``kev`` (and ``s1bench`` with ``adapter:``/``head:`` pointing at the export) loads it.

Other families: the run's ``best/trainable.safetensors`` + ``s1bench.json`` are already self-contained for s1bench.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def export_pointer(run: Path, out: Path) -> dict:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file, save_file

    cfg = json.loads((run / "config.json").read_text())["model"]
    best = json.loads((run / "best" / "s1bench.json").read_text())
    src = cfg["adapter"]
    state = load_file(str(run / "best" / "trainable.safetensors"))
    lora = {k.removeprefix("model.").replace(".default.", "."): v for k, v in state.items() if k.startswith("model.")}
    head = {k.removeprefix("head."): v for k, v in state.items() if k.startswith("head.")}

    orig = hf_hub_download(src["repo"], "adapter_model.safetensors", revision=src.get("revision"))
    want = set(load_file(orig).keys())
    if set(lora) != want:
        raise RuntimeError(f"LoRA keys differ from {src['repo']}: missing {sorted(want - set(lora))[:3]}, extra {sorted(set(lora) - want)[:3]}")

    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.float().contiguous() for k, v in lora.items()}, str(out / "adapter_model.safetensors"))
    shutil.copy(hf_hub_download(src["repo"], "adapter_config.json", revision=src.get("revision")), out / "adapter_config.json")

    hsrc = cfg["head"]
    meta = torch.load(hf_hub_download(hsrc["repo"], hsrc.get("file", "head.pt"), revision=hsrc.get("revision")), map_location="cpu", weights_only=True)
    if set(meta["head"]) != set(head):
        raise RuntimeError(f"head keys {sorted(head)} != {sorted(meta['head'])}")
    meta["head"] = type(meta["head"])((k, head[k].float().contiguous()) for k in meta["head"])
    t = best["temperature"]["choice"]
    meta["temperature"] = float(t)
    meta["init_source"] = {"repo": src["repo"], "revision": src.get("revision")}
    meta["temperature_fit"] = {"method": "NLL on the dev split (choice questions)", "temperature": float(t)}
    run_cfg = json.loads((run / "config.json").read_text())
    meta["args"] = {"trainer": "system1-finetune-bench", "datasets": run_cfg["datasets"], "approach": run_cfg["approach"],
                    **{k: v for k, v in run_cfg["train"].items() if isinstance(v, (int, float, str, bool)) or v is None}}
    torch.save(meta, out / "head.pt")
    for name in ("metrics.json", "config.json"):
        shutil.copy(run / name, out / f"s1bench-{name}")
    return {"lora_tensors": len(lora), "head": sorted(head), "temperature": t}


EXPORTERS = {"pointer": export_pointer}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    args = ap.parse_args()
    run = Path(args.run)
    family = json.loads((run / "config.json").read_text())["model"]["family"]
    if family not in EXPORTERS:
        raise SystemExit(f"no exporter for family {family!r}; best/trainable.safetensors is the checkpoint")
    print(json.dumps(EXPORTERS[family](run, Path(args.out)), indent=2))


if __name__ == "__main__":
    main()
