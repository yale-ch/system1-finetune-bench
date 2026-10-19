"""Submit every run of an experiment file to Slurm (run this on Bouchet; ``bin/bouchet submit`` does it from a laptop).

    python -m s1bench.submit configs/experiments/ten-models.yaml            # submit runs not yet finished
    python -m s1bench.submit configs/experiments/ten-models.yaml --dry-run  # write the job files, print the plan
    python -m s1bench.submit ... --only kev --only plumb                    # runs whose name contains any of these
    python -m s1bench.submit ... --rerun                                    # also resubmit finished runs
    python -m s1bench.submit ... --slurm partition=scavenge_gpu             # preemptible, often starts at once

Resources come from the model config's ``resources`` block, overridden by the experiment's ``slurm`` block
and then by the run's own ``slurm`` block::

    resources: {partition: gpu_rtx6000, gpu: rtx_pro_6000_blackwell, mem: 96G, cpus: 8, time: "08:00:00"}

One job per run; its script and log live in the run directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .models.base import load_config as model_config
from .sweep import ROOT, expand, merge

DEFAULT_RES = {"partition": "gpu_rtx6000", "gpu": "rtx_pro_6000_blackwell", "gpus": 1, "cpus": 8, "mem": "64G",
               "time": "06:00:00", "account": "pi_rs2668", "env": ".venv"}

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=s1-{short}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --gpus={gpu}:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={out}/slurm-%j.out
#SBATCH --error={out}/slurm-%j.out
{extra}
set -euo pipefail
cd {root}
source bouchet/env.sh
echo "job $SLURM_JOB_ID on $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
{env}/bin/python -m s1bench.train {sweep_file} --run {name}
echo "finished $(date -Is)"
"""


def resources(rc: dict) -> dict:
    mcfg = model_config(rc["model"])
    return merge(merge(DEFAULT_RES, mcfg.get("resources") or {}), rc.get("slurm") or {})


def done(rc: dict) -> bool:
    p = Path(rc["out"]) / "metrics.json"
    return p.exists() and json.loads(p.read_text()).get("tuned") is not None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_file")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--slurm", action="append", default=[], metavar="KEY=VALUE",
                    help="override a resource for this submission, e.g. --slurm partition=scavenge_gpu --slurm time=02:00:00")
    args = ap.parse_args()
    sweep_file = Path(args.sweep_file).resolve()
    runs = expand(sweep_file)
    if args.only:
        runs = [r for r in runs if any(o in r["name"] for o in args.only)]
    queued = set(subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"], capture_output=True, text=True).stdout.split())
    plan = []
    for rc in runs:
        if done(rc) and not args.rerun:
            plan.append((rc["name"], "done", None))
            continue
        jf = Path(rc["out"]) / "job_id"
        if jf.exists() and jf.read_text().strip() in queued:
            plan.append((rc["name"], f"already queued/running ({jf.read_text().strip()})", None))
            continue
        res = resources(rc)
        res.update(dict(kv.split("=", 1) for kv in args.slurm))
        out = Path(rc["out"])
        out.mkdir(parents=True, exist_ok=True)
        extra = "\n".join(f"#SBATCH {x}" for x in res.get("sbatch_extra", []))
        script = TEMPLATE.format(short=rc["name"][:60], out=out, root=ROOT, sweep_file=sweep_file, name=rc["name"], extra=extra,
                                 **{k: res[k] for k in ("partition", "account", "gpu", "gpus", "cpus", "mem", "time", "env")})
        path = out / "job.sbatch"
        path.write_text(script)
        if args.dry_run:
            plan.append((rc["name"], f"{res['partition']}/{res['gpu']} {res['mem']} {res['time']}", None))
            continue
        r = subprocess.run(["sbatch", "--parsable", str(path)], capture_output=True, text=True)
        if r.returncode:
            plan.append((rc["name"], "SBATCH FAILED: " + r.stderr.strip().splitlines()[-1], None))
            continue
        job = r.stdout.strip()
        (out / "job_id").write_text(job + "\n")
        plan.append((rc["name"], f"{res['partition']}/{res['gpu']}", job))
    w = max((len(n) for n, _, _ in plan), default=10)
    for name, what, job in plan:
        print(f"{name:<{w}}  {what}  {job or ''}")
    print(f"{len(plan)} runs; {sum(1 for _, _, j in plan if j)} submitted")


if __name__ == "__main__":
    main()
