"""Which runs of a sweep are finished, running, failed: ``python -m s1bench.status runs/<sweep>``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1])
    try:
        queued = set(subprocess.run(["squeue", "-h", "-u", subprocess.os.environ.get("USER", ""), "-o", "%i"],
                                    capture_output=True, text=True).stdout.split())
    except FileNotFoundError:
        queued = set()
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        job = (d / "job_id").read_text().strip() if (d / "job_id").exists() else ""
        m = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
        if m.get("tuned"):
            state = "done   " + " ".join(f"{ds}:{r['adjusted_mean']:.1f}" for ds, r in m["tuned"].items())
        elif job in queued:
            state = "queued/running"
        else:
            logs = sorted(d.glob("slurm-*.out"))
            tail = logs[-1].read_text().strip().splitlines()[-1:] if logs else ["(no log)"]
            state = "FAILED " + tail[0][:150] if logs else "not submitted"
        print(f"{d.name:<80} {job:>9} {state}")


if __name__ == "__main__":
    main()
