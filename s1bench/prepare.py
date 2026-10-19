"""Materialise datasets: ``python -m s1bench.prepare europeana-recon-choice [--force]``."""

from __future__ import annotations

import argparse
import json
import logging

from . import data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for name in args.datasets:
        out = data.prepare(name, force=args.force)
        print(name, json.dumps(json.loads((out / "manifest.json").read_text())["splits"]))


if __name__ == "__main__":
    main()
