"""Evaluate and compare iterative Mahjong prelabeler versions on fixed real_val."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a prelabeler version against the previous version.")
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.yaml"), help="Path to paths.yaml.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_iter.yaml"), help="Path to train_iter.yaml.")
    parser.add_argument("--version", type=int, default=None, help="Version to evaluate; defaults to config version.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    print("eval_prelabeler.py scaffold is ready. Implement task 4 before running evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
