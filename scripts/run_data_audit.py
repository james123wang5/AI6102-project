#!/usr/bin/env python3
"""Run the stage-01 streaming audit for the downloaded Optiver CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai6102.data_audit import audit_train_csv, write_audit_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "optiver-trading-at-the-close" / "train.csv")
    parser.add_argument("--example-test-path", type=Path, default=PROJECT_ROOT / "optiver-trading-at-the-close" / "example_test_files" / "test.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "01_data_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {args.train_path}")
    summary = audit_train_csv(args.train_path, args.example_test_path)
    json_path, markdown_path = write_audit_outputs(summary, args.output_dir)
    print(f"Audited {summary['dataset']['row_count']:,} rows.")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
