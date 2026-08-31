#!/usr/bin/env python3
"""Run stage-02 chronological baselines without touching the final holdout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai6102.baselines import (  # noqa: E402
    folds_from_config,
    load_config,
    load_raw_training_data,
    run_fold,
    write_baseline_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "optiver-trading-at-the-close" / "train.csv")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "02_baselines")
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "02_baselines")
    parser.add_argument(
        "--fold",
        action="append",
        dest="requested_folds",
        help="Run one named development fold. Repeat the argument to run several folds; omit it to run all folds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    frame, invalid_target_rows = load_raw_training_data(args.train_path)
    print(f"Loaded {len(frame):,} usable rows; filtered {invalid_target_rows:,} non-finite targets.", flush=True)
    results = []
    predictions = {}
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    all_folds = folds_from_config(config)
    selected_folds = [fold for fold in all_folds if not args.requested_folds or fold.name in args.requested_folds]
    unknown_folds = set(args.requested_folds or []) - {fold.name for fold in all_folds}
    if unknown_folds:
        raise ValueError(f"Unknown fold name(s): {', '.join(sorted(unknown_folds))}")
    for fold in selected_folds:
        print(f"Running {fold.name}: train {fold.train_start}-{fold.train_end}, validate {fold.validation_start}-{fold.validation_end}", flush=True)
        result, prediction_frame, model = run_fold(frame, fold, config)
        results.append(result)
        predictions[fold.name] = prediction_frame
        model.booster_.save_model(str(args.artifact_dir / f"raw_lightgbm_{fold.name}.txt"))
        write_baseline_outputs(results, predictions, args.output_dir, config, invalid_target_rows)
        print(f"{fold.name} MAE — Zero: {result.zero_mae:.6f}; Median: {result.median_mae:.6f}; Raw LightGBM: {result.raw_lightgbm_mae:.6f}", flush=True)
    print(f"Reports written to: {args.output_dir}")


if __name__ == "__main__":
    main()
