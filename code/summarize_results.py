"""Aggregate continual-learning matrices without inventing missing runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(frame: pd.DataFrame) -> dict[str, float | str | int]:
    final_step = int(frame.learn_step.max())
    final = frame[frame.learn_step == final_step]
    result: dict[str, float | str | int] = {
        "method": str(frame.method.iloc[0]),
        "seed": int(frame.seed.iloc[0]),
        "final_average_ADE_m": float(final.ade.mean()),
        "final_average_FDE_m": float(final.fde.mean()),
        "final_risk_ADE_m": float(final.risk_ade.mean()),
        "final_risk_FDE_m": float(final.risk_fde.mean()),
        "final_worst_FDE_m": float(final.fde.max()),
        "final_worst_risk_FDE_m": float(final.risk_fde.max()),
        "total_train_seconds": float(frame.groupby("learn_step").train_seconds.first().sum()),
    }
    for metric in ("ade", "fde", "risk_ade", "risk_fde"):
        increments = []
        for task_index in range(final_step):
            initial = frame[(frame.learn_step == task_index) & (frame.eval_task_index == task_index)][metric]
            ending = final[final.eval_task_index == task_index][metric]
            if len(initial) == 1 and len(ending) == 1:
                increments.append(float(ending.iloc[0] - initial.iloc[0]))
        result[f"forgetting_{metric}"] = float(np.mean(increments)) if increments else float("nan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    files = sorted(args.input.glob("metrics__*.csv"))
    if not files:
        raise FileNotFoundError(f"No metric files below {args.input}")
    summary = pd.DataFrame([summarize(pd.read_csv(path)) for path in files])
    summary = summary.sort_values(["seed", "final_risk_FDE_m", "final_average_ADE_m"])
    output = args.output or args.input / "summary.csv"
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
