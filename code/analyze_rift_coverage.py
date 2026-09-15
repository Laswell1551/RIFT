"""Relate observable-state memory coverage to risk-stratified forgetting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AGGREGATE = RESULTS / "rift-robustness-aggregate"
SEEDS = (7, 11, 23, 47)
METHODS = ("gss_adapted", "h2c_adapted", "reservoir")
COVERAGE = ("weighted_cover_mean", "high_risk_cover_max")
OUTCOMES = ("final_risk_FDE_m", "forgetting_risk_fde")


def run_directory(kind: str, value: str | int, seed: int) -> Path:
    if kind == "budget":
        if int(value) == 500:
            return (
                RESULTS / "cgsm-3d-cpa-validation-seed7"
                if seed == 7
                else RESULTS / f"cgsm-3d-cpa-validation-frozen-seed{seed}"
            )
        return RESULTS / f"rift-budget-{value}-seed{seed}"
    if value == "primary":
        return run_directory("budget", 500, seed)
    return RESULTS / f"rift-order-{value}-seed{seed}"


def collect(kind: str, values: tuple[str | int, ...], summary_name: str) -> pd.DataFrame:
    summaries = pd.read_csv(AGGREGATE / summary_name)
    rows = []
    condition_column = "memory_budget" if kind == "budget" else "task_order"
    for value in values:
        for seed in SEEDS:
            directory = run_directory(kind, value, seed)
            for method in METHODS:
                metric_path = directory / f"metrics__{method}__seed{seed}.csv"
                metrics = pd.read_csv(metric_path)
                final_step = int(metrics.learn_step.max())
                final_row = metrics[metrics.learn_step == final_step].iloc[0]
                match = summaries[
                    (summaries[condition_column].astype(str) == str(value))
                    & (summaries.seed == seed)
                    & (summaries.method == method)
                ]
                if len(match) != 1:
                    raise ValueError(f"Missing or duplicated summary for {value}, {seed}, {method}")
                summary = match.iloc[0]
                rows.append(
                    {
                        "analysis": kind,
                        "condition": value,
                        "seed": seed,
                        "method": method,
                        **{column: float(final_row[column]) for column in COVERAGE},
                        **{column: float(summary[column]) for column in OUTCOMES},
                    }
                )
    return pd.DataFrame(rows)


def correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for analysis, block in frame.groupby("analysis"):
        for adjustment in ("raw", "condition_seed_residual"):
            working = block.copy()
            if adjustment == "condition_seed_residual":
                columns = list(COVERAGE + OUTCOMES)
                working[columns] = working[columns] - working.groupby(
                    ["condition", "seed"]
                )[columns].transform("mean")
            for coverage in COVERAGE:
                for outcome in OUTCOMES:
                    rows.append(
                        {
                            "analysis": analysis,
                            "adjustment": adjustment,
                            "coverage_metric": coverage,
                            "outcome": outcome,
                            "n": len(working),
                            "pearson_r": float(working[coverage].corr(working[outcome], method="pearson")),
                            "spearman_rho": float(working[coverage].corr(working[outcome], method="spearman")),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    budget = collect("budget", (100, 250, 500, 1000), "budget_per_seed.csv")
    order = collect("order", ("primary", "reverse", "alternating"), "order_per_seed.csv")
    observations = pd.concat((budget, order), ignore_index=True)
    observations.to_csv(AGGREGATE / "coverage_observations.csv", index=False)
    result = correlations(observations)
    result.to_csv(AGGREGATE / "coverage_correlations.csv", index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
