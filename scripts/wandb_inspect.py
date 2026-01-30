#!/usr/bin/env python3
"""WandB metrics inspection utilities for analyzing training runs."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import wandb
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install wandb pandas numpy")
    sys.exit(1)


@dataclass
class MetricStats:
    """Statistics for a single metric."""
    name: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    first: float
    last: float
    trend: str  # "increasing", "decreasing", "stable"

    def __str__(self):
        return (f"{self.name}: {self.last:.4f} (mean={self.mean:.4f}, "
                f"min={self.min:.4f}, max={self.max:.4f}, trend={self.trend})")


def get_api():
    """Get WandB API client."""
    return wandb.Api()


def list_runs(project: str, entity: Optional[str] = None, limit: int = 10) -> List[dict]:
    """List recent runs in a project.

    Args:
        project: WandB project name
        entity: WandB entity (username or team)
        limit: Maximum number of runs to return

    Returns:
        List of run info dicts
    """
    api = get_api()
    path = f"{entity}/{project}" if entity else project

    runs = api.runs(path, order="-created_at")

    result = []
    for i, run in enumerate(runs):
        if i >= limit:
            break
        result.append({
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "created": run.created_at,
            "runtime": run.summary.get("_runtime", 0),
            "step": run.summary.get("_step", 0),
        })

    return result


def get_run(run_id: str, project: str, entity: Optional[str] = None):
    """Get a specific run.

    Args:
        run_id: Run ID
        project: WandB project name
        entity: WandB entity

    Returns:
        WandB Run object
    """
    api = get_api()
    path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    return api.run(path)


def get_metrics_history(
    run_id: str,
    project: str,
    entity: Optional[str] = None,
    keys: Optional[List[str]] = None,
    samples: int = 10000,
) -> pd.DataFrame:
    """Get metrics history for a run.

    Args:
        run_id: Run ID
        project: WandB project name
        entity: WandB entity
        keys: Specific metric keys to fetch (None = all)
        samples: Maximum number of samples

    Returns:
        DataFrame with metrics history
    """
    run = get_run(run_id, project, entity)

    # Fetch history
    if keys:
        history = run.history(keys=keys + ["_step"], samples=samples)
    else:
        history = run.history(samples=samples)

    return history


def compute_metric_stats(df: pd.DataFrame, metric: str) -> Optional[MetricStats]:
    """Compute statistics for a metric.

    Args:
        df: DataFrame with metrics
        metric: Metric name

    Returns:
        MetricStats or None if metric not found
    """
    if metric not in df.columns:
        return None

    series = df[metric].dropna()
    if len(series) == 0:
        return None

    # Compute trend (linear regression slope)
    if len(series) > 10:
        x = np.arange(len(series))
        slope = np.polyfit(x, series.values, 1)[0]
        rel_slope = slope / (series.std() + 1e-8)

        if rel_slope > 0.1:
            trend = "increasing"
        elif rel_slope < -0.1:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"

    return MetricStats(
        name=metric,
        count=len(series),
        mean=series.mean(),
        std=series.std(),
        min=series.min(),
        max=series.max(),
        first=series.iloc[0],
        last=series.iloc[-1],
        trend=trend,
    )


def analyze_training_health(df: pd.DataFrame) -> Dict[str, any]:
    """Analyze training health from metrics.

    Args:
        df: DataFrame with metrics

    Returns:
        Dict with health analysis
    """
    analysis = {
        "warnings": [],
        "info": [],
        "metrics": {},
    }

    # Key metrics to analyze
    key_metrics = [
        # Losses
        "train/policy_loss",
        "train/value_loss",
        "train/entropy_loss",
        # PPO diagnostics
        "train/clip_fraction",
        "train/approx_kl",
        "train/explained_variance",
        "train/learning_rate",
        # Rewards
        "env/mean_reward",
        "env/mean_episode_reward",
        # Goals
        "env/goals_scored",
        "env/goals_conceded",
        "env/goal_diff",
        # Ball interaction
        "env/ball_touches",
        "env/mean_touch_velocity",
        # Performance
        "time/steps_per_second",
    ]

    for metric in key_metrics:
        stats = compute_metric_stats(df, metric)
        if stats:
            analysis["metrics"][metric] = stats

    # Health checks
    m = analysis["metrics"]

    # Check clip fraction (should be 0.1-0.3 ideally)
    if "train/clip_fraction" in m:
        cf = m["train/clip_fraction"]
        if cf.last > 0.5:
            analysis["warnings"].append(
                f"High clip fraction ({cf.last:.2f}) - policy changing too fast, consider lower LR"
            )
        elif cf.last < 0.05:
            analysis["warnings"].append(
                f"Low clip fraction ({cf.last:.2f}) - policy barely changing, consider higher LR"
            )
        else:
            analysis["info"].append(f"Clip fraction healthy ({cf.last:.2f})")

    # Check KL divergence (should be < 0.1 typically)
    if "train/approx_kl" in m:
        kl = m["train/approx_kl"]
        if kl.last > 0.1:
            analysis["warnings"].append(
                f"High KL divergence ({kl.last:.3f}) - policy updates too aggressive"
            )
        elif kl.max > 0.5:
            analysis["warnings"].append(
                f"KL spike detected (max={kl.max:.3f}) - may cause instability"
            )

    # Check explained variance (should be > 0.5, ideally > 0.9)
    if "train/explained_variance" in m:
        ev = m["train/explained_variance"]
        if ev.last < 0.5:
            analysis["warnings"].append(
                f"Low explained variance ({ev.last:.2f}) - value function not fitting well"
            )
        elif ev.last > 0.95 and ev.first > 0.9:
            analysis["info"].append(
                f"Very high explained variance ({ev.last:.2f}) from start - may indicate overfitting or easy task"
            )
        else:
            analysis["info"].append(f"Explained variance healthy ({ev.last:.2f})")

    # Check entropy (should decrease but not collapse)
    if "train/entropy_loss" in m:
        ent = m["train/entropy_loss"]
        # Note: entropy_loss is typically negative (it's -entropy * coef)
        if abs(ent.last) < 0.01:
            analysis["warnings"].append(
                f"Very low entropy ({ent.last:.4f}) - policy may have collapsed"
            )
        elif ent.trend == "decreasing" and abs(ent.last) < abs(ent.first) * 0.1:
            analysis["warnings"].append(
                f"Entropy collapsed from {ent.first:.4f} to {ent.last:.4f}"
            )

    # Check value loss trend
    if "train/value_loss" in m:
        vl = m["train/value_loss"]
        if vl.trend == "increasing":
            analysis["warnings"].append(
                f"Value loss increasing ({vl.first:.4f} -> {vl.last:.4f})"
            )

    # Check for goals
    if "env/goals_scored" in m:
        gs = m["env/goals_scored"]
        if gs.last > 0:
            analysis["info"].append(f"Goals being scored: {gs.last:.2f}/rollout")
        else:
            analysis["warnings"].append("No goals being scored yet")

    # Check ball touches
    if "env/ball_touches" in m:
        bt = m["env/ball_touches"]
        if bt.last < 1:
            analysis["warnings"].append(f"Very few ball touches ({bt.last:.2f}/rollout)")
        else:
            analysis["info"].append(f"Ball touches: {bt.last:.1f}/rollout")

    return analysis


def get_latest_metrics(
    run_id: str,
    project: str,
    entity: Optional[str] = None,
) -> Dict[str, float]:
    """Get the most recent value of all metrics.

    Args:
        run_id: Run ID
        project: WandB project name
        entity: WandB entity

    Returns:
        Dict of metric name -> latest value
    """
    run = get_run(run_id, project, entity)
    return dict(run.summary)


def get_metric_at_step(
    run_id: str,
    project: str,
    step: int,
    entity: Optional[str] = None,
    keys: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Get metrics at a specific step.

    Args:
        run_id: Run ID
        project: WandB project name
        step: Training step
        entity: WandB entity
        keys: Specific keys to fetch

    Returns:
        Dict of metric name -> value at step
    """
    df = get_metrics_history(run_id, project, entity, keys)

    # Find closest step
    if "_step" in df.columns:
        idx = (df["_step"] - step).abs().idxmin()
        return df.iloc[idx].to_dict()

    return {}


def compare_metrics_range(
    run_id: str,
    project: str,
    start_step: int,
    end_step: int,
    entity: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare metrics between two step ranges.

    Args:
        run_id: Run ID
        project: WandB project name
        start_step: Start of range
        end_step: End of range
        entity: WandB entity

    Returns:
        Dict with 'start', 'end', 'change' for each metric
    """
    df = get_metrics_history(run_id, project, entity)

    if "_step" not in df.columns:
        return {}

    # Get data in ranges
    start_data = df[df["_step"] <= start_step].iloc[-1] if len(df[df["_step"] <= start_step]) > 0 else None
    end_data = df[df["_step"] >= end_step].iloc[0] if len(df[df["_step"] >= end_step]) > 0 else df.iloc[-1]

    if start_data is None:
        return {}

    result = {}
    for col in df.columns:
        if col.startswith("_"):
            continue
        try:
            start_val = float(start_data[col])
            end_val = float(end_data[col])
            if not (np.isnan(start_val) or np.isnan(end_val)):
                result[col] = {
                    "start": start_val,
                    "end": end_val,
                    "change": end_val - start_val,
                    "pct_change": (end_val - start_val) / (abs(start_val) + 1e-8) * 100,
                }
        except (ValueError, TypeError):
            continue

    return result


def print_run_summary(run_id: str, project: str, entity: Optional[str] = None):
    """Print a comprehensive summary of a training run.

    Args:
        run_id: Run ID
        project: WandB project name
        entity: WandB entity
    """
    print(f"\n{'='*60}")
    print(f"WandB Run Analysis: {run_id}")
    print(f"{'='*60}\n")

    # Get run info
    run = get_run(run_id, project, entity)
    print(f"Name: {run.name}")
    print(f"State: {run.state}")
    print(f"Created: {run.created_at}")

    # Get latest step
    summary = dict(run.summary)
    step = summary.get("_step", 0)
    print(f"Current step: {step:,}")

    # Get history and analyze
    print("\nFetching metrics history...")
    df = get_metrics_history(run_id, project, entity)
    print(f"Got {len(df)} data points\n")

    # Health analysis
    analysis = analyze_training_health(df)

    print("HEALTH CHECK")
    print("-" * 40)

    if analysis["warnings"]:
        print("\nWarnings:")
        for w in analysis["warnings"]:
            print(f"  [!] {w}")

    if analysis["info"]:
        print("\nInfo:")
        for i in analysis["info"]:
            print(f"  [i] {i}")

    # Key metrics summary
    print(f"\n{'KEY METRICS':^40}")
    print("-" * 40)

    metric_groups = {
        "Losses": ["train/policy_loss", "train/value_loss", "train/entropy_loss"],
        "PPO Diagnostics": ["train/clip_fraction", "train/approx_kl", "train/explained_variance"],
        "Learning": ["train/learning_rate"],
        "Rewards": ["env/mean_reward", "env/mean_episode_reward"],
        "Goals": ["env/goals_scored", "env/goals_conceded", "env/goal_diff"],
        "Ball": ["env/ball_touches", "env/mean_touch_velocity"],
        "Performance": ["time/steps_per_second"],
    }

    for group_name, metrics in metric_groups.items():
        group_stats = [analysis["metrics"].get(m) for m in metrics if m in analysis["metrics"]]
        if group_stats:
            print(f"\n{group_name}:")
            for stats in group_stats:
                short_name = stats.name.split("/")[-1]
                print(f"  {short_name:25} {stats.last:>10.4f}  (mean={stats.mean:.4f}, {stats.trend})")

    print(f"\n{'='*60}\n")


def export_metrics_csv(
    run_id: str,
    project: str,
    output_path: str,
    entity: Optional[str] = None,
):
    """Export metrics to CSV file.

    Args:
        run_id: Run ID
        project: WandB project name
        output_path: Output CSV path
        entity: WandB entity
    """
    df = get_metrics_history(run_id, project, entity)
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} rows to {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="WandB metrics inspection")
    parser.add_argument("--project", "-p", default="rlbot-competitive", help="WandB project")
    parser.add_argument("--entity", "-e", default=None, help="WandB entity")

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # List runs
    list_parser = subparsers.add_parser("list", help="List recent runs")
    list_parser.add_argument("--limit", "-n", type=int, default=10, help="Max runs")

    # Analyze run
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a run")
    analyze_parser.add_argument("run_id", help="Run ID")

    # Export metrics
    export_parser = subparsers.add_parser("export", help="Export metrics to CSV")
    export_parser.add_argument("run_id", help="Run ID")
    export_parser.add_argument("--output", "-o", default="metrics.csv", help="Output path")

    # Get latest
    latest_parser = subparsers.add_parser("latest", help="Get latest metrics")
    latest_parser.add_argument("run_id", help="Run ID")

    args = parser.parse_args()

    if args.command == "list":
        runs = list_runs(args.project, args.entity, args.limit)
        print(f"\nRecent runs in {args.project}:\n")
        for run in runs:
            print(f"  {run['id']}  {run['name']:30}  step={run['step']:>10,}  {run['state']}")
        print()

    elif args.command == "analyze":
        print_run_summary(args.run_id, args.project, args.entity)

    elif args.command == "export":
        export_metrics_csv(args.run_id, args.project, args.output, args.entity)

    elif args.command == "latest":
        metrics = get_latest_metrics(args.run_id, args.project, args.entity)
        print(f"\nLatest metrics for {args.run_id}:\n")
        for k, v in sorted(metrics.items()):
            if not k.startswith("_"):
                print(f"  {k}: {v}")
        print()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
