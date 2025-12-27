#!/usr/bin/env python3
"""
Plot RNN log metrics on a single figure with one subplot per metric.
Each subplot overlays curves from multiple runs for easy comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

PREFERRED_METRICS = [
    "train_loss",
    "train_ppl",
    "train_tok_per_sec",
    "train_samples_per_sec",
    "valid_loss",
    "valid_ppl",
    "valid_tok_per_sec",
    "valid_bleu",
    "valid_chrf",
    "avg_decode_time",
]

RNN_LOG_PATTERN = re.compile(
    r"^rnn_(dot|additive|multiplicative)_(teacher|running)(?:_beam)?$"
)


def load_runs(log_dir: Path) -> Dict[str, List[Dict]]:
    runs: Dict[str, List[Dict]] = {}
    for path in sorted(log_dir.iterdir()):
        if not path.is_file():
            continue
        if not RNN_LOG_PATTERN.match(path.name):
            continue
        records: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(record)
        if records:
            records.sort(key=lambda r: r.get("epoch", 0))
            runs[path.name] = records
    return runs


def collect_metrics(runs: Dict[str, List[Dict]]) -> List[str]:
    metrics: List[str] = []
    for metric in PREFERRED_METRICS:
        if any(metric in record for records in runs.values() for record in records):
            metrics.append(metric)
    return metrics


def smooth_series(values: List[float], window: int) -> List[float]:
    if window <= 1 or window > len(values):
        return values
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def plot_metric(
    metric: str,
    runs: Dict[str, List[Dict]],
    output_path: Path,
    smooth_window: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for run_name, records in runs.items():
        values = [
            record.get(metric)
            for record in records
            if record.get(metric) is not None
        ]
        epochs = [
            record.get("epoch", i + 1)
            for i, record in enumerate(records)
            if record.get(metric) is not None
        ]
        if not values:
            continue
        if smooth_window > 1:
            values = smooth_series(values, smooth_window)
        ax.plot(epochs, values, marker="o", linewidth=1.2, markersize=3, label=run_name)
    ax.set_title(metric)
    ax.set_xlabel("Epoch")
    ax.grid(True, linestyle="--", alpha=0.4)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_comparison(
    runs: Dict[str, List[Dict]],
    output_dir: Path,
    smooth_window: int,
) -> List[Tuple[str, Path]]:
    metrics = collect_metrics(runs)
    if not metrics:
        raise ValueError("No metrics found in the selected logs.")
    generated: List[Tuple[str, Path]] = []
    for metric in metrics:
        output_path = output_dir / f"{metric}.png"
        plot_metric(metric, runs, output_path, smooth_window)
        generated.append((metric, output_path))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RNN log metrics in one figure.")
    parser.add_argument("--log_dir", type=Path, default=Path("logs"), help="Directory containing log files.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("logs/plots/rnn_metrics_comparison"),
        help="Directory to store per-metric comparison plots.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=3,
        help="Trailing moving average window (>=1).",
    )
    args = parser.parse_args()

    runs = load_runs(args.log_dir)
    if not runs:
        raise SystemExit(f"No matching RNN logs found under {args.log_dir}")

    generated = plot_comparison(runs, args.output_dir, args.smooth_window)
    print(f"Saved {len(generated)} plots to {args.output_dir}")


if __name__ == "__main__":
    main()
