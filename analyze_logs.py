#!/usr/bin/env python3
"""
Parse all JSONL logs under logs/ and generate per-run metric plots plus a
summary JSON file that captures the best BLEU/chrF epoch for each run.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

MetricRecord = Dict[str, float]

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


def derive_run_name(path: Path, record: Dict) -> str:
    """Build a descriptive name for a sequence of records."""
    base = path.stem
    variant = record.get("variant")
    if path.name == "transformer_train_metrics.jsonl":
        positional = record.get("positional_encoding")
        norm = record.get("norm_type")
        pretrained = record.get("pretrained_model")
        if pretrained:
            parts = [base, variant or "t5", pretrained]
        else:
            parts = [base]
            if variant:
                parts.append(variant)
            if positional:
                parts.append(positional)
            if norm:
                parts.append(norm)
        name = "_".join(parts)
    else:
        name = base
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return sanitized or base


def load_logs(log_dir: Path) -> Dict[str, Dict]:
    """Read every log file and bucket records by run name."""
    runs: Dict[str, Dict] = {}
    for path in sorted(log_dir.iterdir()):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                run_name = derive_run_name(path, record)
                info = runs.setdefault(
                    run_name,
                    {
                        "file": str(path),
                        "records": [],
                    },
                )
                info["records"].append(record)
    return runs


def collect_metrics(records: List[Dict]) -> List[str]:
    """Return ordered metric names to visualize."""
    available = []
    for metric in PREFERRED_METRICS:
        if any(metric in record for record in records):
            available.append(metric)
    return available


def plot_run(run_name: str, records: List[Dict], output_dir: Path) -> Optional[Path]:
    metrics = collect_metrics(records)
    if not metrics:
        return None
    epochs = [record.get("epoch", idx + 1) for idx, record in enumerate(records)]
    cols = 2
    rows = math.ceil(len(metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    for idx, metric in enumerate(metrics):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        values = [
            record.get(metric)
            for record in records
            if record.get(metric) is not None
        ]
        metric_epochs = [
            records[i].get("epoch", i + 1)
            for i, record in enumerate(records)
            if record.get(metric) is not None
        ]
        ax.plot(metric_epochs, values, marker="o", linewidth=1.5)
        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.4)
    # Hide unused axes
    total_axes = rows * cols
    for idx in range(len(metrics), total_axes):
        row, col = divmod(idx, cols)
        axes[row][col].axis("off")
    fig.suptitle(run_name, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_name}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def find_best_metric(records: List[Dict], key: str) -> Optional[Dict]:
    best_record = None
    best_value = None
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_record = record
    if best_record is None:
        return None
    return {
        "value": best_value,
        "epoch": best_record.get("epoch"),
        "timestamp": best_record.get("timestamp"),
    }


def summarize_runs(runs: Dict[str, Dict]) -> Dict[str, Dict]:
    summary = {}
    for run_name, info in runs.items():
        records = info["records"]
        # sort by epoch for reproducibility
        records = sorted(records, key=lambda r: r.get("epoch", 0))
        info["records"] = records
        best_bleu = find_best_metric(records, "valid_bleu")
        best_chrf = find_best_metric(records, "valid_chrf")
        summary[run_name] = {
            "log_file": info["file"],
            "best_bleu": best_bleu,
            "best_chrf": best_chrf,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze training logs and plot metrics.")
    parser.add_argument("--log_dir", type=Path, default=Path("logs"), help="Directory containing JSONL logs.")
    parser.add_argument("--plot_dir", type=Path, default=Path("logs/plots"), help="Directory to store metric plots.")
    parser.add_argument("--summary_path", type=Path, default=Path("logs/best_metrics_summary.json"), help="Output JSON path.")
    args = parser.parse_args()

    runs = load_logs(args.log_dir)
    summary = summarize_runs(runs)

    generated_plots = []
    for run_name, info in runs.items():
        output_path = plot_run(run_name, info["records"], args.plot_dir)
        if output_path:
            generated_plots.append(str(output_path))

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(runs)} runs.")
    print(f"Saved summary to {args.summary_path}")
    if generated_plots:
        print("Generated plots:")
        for path in generated_plots:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
