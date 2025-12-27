#!/usr/bin/env python3
"""
Plot transformer log metrics for requested comparison groups.
Each metric is saved as its own image under subfolders.
"""
from __future__ import annotations

import argparse
import json
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


def smooth_series(values: List[float], window: int) -> List[float]:
    if window <= 1 or window > len(values):
        return values
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def load_records(path: Path) -> List[Dict]:
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
    records.sort(key=lambda r: r.get("epoch", 0))
    return records


def load_runs(mapping: Dict[str, Path]) -> Tuple[Dict[str, List[Dict]], List[str]]:
    runs: Dict[str, List[Dict]] = {}
    missing: List[str] = []
    for label, path in mapping.items():
        if not path.exists():
            missing.append(f"{label} -> {path}")
            continue
        records = load_records(path)
        if records:
            runs[label] = records
        else:
            missing.append(f"{label} -> {path} (empty)")
    return runs, missing


def collect_metrics(runs: Dict[str, List[Dict]]) -> List[str]:
    metrics: List[str] = []
    for metric in PREFERRED_METRICS:
        if any(metric in record for records in runs.values() for record in records):
            metrics.append(metric)
    return metrics


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
        if len(handles) > 8:
            ax.legend(
                handles,
                labels,
                fontsize=7,
                ncol=2,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
            )
            fig.tight_layout(rect=[0, 0, 0.78, 1])
        else:
            ax.legend(handles, labels, fontsize=8)
            fig.tight_layout()
    else:
        fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_group(
    group_name: str,
    mapping: Dict[str, Path],
    output_dir: Path,
    smooth_window: int,
) -> List[Path]:
    runs, missing = load_runs(mapping)
    if missing:
        print(f"[{group_name}] missing logs:")
        for item in missing:
            print(f"  - {item}")
    if not runs:
        print(f"[{group_name}] no valid logs found.")
        return []
    metrics = collect_metrics(runs)
    generated: List[Path] = []
    group_dir = output_dir / group_name
    for metric in metrics:
        output_path = group_dir / f"{metric}.png"
        plot_metric(metric, runs, output_path, smooth_window)
        generated.append(output_path)
    return generated


def plot_single_run(
    run_name: str,
    records: List[Dict],
    output_dir: Path,
    smooth_window: int,
) -> List[Path]:
    metrics = collect_metrics({run_name: records})
    if not metrics:
        raise ValueError(f"No metrics found in {run_name}.")
    generated: List[Path] = []
    for metric in metrics:
        output_path = output_dir / f"{metric}.png"
        plot_metric(metric, {run_name: records}, output_path, smooth_window)
        generated.append(output_path)
    return generated


def build_positional_norm_mapping(log_dir: Path) -> Dict[str, Path]:
    variants = [
        ("sin", "layer"),
        ("sin", "rms"),
        ("learn", "layer"),
        ("learn", "rms"),
        ("relative", "layer"),
        ("relative", "rms"),
    ]
    mapping: Dict[str, Path] = {}
    for pos, norm in variants:
        label = f"{pos}_{norm}"
        mapping[label] = log_dir / f"transformer_{pos}_{norm}"
    return mapping


def build_batch_lr_mapping(log_dir: Path) -> Dict[str, Path]:
    batches = [64, 128, 256]
    lrs = ["3", "4", "5"]
    mapping: Dict[str, Path] = {}
    for batch in batches:
        for lr in lrs:
            label = f"batch{batch}_lr{lr}"
            path = log_dir / f"transformer_batch{batch}_lr{lr}"
            mapping[label] = path
    # Override: batch256 lr4 uses sin layer log.
    mapping["batch256_lr4_sin_layer"] = log_dir / "transformer_sin_layer"
    mapping.pop("batch256_lr4", None)
    return mapping


def build_dim_head_layer_mapping(log_dir: Path) -> Dict[str, Path]:
    dims = [256, 512, 1024]
    heads = [4, 8, 16]
    layers = [4, 6, 8]
    mapping: Dict[str, Path] = {}
    for d_model in dims:
        for num_heads in heads:
            for num_layers in layers:
                label = f"d{d_model}_h{num_heads}_l{num_layers}"
                path = log_dir / f"transformer_d{d_model}_h{num_heads}_l{num_layers}"
                mapping[label] = path
    # Override: d512 h8 l6 uses sin layer log.
    mapping["d512_h8_l6_sin_layer"] = log_dir / "transformer_sin_layer"
    mapping.pop("d512_h8_l6", None)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare transformer log metrics.")
    parser.add_argument("--log_dir", type=Path, default=Path("logs"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("logs/plots/transformer_comparisons"),
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=3,
        help="Trailing moving average window (>=1).",
    )
    parser.add_argument(
        "--t5_log",
        type=Path,
        default=Path("logs/transformer_t5base"),
        help="T5 base log file path.",
    )
    parser.add_argument(
        "--t5_output",
        type=Path,
        default=Path("logs/plots/transformer_t5base"),
        help="Output directory for T5 base plots.",
    )
    args = parser.parse_args()

    groups = {
        "positional_norm": build_positional_norm_mapping(args.log_dir),
        "batch_lr": build_batch_lr_mapping(args.log_dir),
        "dim_head_layer": build_dim_head_layer_mapping(args.log_dir),
    }

    total = 0
    for group_name, mapping in groups.items():
        generated = plot_group(group_name, mapping, args.output_dir, args.smooth_window)
        total += len(generated)
        if generated:
            print(f"[{group_name}] saved {len(generated)} plots to {args.output_dir / group_name}")
    if args.t5_log.exists():
        records = load_records(args.t5_log)
        if records:
            generated = plot_single_run("t5base", records, args.t5_output, args.smooth_window)
            print(f"[t5base] saved {len(generated)} plots to {args.t5_output}")
            total += len(generated)
        else:
            print(f"[t5base] log is empty: {args.t5_log}")
    else:
        print(f"[t5base] log not found: {args.t5_log}")
    if total == 0:
        raise SystemExit("No plots generated. Check that logs exist under the log_dir.")


if __name__ == "__main__":
    main()
