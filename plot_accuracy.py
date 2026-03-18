#!/usr/bin/env python3
"""Plot accuracy from result JSON files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot accuracy from subject result JSON files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing subject_*.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/accuracy_plot.png"),
        help="Path to save the accuracy plot image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window in addition to saving.",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=Path("results/accuracy_table.png"),
        help="Path to save the accuracy table image.",
    )
    return parser.parse_args()


def load_accuracy(results_dir: Path) -> tuple[list[int], dict[str, list[float]]]:
    files = sorted(results_dir.glob("subject_*.json"))
    if not files:
        raise FileNotFoundError(f"No subject_*.json files found in: {results_dir}")

    subject_to_attention_values: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        subject_id = payload.get("subject_id")
        if subject_id is None:
            continue

        for run in payload.get("runs", []):
            if run.get("status") != "ok":
                continue
            accuracy = run.get("test_accuracy")
            attention = run.get("attention", "unknown")
            if accuracy is None:
                continue
            subject_to_attention_values[subject_id][attention].append(float(accuracy))

    if not subject_to_attention_values:
        raise ValueError("No valid accuracy data found.")

    subjects = sorted(subject_to_attention_values.keys())
    attentions = sorted(
        {
            attention
            for by_attention in subject_to_attention_values.values()
            for attention in by_attention.keys()
        }
    )

    aggregated: dict[str, list[float]] = {attention: [] for attention in attentions}
    for subject_id in subjects:
        by_attention = subject_to_attention_values[subject_id]
        for attention in attentions:
            values = by_attention.get(attention, [])
            aggregated[attention].append(sum(values) / len(values) if values else 0.0)

    return subjects, aggregated


def plot_accuracy(
    subjects: list[int],
    aggregated: dict[str, list[float]],
    output_path: Path,
    show_plot: bool,
) -> None:
    attentions = list(aggregated.keys())
    x_positions = list(range(len(subjects)))
    width = 0.8 / max(1, len(attentions))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, attention in enumerate(attentions):
        offset = (i - (len(attentions) - 1) / 2) * width
        ax.bar(
            [x + offset for x in x_positions],
            aggregated[attention],
            width=width,
            label=attention,
        )

    ax.set_title("Test Accuracy by Subject")
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(subjects)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Attention")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

    if show_plot:
        plt.show()
    plt.close(fig)


def plot_accuracy_table(
    subjects: list[int],
    aggregated: dict[str, list[float]],
    output_path: Path,
    show_plot: bool,
) -> None:
    attentions = list(aggregated.keys())
    column_labels = [f"S{subject_id}" for subject_id in subjects]
    row_labels = attentions
    cell_text = [
        [f"{value * 100:.1f}%" for value in aggregated[attention]]
        for attention in attentions
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(subjects) * 1.2), 2 + len(attentions)))
    ax.axis("off")
    ax.set_title("Mean Test Accuracy Across Seeds", pad=12)

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=column_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

    if show_plot:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    subjects, aggregated = load_accuracy(args.results_dir)
    plot_accuracy(subjects, aggregated, args.output, args.show)
    plot_accuracy_table(subjects, aggregated, args.table_output, args.show)
    print(f"Saved accuracy plot to: {args.output}")
    print(f"Saved accuracy table to: {args.table_output}")


if __name__ == "__main__":
    main()
