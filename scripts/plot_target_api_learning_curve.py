#!/usr/bin/env python3
"""Plot the formal three-API fine-tuning learning curve."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TASKS = ["binary", "four-class"]
METRICS = ["accuracy", "balanced_accuracy"]


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect_curve_points(learning_summary, formal_summary):
    learning = _read_json(learning_summary)
    formal = _read_json(formal_summary)
    points = []

    for row in formal["aggregate"]:
        if row["stage"] not in {"zero_shot", "fine_tuned_full"}:
            continue
        points.append(
            {
                "task": row["task"],
                "stage": row["stage"],
                "fine_tuning_pairs": int(row["fine_tuning_pairs"]),
                "runs": int(row["runs"]),
                **{
                    f"{metric}_{suffix}": float(row[f"{metric}_{suffix}"])
                    for metric in METRICS
                    for suffix in ("mean", "std")
                },
            }
        )
    for row in learning["aggregate"]:
        if row["stage"] != "fine_tuned_learning_curve":
            continue
        points.append(
            {
                "task": row["task"],
                "stage": row["stage"],
                "fine_tuning_pairs": int(row["fine_tuning_pairs"]),
                "runs": int(row["runs"]),
                **{
                    f"{metric}_{suffix}": float(row[f"{metric}_{suffix}"])
                    for metric in METRICS
                    for suffix in ("mean", "std")
                },
            }
        )

    expected_sizes = [0, 8, 16, 32, 48, 136]
    for task in TASKS:
        task_points = [point for point in points if point["task"] == task]
        sizes = sorted(point["fine_tuning_pairs"] for point in task_points)
        if sizes != expected_sizes:
            raise ValueError(
                f"Expected {task} learning-curve sizes {expected_sizes}, got {sizes}"
            )
        if any(point["runs"] != 3 for point in task_points):
            raise ValueError(f"Expected three model seeds for every {task} point")
    return sorted(
        points,
        key=lambda point: (TASKS.index(point["task"]), point["fine_tuning_pairs"]),
    )


def write_points(path, points):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(points[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(points)


def plot_curve(points, output_prefix):
    import matplotlib.pyplot as plt

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    style = {
        "accuracy": {
            "label": "Accuracy",
            "color": "#1F77B4",
            "marker": "o",
        },
        "balanced_accuracy": {
            "label": "Balanced accuracy",
            "color": "#D55E00",
            "marker": "s",
        },
    }
    titles = {
        "binary": "Binary classification",
        "four-class": "Four-class classification",
    }
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.4, 4.2),
        sharey=True,
        constrained_layout=True,
    )
    for axis, task in zip(axes, TASKS):
        task_points = [point for point in points if point["task"] == task]
        x_values = [point["fine_tuning_pairs"] for point in task_points]
        for metric in METRICS:
            axis.errorbar(
                x_values,
                [point[f"{metric}_mean"] for point in task_points],
                yerr=[point[f"{metric}_std"] for point in task_points],
                label=style[metric]["label"],
                color=style[metric]["color"],
                marker=style[metric]["marker"],
                linewidth=1.8,
                markersize=5.5,
                capsize=3,
            )
        axis.set_title(titles[task], fontsize=11, fontweight="bold")
        axis.set_xlabel("Fine-tuning pairs per outer fold")
        axis.set_xticks([0, 8, 16, 32, 48, 136])
        axis.set_xlim(-5, 143)
        axis.set_ylim(0.3, 1.0)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Performance")
    axes[0].legend(frameon=False, loc="lower right")
    for suffix in ("png", "pdf"):
        figure.savefig(
            output_prefix.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-summary", required=True)
    parser.add_argument("--formal-summary", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--points-output")
    return parser.parse_args()


def main():
    args = parse_args()
    points = collect_curve_points(args.learning_summary, args.formal_summary)
    output_prefix = Path(args.output_prefix)
    points_output = (
        Path(args.points_output)
        if args.points_output
        else output_prefix.with_suffix(".csv")
    )
    write_points(points_output, points)
    plot_curve(points, output_prefix)
    print(f"Learning-curve points: {points_output}")
    print(f"Learning-curve PNG: {output_prefix.with_suffix('.png')}")
    print(f"Learning-curve PDF: {output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
