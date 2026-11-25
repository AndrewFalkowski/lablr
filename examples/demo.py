"""Demonstration of lablr across multiple plot types with flush, color-matched leashes."""

from __future__ import annotations

import argparse
import time
from typing import Any

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from lablr import PlacementResult, place_labels


def _run_panel(
    ax: matplotlib.axes.Axes,
    points: Any,
    labels: list[str],
    *,
    distance: float = 8.0,
    text_kwargs: dict[str, Any] | None = None,
    bbox_kwargs: dict[str, Any] | None = None,
) -> PlacementResult:
    t0 = time.perf_counter()
    res = place_labels(
        ax,
        points,
        labels,
        distance=distance,
        text_kwargs=text_kwargs,
        bbox_kwargs=bbox_kwargs,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    m = res.metrics
    print(
        f"{ax.get_title():44s} | {dt_ms:5.1f} ms | "
        f"lbl_ov={m.n_label_overlaps} pt_ov={m.n_point_overlaps} "
        f"line_ov={m.n_line_overlaps} frame={m.n_frame_clips} cross={m.n_crossings}"
    )
    return res


def build_demo_figure() -> matplotlib.figure.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=150)

    # 1. Dense random scatter (N=40)
    ax = axes[0, 0]
    np.random.seed(1)
    points = np.random.rand(40, 2)
    labels = [f"Label {i}" for i in range(40)]
    ax.scatter(points[:, 0], points[:, 1], s=50, color="tab:blue", zorder=3)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_title("Dense Scatter (N=40)")

    # 2. Two dense clusters with varied label lengths
    ax = axes[0, 1]
    rng = np.random.default_rng(12)
    c1 = rng.normal([0.25, 0.72], 0.06, size=(12, 2))
    c2 = rng.normal([0.75, 0.30], 0.06, size=(12, 2))
    pts2 = np.vstack([c1, c2])
    labels2 = [f"Cluster A #{i}" if i % 3 == 0 else f"A{i}" for i in range(12)] + [
        f"Long Label B-{i}" if i % 4 == 0 else f"B{i}" for i in range(12)
    ]
    ax.scatter(pts2[:, 0], pts2[:, 1], s=65, color="tab:orange", zorder=3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Two Dense Clusters + Varied Label Lengths")

    # 3. Log-X + extreme aspect ratio + large markers
    ax = axes[1, 0]
    rng3 = np.random.default_rng(42)
    x3 = 10.0 ** rng3.uniform(0.5, 4.5, size=20)
    y3 = rng3.uniform(0.001, 0.01, size=20)
    labels3 = [f"Pt {i} ({y3[i] * 1e3:.1f}m)" for i in range(20)]
    ax.scatter(x3, y3, s=110, color="tab:green", zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(1.0, 1e5)
    ax.set_ylim(0.0, 0.0115)
    ax.set_title("Log-X + Extreme Aspect Ratio + Large Markers")

    # 4. Selective labeling while avoiding unlabeled background points
    ax = axes[1, 1]
    rng4 = np.random.default_rng(99)
    bg_pts = rng4.uniform(0.05, 0.95, size=(70, 2))
    hi_idx = rng4.choice(len(bg_pts), size=14, replace=False)
    hi_pts = bg_pts[hi_idx]
    hi_labels = [f"Highlight {i}" for i in range(len(hi_idx))]
    ax.scatter(bg_pts[:, 0], bg_pts[:, 1], s=35, color="0.78", zorder=2)
    ax.scatter(hi_pts[:, 0], hi_pts[:, 1], s=80, color="tab:red", zorder=3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Selective Labels Avoiding Unlabeled Points")

    fig.tight_layout(pad=2.0)

    box: dict[str, Any] = {"boxstyle": "round,pad=0.2", "facecolor": "white", "linewidth": 0.9}
    _run_panel(axes[0, 0], points, labels, text_kwargs={"fontsize": 8.5}, bbox_kwargs=box)
    _run_panel(
        axes[0, 1], pts2, labels2, distance=10.0, text_kwargs={"fontsize": 8.5}, bbox_kwargs=box
    )
    _run_panel(axes[1, 0], (x3, y3), labels3, text_kwargs={"fontsize": 8.5}, bbox_kwargs=box)
    _run_panel(
        axes[1, 1],
        hi_pts,
        hi_labels,
        text_kwargs={"fontsize": 8.5, "fontweight": "bold", "color": "#8b0000"},
        bbox_kwargs=box,
    )

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lablr visual demo.")
    parser.add_argument(
        "--save", type=str, default="examples/demo.png", help="Save figure to path."
    )
    args = parser.parse_args()

    fig = build_demo_figure()
    if args.save:
        fig.savefig(args.save, dpi=180)
        print(f"Saved demo figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
