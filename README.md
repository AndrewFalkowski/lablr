# lablr

Automatic point-label placement for Matplotlib using **exact 2D signed distance fields (SDFs)**, **pairwise target-distance potentials**, and **AABB-accelerated FIRE 2.0** in physical typography points.

## How it works

- **Two intuitive spacing controls (in points)**:
  - `distance` (default `8.0` pt): preferred edge-to-marker gap between a label and its own point (well width controlled by `tolerance`).
  - `padding` (default `4.0` pt): minimum clearance from everything else — other labels, other points, leader lines, and the plot border.
- **Exact 2D SDFs (`lablr.sdf`)**: Computes exact surface-to-surface separation ($d > 0$), penetration depth ($d < 0$), and analytic gradients for boxes, circular markers, leader-line segments, crossed leader lines (`segment_segment_sdf`), fixed obstacles, and plot walls, accelerated by exact AABB cutoff pruning.
- **Racetrack initialization & capsule-smooth FIRE 2.0 (`lablr.optimizer`)**:
  - Samples initial candidates along exact SDF level-set racetracks and primary compass directions, measuring gaps from the **nearest box edge or corner** so short and long labels sit at identical visual gaps.
  - Gently prefers clean 8-direction compass angles at snug distances and fades smoothly when labels reach out of crowded clusters.
  - Relaxes coupled labels via line-search-free **FIRE 2.0** (Fast Inertial Relaxation Engine), using a brief capsule-rounded warm-up pass so labels slide smoothly past corners before exact rectangular polishing and size-aware 2-opt uncrossing.
- **Accurate Matplotlib sizing & styling (`lablr.mpl`)**:
  - Optimizes in isotropic **typography points** (`1 pt = 1/72 in`), invariant to aspect ratio, log/skewed axes, and export DPI.
  - Measures true `FancyBboxPatch` extents (including padding and border stroke), auto-detects marker radii and unlabeled background points, and draws flush leashes terminating at the marker rim, attached to the nearest box edge/corner and color-matched to each point.
- **Minimal dependencies**: Only `numpy` and `matplotlib`.

## Quick start

```python
import matplotlib.pyplot as plt
import numpy as np
from lablr import place_labels

rng = np.random.default_rng(1)
points = rng.random((40, 2))
labels = [f"Label {i}" for i in range(len(points))]

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
ax.scatter(points[:, 0], points[:, 1], s=50, color="tab:blue", zorder=3)
ax.set_xlim(-0.08, 1.08)
ax.set_ylim(-0.08, 1.08)
fig.tight_layout()  # finalize axes layout before placing labels

result = place_labels(
    ax,
    points,
    labels,
    distance=8.0,  # preferred label-to-point gap (pt)
    padding=4.0,  # clearance from other labels, points, lines & border (pt)
    text_kwargs={"fontsize": 9},
    bbox_kwargs={"boxstyle": "round,pad=0.2", "facecolor": "white"},
)
print(result.metrics)
plt.show()
```

## Development

```bash
pip install -e ".[dev]"
pytest
python examples/demo.py
```
