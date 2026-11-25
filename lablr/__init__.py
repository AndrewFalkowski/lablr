"""lablr: Automatic point-label placement for Matplotlib.

Uses exact 2D signed distance fields (SDFs), pairwise target-distance
potentials, and AABB-accelerated L-BFGS in physical typography points.
"""

from .mpl import PlacementResult, measure_labels_pt, place_labels
from .optimizer import DEFAULT_ANGLES_DEG, LayoutMetrics, LayoutResult, optimize_layout
from .potentials import Potential
from .sdf import (
    box_box_sdf,
    box_circle_sdf,
    box_frame_sdf,
    box_point_sdf,
    box_segment_sdf,
    circle_segment_sdf,
    count_segment_crossings,
)

__all__ = [
    "place_labels",
    "measure_labels_pt",
    "PlacementResult",
    "Potential",
    "DEFAULT_ANGLES_DEG",
    "optimize_layout",
    "LayoutResult",
    "LayoutMetrics",
    "box_point_sdf",
    "box_box_sdf",
    "box_circle_sdf",
    "box_frame_sdf",
    "box_segment_sdf",
    "circle_segment_sdf",
    "count_segment_crossings",
]
