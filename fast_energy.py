import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from typing import Union, List
from numpy.typing import ArrayLike, NDArray


@njit
def line_intersection_on_rect_fast(width, height, xB, yB, xA, yA):
    """
    Optimized version using numba for JIT compilation.
    """
    # Half dimensions
    w = width / 2
    h = height / 2

    # Direction vector from B to A
    dx = xA - xB
    dy = yA - yB

    # If A=B return B itself
    if dx == 0 and dy == 0:
        return xB, yB

    # Calculate slopes
    tan_phi = h / w
    tan_theta = abs(dy / dx) if dx != 0 else np.inf

    # Determine quadrant
    qx = 1 if dx > 0 else -1 if dx < 0 else 0
    qy = 1 if dy > 0 else -1 if dy < 0 else 0

    # Calculate intersection
    if dx == 0:  # Vertical line
        xI = xB
        yI = yB + h * qy
    elif dy == 0:  # Horizontal line
        xI = xB + w * qx
        yI = yB
    elif tan_theta > tan_phi:  # Intersects top or bottom edge
        xI = xB + (h / tan_theta) * qx
        yI = yB + h * qy
    else:  # Intersects left or right edge
        xI = xB + w * qx
        yI = yB + w * tan_theta * qy

    return xI, yI


@njit
def compute_adjusted_distance_label_point(px, py, lx, ly, lw, lh):
    """
    Calculate distance from point to label, subtracting the internal label rectangle distance.

    Parameters:
    px, py: Point coordinates
    lx, ly: Label center coordinates
    lw, lh: Label width and height
    """
    # Calculate total distance
    total_distance = np.sqrt((lx - px) ** 2 + (ly - py) ** 2)

    # Calculate intersection with label rectangle
    int_x, int_y = line_intersection_on_rect_fast(lw, lh, lx, ly, px, py)
    internal_distance = np.sqrt((int_x - lx) ** 2 + (int_y - ly) ** 2)

    # Return adjusted distance (max to ensure non-negative)
    return max(total_distance - internal_distance, 0.0)


@njit
def compute_adjusted_distance_label_label(l1x, l1y, l1w, l1h, l2x, l2y, l2w, l2h):
    """
    Calculate distance between two labels, subtracting internal rectangle distances.

    Parameters:
    l1x, l1y: Label 1 center coordinates
    l1w, l1h: Label 1 width and height
    l2x, l2y: Label 2 center coordinates
    l2w, l2h: Label 2 width and height
    """
    # Calculate total distance
    total_distance = np.sqrt((l2x - l1x) ** 2 + (l2y - l1y) ** 2)

    # Calculate intersection with both rectangles
    int1_x, int1_y = line_intersection_on_rect_fast(l1w, l1h, l1x, l1y, l2x, l2y)
    int2_x, int2_y = line_intersection_on_rect_fast(l2w, l2h, l2x, l2y, l1x, l1y)

    internal_distance1 = np.sqrt((int1_x - l1x) ** 2 + (int1_y - l1y) ** 2)
    internal_distance2 = np.sqrt((int2_x - l2x) ** 2 + (int2_y - l2y) ** 2)

    # Return adjusted distance (max to ensure non-negative)
    return max(total_distance - internal_distance1 - internal_distance2, 0.0)


def get_label_dimensions(ax, labels):
    """
    Pre-compute width and height for all labels.

    Returns:
    numpy.ndarray: Array of shape (n_labels, 2) with [width, height] for each label
    """
    label_dims = np.zeros((len(labels), 2), dtype=np.float64)

    for i, label in enumerate(labels):
        text = ax.text(0, 0, label, ha="center", va="center")
        bbox = text.get_window_extent().transformed(ax.transData.inverted())
        text.remove()
        label_dims[i, 0] = bbox.width
        label_dims[i, 1] = bbox.height

    return label_dims


def calc_label_point_energies_fast(
    anchors: np.ndarray,
    label_pos: np.ndarray,
    label_dims: np.ndarray,
    influence_extent: float,
) -> np.ndarray:
    """
    Calculate energy for each label based on distances to anchor points,
    accounting for label rectangle dimensions.

    Parameters:
    anchors: Array of shape (n_anchors, 2) with anchor coordinates
    label_pos: Array of shape (n_labels, 2) with label positions
    label_dims: Array of shape (n_labels, 2) with [width, height] for each label
    influence_extent: The influence range for energy calculation

    Returns:
    Array of shape (n_labels,) with energy values
    """
    n_anchors = len(anchors)
    n_labels = len(label_pos)

    # Pre-allocate distance matrix
    distances = np.zeros((n_anchors, n_labels), dtype=np.float64)

    # Compute adjusted distances
    for i in range(n_anchors):
        for j in range(n_labels):
            distances[i, j] = compute_adjusted_distance_label_point(
                anchors[i, 0],
                anchors[i, 1],
                label_pos[j, 0],
                label_pos[j, 1],
                label_dims[j, 0],
                label_dims[j, 1],
            )

    # Apply energy function
    energies = energy_func(distances, influence_extent)

    # Sum energies for each label
    return np.sum(energies, axis=0)


def calc_label_label_energies_fast(
    label_pos: np.ndarray, label_dims: np.ndarray, influence_extent: float
) -> np.ndarray:
    """
    Calculate energy for each label based on distances to other labels,
    accounting for label rectangle dimensions.

    Parameters:
    label_pos: Array of shape (n_labels, 2) with label positions
    label_dims: Array of shape (n_labels, 2) with [width, height] for each label
    influence_extent: The influence range for energy calculation

    Returns:
    Array of shape (n_labels,) with energy values
    """
    n_labels = len(label_pos)

    # Pre-allocate distance matrix
    distances = np.zeros((n_labels, n_labels), dtype=np.float64)

    # Compute adjusted distances
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            distance = compute_adjusted_distance_label_label(
                label_pos[i, 0],
                label_pos[i, 1],
                label_dims[i, 0],
                label_dims[i, 1],
                label_pos[j, 0],
                label_pos[j, 1],
                label_dims[j, 0],
                label_dims[j, 1],
            )
            distances[i, j] = distance
            distances[j, i] = distance  # Symmetric

    # Apply energy function
    energies = energy_func(distances, influence_extent)

    # Sum energies for each label (excluding self-interactions)
    np.fill_diagonal(energies, 0)
    return np.sum(energies, axis=0)


def energy_func(
    distance: Union[float, ArrayLike], influence_extent: float
) -> Union[float, NDArray[np.float64]]:
    """Compute energy values using a Gaussian function centered at zero.

    Args:
        distance: Scalar distance value or array of distances.
        influence_extent: Distance at which the Gaussian PDF equals 0.01.

    Returns:
        Scalar or array of energy values based on Gaussian PDF.
    """

    sigma = influence_extent / np.sqrt(2 * np.log(100))

    # Gaussian PDF
    return np.exp(-(distance**2) / (2 * sigma**2))


import numpy as np
from numba import njit
import matplotlib.pyplot as plt


@njit
def distance_between_segments_fast(x0, y0, x1, y1, x2, y2, x3, y3):
    """
    Calculate shortest distance between two line segments using Numba.

    Parameters:
    x0, y0, x1, y1: Coordinates of first line segment
    x2, y2, x3, y3: Coordinates of second line segment

    Returns:
    float: shortest distance between the segments
    """
    # Convert to vectors
    ux = x1 - x0
    uy = y1 - y0
    vx = x3 - x2
    vy = y3 - y2
    wx = x0 - x2
    wy = y0 - y2

    a = ux * ux + uy * uy
    b = ux * vx + uy * vy
    c = vx * vx + vy * vy
    d = ux * wx + uy * wy
    e = vx * wx + vy * wy

    D = a * c - b * b

    # Initialize
    sc = sN = sD = D
    tc = tN = tD = D

    # Compute the line parameters of the two closest points
    if D < 1e-8:  # The lines are almost parallel
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = b * e - c * d
        tN = a * e - b * d
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = -d + b
            sD = a

    sc = 0.0 if abs(sN) < 1e-8 else sN / sD
    tc = 0.0 if abs(tN) < 1e-8 else tN / tD

    # Calculate the closest points
    p1x = x0 + sc * ux
    p1y = y0 + sc * uy
    p2x = x2 + tc * vx
    p2y = y2 + tc * vy

    # Return the distance between the closest points
    dx = p1x - p2x
    dy = p1y - p2y
    return np.sqrt(dx * dx + dy * dy)


def calc_line_proximity_energies_fast(anchors, label_pos, influence_extent):
    """
    Calculate energy based on distances between anchor-label lines.
    Uses the existing energy function form.

    Parameters:
    anchors: Array of shape (n, 2) containing anchor points
    label_pos: Array of shape (n, 2) containing label positions
    influence_extent: The influence range for energy calculation

    Returns:
    Array of shape (n,) containing energy values for each label position
    """
    n_labels = len(label_pos)
    energies = np.zeros(n_labels)

    # Check each pair of lines for proximity
    for i in range(n_labels):
        for j in range(n_labels):
            if i == j:
                continue

            distance = distance_between_segments_fast(
                anchors[i, 0],
                anchors[i, 1],
                label_pos[i, 0],
                label_pos[i, 1],
                anchors[j, 0],
                anchors[j, 1],
                label_pos[j, 0],
                label_pos[j, 1],
            )

            # Use the existing energy function
            energies[i] += energy_func(distance, influence_extent)

    return energies


@njit
def calc_line_midpoint_energies_fast(anchors, label_pos, influence_extent):
    """
    Calculate energy based on distances between midpoints of anchor-label lines.
    This is a simplified but faster approach.

    Parameters:
    anchors: Array of shape (n, 2) containing anchor points
    label_pos: Array of shape (n, 2) containing label positions
    influence_extent: The influence range for energy calculation

    Returns:
    Array of shape (n,) containing energy values for each label position
    """
    n_labels = len(label_pos)
    energies = np.zeros(n_labels)

    # Pre-compute midpoints for all lines
    midpoints = np.empty((n_labels, 2))
    for i in range(n_labels):
        midpoints[i, 0] = (anchors[i, 0] + label_pos[i, 0]) / 2
        midpoints[i, 1] = (anchors[i, 1] + label_pos[i, 1]) / 2

    # Calculate sigma for Gaussian
    sigma = influence_extent / np.sqrt(2 * np.log(100))

    # Calculate pairwise distances between midpoints
    for i in range(n_labels):
        for j in range(n_labels):
            if i == j:
                continue

            dx = midpoints[i, 0] - midpoints[j, 0]
            dy = midpoints[i, 1] - midpoints[j, 1]
            distance = np.sqrt(dx * dx + dy * dy)

            # Apply Gaussian energy function directly
            energies[i] += np.exp(-(distance**2) / (2 * sigma**2))

    return energies
