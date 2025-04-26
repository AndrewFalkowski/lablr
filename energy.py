import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
from numpy.typing import ArrayLike, NDArray


def calculate_total_energy(anchors, label_pos, labels, ax, allowed_radii):
    """Helper function to calculate total energy for a given label configuration."""
    point_energies = calc_label_point_energies(
        anchors, label_pos, influence_extent=np.mean(allowed_radii)
    )
    label_energies = calc_label_label_energies(
        label_pos, influence_extent=np.mean(allowed_radii)
    )
    crossing_energies = calc_crossing_energies(anchors, label_pos)
    overlap_energies = calc_overlap_energies(anchors, label_pos, labels, ax=ax)

    total_energy = np.sum(
        point_energies + label_energies + crossing_energies + overlap_energies
    )
    return total_energy


def calc_label_point_energies(
    anchors: ArrayLike, label_pos: ArrayLike, influence_extent: float
) -> NDArray[np.float64]:
    """Calculate total energy for each label position based on all anchor points.

    For each label position, computes the sum of energies from all anchor points
    using the energy function which applies a power law decay based on distance.

    Args:
        anchors: Array of shape (n_anchors, 2) containing anchor point coordinates.
        label_pos: Array of shape (n_labels, 2) containing label position coordinates.

    Returns:
        Array of shape (n_labels,) containing total energy values, where element
        i represents the total energy for label position i from all anchors.
    """
    # for each label point, compute the distances to all anchor points, and apply the energy function
    distances = np.linalg.norm(anchors[:, None, :] - label_pos[None, :, :], axis=-1)
    energies = energy_func(distances, influence_extent)

    # sum energies for each label point
    energies = np.sum(energies, axis=0)  # Changed axis from 1 to 0

    return energies


def calc_label_label_energies(
    label_pos: ArrayLike, influence_extent: float
) -> NDArray[np.float64]:
    """Calculate energy for each label position based on distances to other labels.

    Computes the energy for each label position based on the distances to all other
    label positions using the energy function which applies a power law decay based on
    distance.

    Args:
        label_pos: Array of shape (n_labels, 2) containing label position coordinates.

    Returns:
        Array of shape (n_labels,) containing energy values for each label position,
        where element i represents the energy for label position i from all other labels.
    """
    # for each label point, compute the distances to all other label points, and apply the energy function
    distances = np.linalg.norm(label_pos[:, None, :] - label_pos[None, :, :], axis=-1)
    energies = energy_func(distances, influence_extent)

    # sum energies for each label point
    energies = np.sum(energies, axis=0)  # Changed axis from 1 to 0

    return energies


def calc_crossing_energies(
    anchors: ArrayLike, label_pos: ArrayLike
) -> NDArray[np.float64]:
    """Calculate energy based on whether the line between anchor and label crosses another line.

    For each anchor-label pair, check if the line segment intersects with any other
    anchor-label line segment. If there's an intersection, assign energy = 1, otherwise 0.

    Args:
        anchors: Array of shape (n, 2) containing anchor point coordinates.
        label_pos: Array of shape (n, 2) containing label position coordinates.

    Returns:
        Array of shape (n,) containing energy values (0 or 1) for each label position.
    """
    anchors = np.asarray(anchors)
    label_pos = np.asarray(label_pos)
    n_labels = len(label_pos)

    # Initialize energy array
    energies = np.zeros(n_labels)

    # Pre-compute bounding boxes for each segment
    min_x = np.minimum(anchors[:, 0], label_pos[:, 0])
    max_x = np.maximum(anchors[:, 0], label_pos[:, 0])
    min_y = np.minimum(anchors[:, 1], label_pos[:, 1])
    max_y = np.maximum(anchors[:, 1], label_pos[:, 1])

    # Helper function to compute orientation of triplet (p, q, r)
    def orientation(p, q, r):
        """
        Returns:
        0: Collinear
        1: Clockwise
        2: Counterclockwise
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-10:  # Using small epsilon for floating point comparison
            return 0
        return 1 if val > 0 else 2

    # Helper function to check if point q lies on segment pr
    def on_segment(p, q, r):
        """Check if point q lies on line segment pr"""
        return (
            q[0] <= max(p[0], r[0])
            and q[0] >= min(p[0], r[0])
            and q[1] <= max(p[1], r[1])
            and q[1] >= min(p[1], r[1])
        )

    # Check each pair of segments for intersection
    for i in range(n_labels):
        if energies[i] == 1:  # Skip if we already found an intersection for this label
            continue

        for j in range(n_labels):
            if i == j:
                continue

            # Quick bounding box check to skip obvious non-intersections
            if (
                max_x[i] < min_x[j]
                or min_x[i] > max_x[j]
                or max_y[i] < min_y[j]
                or min_y[i] > max_y[j]
            ):
                continue

            # Define segments
            p1, q1 = anchors[i], label_pos[i]
            p2, q2 = anchors[j], label_pos[j]

            # Find orientations
            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)

            # General case
            if o1 != o2 and o3 != o4:
                energies[i] = 10
                break  # No need to check more segments once we find an intersection

            # Special cases for collinear segments
            # p1, q1 and p2 are collinear and p2 lies on segment p1q1
            if o1 == 0 and on_segment(p1, p2, q1):
                energies[i] = 10
                break

            # p1, q1 and q2 are collinear and q2 lies on segment p1q1
            if o2 == 0 and on_segment(p1, q2, q1):
                energies[i] = 10
                break

            # p2, q2 and p1 are collinear and p1 lies on segment p2q2
            if o3 == 0 and on_segment(p2, p1, q2):
                energies[i] = 10
                break

            # p2, q2 and q1 are collinear and q1 lies on segment p2q2
            if o4 == 0 and on_segment(p2, q1, q2):
                energies[i] = 10
                break

    return energies


def calc_crossing_energies_old(
    anchors: ArrayLike, label_pos: ArrayLike
) -> NDArray[np.float64]:
    """Calculate energy based on whether the line between anchor and label crosses another line.

    For each anchor-label pair, check if the line segment intersects with any other
    anchor-label line segment. If there's an intersection, assign energy = 1, otherwise 0.

    Args:
        anchors: Array of shape (n, 2) containing anchor point coordinates.
        label_pos: Array of shape (n, 2) containing label position coordinates.

    Returns:
        Array of shape (n,) containing energy values (0 or 1) for each label position.
    """
    anchors = np.asarray(anchors)
    label_pos = np.asarray(label_pos)
    n_labels = len(label_pos)

    # Initialize energy array
    energies = np.zeros(n_labels)

    # Helper function to compute orientation of triplet (p, q, r)
    def orientation(p, q, r):
        """
        Returns:
        0: Collinear
        1: Clockwise
        2: Counterclockwise
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-10:  # Using small epsilon for floating point comparison
            return 0
        return 1 if val > 0 else 2

    # Helper function to check if point q lies on segment pr
    def on_segment(p, q, r):
        """Check if point q lies on line segment pr"""
        return (
            q[0] <= max(p[0], r[0])
            and q[0] >= min(p[0], r[0])
            and q[1] <= max(p[1], r[1])
            and q[1] >= min(p[1], r[1])
        )

    # Check each pair of segments for intersection
    for i in range(n_labels):
        for j in range(n_labels):
            if i == j:
                continue

            # Define segments
            p1, q1 = anchors[i], label_pos[i]
            p2, q2 = anchors[j], label_pos[j]

            # Find orientations
            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)

            # General case
            if o1 != o2 and o3 != o4:
                energies[i] = 1
                break  # No need to check more segments once we find an intersection

            # Special cases for collinear segments
            # p1, q1 and p2 are collinear and p2 lies on segment p1q1
            if o1 == 0 and on_segment(p1, p2, q1):
                energies[i] = 1
                break

            # p1, q1 and q2 are collinear and q2 lies on segment p1q1
            if o2 == 0 and on_segment(p1, q2, q1):
                energies[i] = 1
                break

            # p2, q2 and p1 are collinear and p1 lies on segment p2q2
            if o3 == 0 and on_segment(p2, p1, q2):
                energies[i] = 1
                break

            # p2, q2 and q1 are collinear and q1 lies on segment p2q2
            if o4 == 0 and on_segment(p2, q1, q2):
                energies[i] = 1
                break

    return energies


def calc_overlap_energies(
    anchors: ArrayLike, label_pos: ArrayLike, labels: List[str], ax: plt.Axes
) -> NDArray[np.float64]:
    """Calculate energy based on whether text labels overlap with other geometry."""
    anchors = np.asarray(anchors)
    label_pos = np.asarray(label_pos)
    n_labels = len(labels)
    energies = np.zeros(n_labels)

    # Ensure up-to-date figure state
    ax.figure.canvas.draw_idle()

    # Use a more precise renderer
    renderer = ax.figure.canvas.get_renderer()

    # Get bounding boxes with proper renderer
    bboxes = []
    for i, (pos, label) in enumerate(zip(label_pos, labels)):
        # Create text object without drawing it
        text_obj = ax.text(
            pos[0],
            pos[1],
            label,
            ha="center",
            va="center",
            bbox=dict(boxstyle="square,pad=0.3", alpha=0),
            zorder=999,  # Ensure text is on top
        )

        # Get precise bounding box
        bbox = text_obj.get_window_extent(renderer=renderer).transformed(
            ax.transData.inverted()
        )

        bboxes.append(bbox)
        text_obj.remove()

    # Check overlaps with better precision
    for i in range(n_labels):
        # Check label-label overlaps
        for j in range(n_labels):
            if i != j and bboxes[i].overlaps(bboxes[j]):
                energies[i] = 10
                break

        # Check label-anchor overlaps with buffer
        for anchor in anchors:
            # Expand bbox slightly for better detection
            expanded_bbox = bboxes[i].expanded(1.1, 1.1)
            if expanded_bbox.contains(anchor[0], anchor[1]):
                energies[i] = 10
                break

        # Check label-line overlaps with improved algorithm
        for j in range(n_labels):
            if i != j:
                # Add more sampling points along the line
                num_samples = 10
                t = np.linspace(0, 1, num_samples)
                line_points = anchors[j] + t[:, None] * (label_pos[j] - anchors[j])

                for point in line_points:
                    if bboxes[i].contains(point[0], point[1]):
                        energies[i] = 10
                        break
                if energies[i] == 10:
                    break

    return energies


def calc_overlap_energies_old(
    anchors: ArrayLike, label_pos: ArrayLike, labels: List[str], ax: plt.Axes
) -> NDArray[np.float64]:
    """Calculate energy based on whether text labels overlap with other geometry.

    This function checks if text labels overlap with:
    1. Other labels
    2. Lines connecting anchors to labels
    3. Anchor points

    Args:
        anchors: Array of shape (n, 2) containing anchor points.
        label_pos: Array of shape (n, 2) containing label positions.
        labels: List of label strings.
        ax: Matplotlib Axes object for text rendering.

    Returns:
        Array of shape (n,) containing energy values. Energy = 1 if label overlaps
        with anything, 0 otherwise.
    """
    anchors = np.asarray(anchors)
    label_pos = np.asarray(label_pos)
    n_labels = len(labels)
    energies = np.zeros(n_labels)

    # Cache the bounding boxes for each label text
    # This is done once per function call instead of creating text objects each time
    if not hasattr(ax, "_label_bbox_cache"):
        ax._label_bbox_cache = {}

    bboxes = []
    for i, (pos, label) in enumerate(zip(label_pos, labels)):
        # Create a unique key for each label position
        cache_key = (label, pos[0], pos[1])

        if cache_key not in ax._label_bbox_cache:
            # Only create text object if not in cache
            text_obj = ax.text(
                pos[0],
                pos[1],
                label,
                ha="center",
                va="center",
                bbox=dict(boxstyle="square,pad=0.5", alpha=0),
            )
            # Force draw only when necessary
            ax.figure.canvas.draw()
            bbox = text_obj.get_window_extent().transformed(ax.transData.inverted())
            ax._label_bbox_cache[cache_key] = bbox
            text_obj.remove()

        bboxes.append(ax._label_bbox_cache[cache_key])

    # Check for overlaps
    for i in range(n_labels):
        # Check label-label overlaps
        for j in range(n_labels):
            if i != j and bboxes[i].overlaps(bboxes[j]):
                energies[i] = 10
                break

        # Check if label overlaps with any anchor points
        for anchor in anchors:
            if bboxes[i].contains(anchor[0], anchor[1]):
                energies[i] = 10
                break

        # Check if label overlaps with any connecting lines
        for j in range(n_labels):
            if i != j:
                # Simple approximation: check if line segment intersects bbox
                if _line_bbox_intersect(anchors[j], label_pos[j], bboxes[i]):
                    energies[i] = 10
                    break

    return energies


def _line_bbox_intersect(p1: NDArray, p2: NDArray, bbox) -> bool:
    """Check if a line segment intersects with a bounding box.

    Args:
        p1, p2: Endpoints of line segment
        bbox: Matplotlib Bbox object

    Returns:
        True if intersection exists
    """
    # Simple approach: check if either endpoint is inside bbox
    if bbox.contains(p1[0], p1[1]) or bbox.contains(p2[0], p2[1]):
        return True

    # Check if line crosses any of the four bbox edges
    corners = [
        (bbox.x0, bbox.y0),
        (bbox.x1, bbox.y0),
        (bbox.x1, bbox.y1),
        (bbox.x0, bbox.y1),
    ]

    for i in range(4):
        c1 = corners[i]
        c2 = corners[(i + 1) % 4]
        if _line_segments_intersect(p1, p2, np.array(c1), np.array(c2)):
            return True

    return False


def _line_segments_intersect(
    p1: NDArray, q1: NDArray, p2: NDArray, q2: NDArray
) -> bool:
    """Check if two line segments intersect."""

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else 2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    return False


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
