import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def optimize_box_positions(
    ax,
    points,
    labels,
    sigma_label=0.1,
    sigma_point=0.15,
    max_distance=0.3,
    n_steps=50,
    step_size=0.01,
    text_kwargs=None,
    bbox_kwargs=None,
):

    label_dims = get_label_dimensions(ax, labels, text_kwargs, bbox_kwargs)

    positions_history, energy_history = iterative_repulsion(
        points,
        label_dims,
        n_steps=n_steps,
        step_size=step_size,
        max_distance=max_distance,
        sigma_box=sigma_label,
        sigma_fixed=sigma_point,
    )

    # Extract final positions
    final_positions = positions_history[-1]

    for i, (x, y, w, h) in enumerate(final_positions):
        ax.text(x, y, labels[i], **text_kwargs, bbox=bbox_kwargs)

    return positions_history, energy_history


def get_label_dimensions(ax, labels, text_kwargs=None, bbox_kwargs=None):
    """
    Pre-compute width and height for all labels.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add the temporary text elements to
    labels : list of str
        List of strings representing labels
    text_kwargs : dict, optional
        Keyword arguments for Text creation (e.g., fontsize, fontfamily, fontweight)
        Default is None, which uses matplotlib defaults
    bbox_kwargs : dict, optional
        Keyword arguments for the text bbox (e.g., boxstyle, pad, edgecolor, facecolor)
        Default is None, which means no bbox

    Returns:
    --------
    numpy.ndarray
        Array of shape (n_labels, 2) with [width, height] for each label
    """

    # Ensure the figure is drawn first
    fig = ax.get_figure()
    fig.canvas.draw()

    label_dims = np.zeros((len(labels), 2), dtype=np.float64)

    # Set default text kwargs if none provided
    if text_kwargs is None:
        text_kwargs = {}

    # Default ha and va if not specified
    if "ha" not in text_kwargs:
        text_kwargs["ha"] = "center"
    if "va" not in text_kwargs:
        text_kwargs["va"] = "center"

    for i, label in enumerate(labels):
        # Create text with specified kwargs
        if bbox_kwargs is not None:
            # Create with bbox
            text = ax.text(0, 0, label, bbox=dict(bbox_kwargs), **text_kwargs)
        else:
            # Create without bbox
            text = ax.text(0, 0, label, **text_kwargs)

        bbox = text.get_window_extent().transformed(ax.transData.inverted())
        text.remove()

        label_dims[i, 0] = bbox.width
        label_dims[i, 1] = bbox.height

    return label_dims


def find_penetration(box, dx, dy):
    """
    Find how far a ray from box center in direction (dx, dy) penetrates the box.
    Returns distance from center to box boundary along the ray.
    """
    _, _, w, h = box

    if dx == 0 and dy == 0:
        return 0

    # Half dimensions
    half_w = w / 2
    half_h = h / 2

    # Find where the ray intersects the box boundary
    t_values = []

    if dx != 0:
        # Check left and right sides
        t_right = half_w / dx if dx > 0 else float("inf")
        t_left = -half_w / dx if dx < 0 else float("inf")
        t_values.extend([t_right, t_left])

    if dy != 0:
        # Check top and bottom sides
        t_top = half_h / dy if dy > 0 else float("inf")
        t_bottom = -half_h / dy if dy < 0 else float("inf")
        t_values.extend([t_top, t_bottom])

    # Find the smallest positive t value
    valid_t_values = [t for t in t_values if t > 0]
    if not valid_t_values:
        return 0

    t_min = min(valid_t_values)

    # Calculate the actual distance using the parameter
    return t_min * np.sqrt(dx**2 + dy**2)


def compute_box_distance(box1, box2):
    """
    Compute distance between two boxes along the center-to-center line.
    Accounts for box dimensions by subtracting penetration distances.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Center-to-center vector
    dx = x2 - x1
    dy = y2 - y1
    center_distance = np.sqrt(dx**2 + dy**2)

    if center_distance == 0:
        return 0

    # Find intersection points on both boxes
    penetration1 = find_penetration(box1, dx, dy)
    penetration2 = find_penetration(box2, -dx, -dy)

    # Effective distance is center distance minus penetrations
    effective_distance = center_distance - penetration1 - penetration2

    # If boxes overlap, return 0
    return max(0, effective_distance)


def gaussian_repulsion(distance, sigma=1.0):
    """Calculate repulsion energy using Gaussian PDF."""
    # return 1 / (1+1e-9)**4
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (distance / sigma) ** 2)


def compute_repulsion_vectors(
    movable_boxes, fixed_points, sigma_box=1.0, sigma_fixed=1.0
):
    """Compute the net repulsion vector for each movable box with separate sigmas."""
    n_movable = len(movable_boxes)
    n_fixed = len(fixed_points)
    repulsion_vectors = np.zeros((n_movable, 2))

    # Repulsion from other movable boxes (using sigma_box)
    for i in range(n_movable):
        for j in range(n_movable):
            if i != j:
                box1 = movable_boxes[i]
                box2 = movable_boxes[j]

                # Center-to-center vector
                dx = box2[0] - box1[0]
                dy = box2[1] - box1[1]
                center_distance = np.sqrt(dx**2 + dy**2)

                if center_distance > 0:
                    # Compute effective distance
                    distance = compute_box_distance(box1, box2)
                    # Clamp the distance to a minimum positive value
                    distance = max(distance, 1e-3)

                    if distance > 0:
                        direction = np.array([dx, dy]) / center_distance
                        repulsion = gaussian_repulsion(distance, sigma_box)
                        repulsion_vectors[i] -= direction * repulsion

    # Repulsion from fixed points (using sigma_fixed)
    for i in range(n_movable):
        for j in range(n_fixed):
            box = movable_boxes[i]
            # Convert fixed point to tiny box format
            fixed_box = [fixed_points[j, 0], fixed_points[j, 1], 1e-9, 1e-9]

            # Center-to-center vector
            dx = fixed_box[0] - box[0]
            dy = fixed_box[1] - box[1]
            center_distance = np.sqrt(dx**2 + dy**2)

            if center_distance > 0:
                # Compute effective distance
                distance = compute_box_distance(box, fixed_box)
                # Clamp the distance to a minimum positive value
                distance = max(distance, 1e-3)

                if distance > 0:
                    direction = np.array([dx, dy]) / center_distance
                    repulsion = gaussian_repulsion(distance, sigma_fixed)
                    repulsion_vectors[i] -= direction * repulsion

    return repulsion_vectors


def compute_total_energy(movable_boxes, fixed_points, sigma_box=1.0, sigma_fixed=1.0):
    """Compute total repulsion energy with separate sigmas."""
    total_energy = 0.0
    n_movable = len(movable_boxes)
    n_fixed = len(fixed_points)

    # Energy between movable boxes
    for i in range(n_movable):
        for j in range(i + 1, n_movable):
            distance = compute_box_distance(movable_boxes[i], movable_boxes[j])
            total_energy += gaussian_repulsion(
                distance, sigma_box
            ) / gaussian_repulsion(0, sigma_box)

    # Energy between movable boxes and fixed points
    for i in range(n_movable):
        for j in range(n_fixed):
            fixed_box = [fixed_points[j, 0], fixed_points[j, 1], 1e-9, 1e-9]
            distance = compute_box_distance(movable_boxes[i], fixed_box)
            total_energy += gaussian_repulsion(
                distance, sigma_fixed
            ) / gaussian_repulsion(0, sigma_fixed)

    return total_energy


def iterative_repulsion(
    starting_points,
    box_sizes,
    n_steps=10,
    step_size=0.1,
    max_distance=0.3,
    sigma_box=1.0,
    sigma_fixed=1.0,
):
    """Iterate repulsion with separate sigmas for boxes and fixed points."""
    fixed_points = starting_points.copy()  # These remain as points

    # Create boxes from starting points
    movable_boxes = []
    for i in range(len(starting_points)):
        # add jitter to box positions
        x, y = starting_points[i] + np.random.uniform(-1e-6, 1e-6, 2)
        w, h = box_sizes[i] if i < len(box_sizes) else (0.1, 0.1)
        movable_boxes.append([x, y, w, h])

    original_boxes = [box.copy() for box in movable_boxes]

    positions_history = [[box.copy() for box in movable_boxes]]
    energy_history = [
        compute_total_energy(movable_boxes, fixed_points, sigma_box, sigma_fixed)
    ]

    for _ in range(n_steps):
        # Compute repulsion vectors with separate sigmas
        repulsion_vectors = compute_repulsion_vectors(
            movable_boxes, fixed_points, sigma_box, sigma_fixed
        )

        # Update positions (only centers)
        new_boxes = []
        for i, box in enumerate(movable_boxes):
            new_box = box.copy()
            new_box[0] += step_size * repulsion_vectors[i, 0]
            new_box[1] += step_size * repulsion_vectors[i, 1]
            new_boxes.append(new_box)

        # Apply constraint relative to original positions
        new_boxes = apply_original_position_constraint(
            new_boxes, original_boxes, max_distance
        )

        # Compute energy
        energy = compute_total_energy(new_boxes, fixed_points, sigma_box, sigma_fixed)

        # Store history
        positions_history.append([box.copy() for box in new_boxes])
        energy_history.append(energy)
        movable_boxes = new_boxes

        # Check if the last 10 energy values are within x% of each other
        # if len(energy_history) > 20:
        #     last_10_energies = energy_history[-20:]
        #     mean_energy = np.mean(last_10_energies)

        #     # Calculate percentage difference from mean
        #     percent_threshold = 0.005  # Adjust this value for x%

        #     if np.all(
        #         np.abs(last_10_energies - mean_energy)
        #         <= mean_energy * percent_threshold
        #     ):
        #         print(
        #             f"Energy values have been within {percent_threshold*100}% of each other for the last 10 iterations, stopping early."
        #         )
        #         break

    return positions_history, energy_history


def apply_original_position_constraint(boxes, original_boxes, max_distance=0.3):
    """Constrain box centers to stay within max_distance from original centers."""
    constrained_boxes = []

    for i in range(len(boxes)):
        box = boxes[i].copy()
        original_box = original_boxes[i]

        # Only constrain center position, not dimensions
        displacement = np.array([box[0] - original_box[0], box[1] - original_box[1]])
        distance_from_original = np.linalg.norm(displacement)

        if distance_from_original > max_distance:
            new_center = np.array([original_box[0], original_box[1]]) + displacement * (
                max_distance / distance_from_original
            )
            box[0] = new_center[0]
            box[1] = new_center[1]

        constrained_boxes.append(box)

    return constrained_boxes
