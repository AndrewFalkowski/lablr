import numpy as np
import matplotlib.pyplot as plt  # Assuming matplotlib is used for get_label_dims and plotting
from typing import List, Tuple, Optional, Sequence

# ==============================================================================
# 1. Core Physics/Energy Functions
# ==============================================================================


def gaussian_energy(
    distance: np.ndarray | float, mu: float = 0.0, sigma: float = 1.0
) -> np.ndarray | float:
    """
    Calculates energy based on a Gaussian function (unnormalized PDF form).
    Lower energy for distances close to mu.

    Args:
        distance: Distance(s) between entities.
        mu: The ideal distance (where energy is lowest/zero exponent).
        sigma: Controls the "width" of the interaction potential.

    Returns:
        Calculated energy/energies.
    """
    if sigma <= 1e-10:  # Avoid division by zero or precision issues
        # If sigma is tiny, energy is high unless distance is exactly mu.
        # Return a large value (or handle based on specific needs).
        # For simplicity, returning 0, assuming non-interaction.
        return np.zeros_like(distance) if isinstance(distance, np.ndarray) else 0.0

    # Using exp(- (d - mu)^2 / (2 * sigma^2)) form.
    # Note: The previous `gaussian_energy` was a normalized PDF.
    # This form exp(-x^2) is more common for potential energies.
    # Adjust if the PDF form was specifically intended.
    exponent = -((distance - mu) ** 2) / (2 * sigma**2)
    return np.exp(exponent)


# ==============================================================================
# 2. Geometry Calculation Helpers
# ==============================================================================


def _calculate_box_geometry(
    matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates box centers and boundaries from the matrix."""
    n = matrix.shape[0]
    if n == 0:
        return (np.empty((0, 2)), np.empty(0), np.empty(0), np.empty(0), np.empty(0))

    point_x = matrix[:, 0]
    point_y = matrix[:, 1]
    radius = matrix[:, 2]
    # IMPORTANT: Assume angle in column 3 is ALREADY IN RADIANS internally
    angle_rad = matrix[:, 3]
    widths = matrix[:, 4]
    heights = matrix[:, 5]

    centers_x = point_x + radius * np.cos(angle_rad)
    centers_y = point_y + radius * np.sin(angle_rad)
    centers = np.stack((centers_x, centers_y), axis=-1)  # Shape (n, 2)

    half_widths = widths / 2.0
    half_heights = heights / 2.0
    x_min = centers_x - half_widths
    x_max = centers_x + half_widths
    y_min = centers_y - half_heights
    y_max = centers_y + half_heights

    return centers, x_min, x_max, y_min, y_max


def _compute_point_box_distances(
    points: np.ndarray,
    box_x_min: np.ndarray,
    box_x_max: np.ndarray,
    box_y_min: np.ndarray,
    box_y_max: np.ndarray,
) -> np.ndarray:
    """
    Computes distances from points to nearest edges of multiple boxes.

    Args:
        points: Shape (n_pts, 2)
        box_x_min, box_x_max, box_y_min, box_y_max: Shape (n_boxes,)

    Returns:
        Distance matrix, shape (n_boxes, n_pts). dist[i, j] is distance
        from point j to box i.
    """
    n_boxes = box_x_min.shape[0]
    n_pts = points.shape[0]
    if n_boxes == 0 or n_pts == 0:
        return np.empty((n_boxes, n_pts))

    # Expand dims for broadcasting: boxes (n_boxes, 1), points (1, n_pts)
    px_j = points[:, 0][np.newaxis, :]  # Shape (1, n_pts)
    py_j = points[:, 1][np.newaxis, :]  # Shape (1, n_pts)

    x_min_i = box_x_min[:, np.newaxis]  # Shape (n_boxes, 1)
    x_max_i = box_x_max[:, np.newaxis]  # Shape (n_boxes, 1)
    y_min_i = box_y_min[:, np.newaxis]  # Shape (n_boxes, 1)
    y_max_i = box_y_max[:, np.newaxis]  # Shape (n_boxes, 1)

    # Calculate distance components (element [i,j] is for box i, point j)
    dx = np.maximum(0.0, np.maximum(x_min_i - px_j, px_j - x_max_i))
    dy = np.maximum(0.0, np.maximum(y_min_i - py_j, py_j - y_max_i))
    distances = np.sqrt(dx**2 + dy**2)  # Shape (n_boxes, n_pts)

    # Check which points are inside which boxes
    inside_x = (px_j >= x_min_i) & (px_j <= x_max_i)
    inside_y = (py_j >= y_min_i) & (py_j <= y_max_i)
    inside = inside_x & inside_y  # Shape (n_boxes, n_pts)

    distances[inside] = 0.0
    return distances


def _compute_box_box_distances(
    x_min: np.ndarray, x_max: np.ndarray, y_min: np.ndarray, y_max: np.ndarray
) -> np.ndarray:
    """
    Computes minimum distances between all pairs of boxes.

    Args:
        x_min, x_max, y_min, y_max: Box boundaries, shape (n,).

    Returns:
        Distance matrix, shape (n, n). dist[i, j] is distance
        between box i and box j. Diagonal is 0.
    """
    n = x_min.shape[0]
    if n <= 1:
        return np.zeros((n, n))

    # Expand dims for broadcasting: box i (n, 1), box j (1, n)
    x_min_i, x_max_i = x_min[:, np.newaxis], x_max[:, np.newaxis]
    y_min_i, y_max_i = y_min[:, np.newaxis], y_max[:, np.newaxis]
    x_min_j, x_max_j = x_min[np.newaxis, :], x_max[np.newaxis, :]
    y_min_j, y_max_j = y_min[np.newaxis, :], y_max[np.newaxis, :]

    # Calculate separation distance components dx, dy
    dx1 = x_min_i - x_max_j
    dx2 = x_min_j - x_max_i
    dx = np.maximum(0.0, np.maximum(dx1, dx2))  # Shape (n, n)

    dy1 = y_min_i - y_max_j
    dy2 = y_min_j - y_max_i
    dy = np.maximum(0.0, np.maximum(dy1, dy2))  # Shape (n, n)

    # Calculate Euclidean distance. Diagonal will automatically be 0.
    distances = np.sqrt(dx**2 + dy**2)
    return distances


# ==============================================================================
# 3. Energy Calculation Functions (Vectorized)
# ==============================================================================


def compute_point_box_energy_matrix(matrix: np.ndarray, sigma: float) -> np.ndarray:
    """
    Computes the energy matrix between all points and all boxes.

    Args:
        matrix: Configuration matrix (n, 6).
        sigma: Gaussian width parameter.

    Returns:
        Energy matrix (n, n), where E[i, j] is energy between box i and point j.
    """
    n = matrix.shape[0]
    if n == 0:
        return np.empty((0, 0))

    points = matrix[:, :2]  # Points are the first two columns
    _, x_min, x_max, y_min, y_max = _calculate_box_geometry(matrix)

    distances = _compute_point_box_distances(points, x_min, x_max, y_min, y_max)
    energy_matrix = gaussian_energy(distances, mu=0, sigma=sigma)
    return energy_matrix  # Shape (n_boxes, n_points) -> (n, n)


def compute_box_box_energy_matrix(
    matrix: np.ndarray, sigma: float, overlap_penalty: float = 100.0
) -> np.ndarray:
    """
    Computes the energy matrix between all pairs of boxes.

    Args:
        matrix: Configuration matrix (n, 6).
        sigma: Gaussian width parameter.

    Returns:
        Energy matrix (n, n), where E[i, j] is energy between box i and box j.
        Diagonal (self-energy) is zero.
    """
    n = matrix.shape[0]
    if n <= 1:
        return np.zeros((n, n))

    _, x_min, x_max, y_min, y_max = _calculate_box_geometry(matrix)
    distances = _compute_box_box_distances(x_min, x_max, y_min, y_max)
    energy_matrix = gaussian_energy(distances, mu=0, sigma=sigma)

    overlap_indices = np.isclose(distances, 0.0)
    energy_matrix[overlap_indices] = overlap_penalty

    # Exclude self-interaction energy
    np.fill_diagonal(energy_matrix, 0)
    return energy_matrix


def compute_total_energy(
    point_box_energy_matrix: np.ndarray, box_box_energy_matrix: np.ndarray
) -> float:
    """Computes the total scalar energy of the system."""
    # Sum point-box energies (sum over all points j for each box i, then sum over i)
    total_point_energy = np.sum(point_box_energy_matrix)
    # Sum box-box energies (already excludes self, sum over all pairs)
    total_box_energy = np.sum(box_box_energy_matrix)
    # Note: Summing the box_box_energy_matrix directly double counts pairs (Eij + Eji).
    # If energy is symmetric (Eij=Eji), divide by 2.
    # However, the original code summed the results of compute_box_to_box_energies,
    # which *itself* summed Eij over j for each i, so sum(results) *did* double count.
    # To match that, we sum the full matrix here. Adjust if needed.
    return total_point_energy + total_box_energy


# ==============================================================================
# 4. Label Dimension Calculation (Requires Matplotlib)
# ==============================================================================


def get_label_dimensions(
    ax: plt.Axes, labels: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the width and height of labels when rendered on an Axes.

    Args:
        ax: Matplotlib Axes object to use for rendering context.
        labels: List of label strings.

    Returns:
        Tuple of (widths, heights) as numpy arrays.
    """
    widths = np.zeros(len(labels))
    heights = np.zeros(len(labels))
    # Temporarily add text to axes to get bounding box in data coordinates
    for i, label in enumerate(labels):
        text_obj = ax.text(0, 0, label, ha="center", va="center")
        # Ensure graphics backend has processed the text object
        ax.figure.canvas.draw()  # Force draw (can be slow if done repeatedly)
        try:
            bbox = text_obj.get_window_extent(ax.figure.canvas.get_renderer())
            # Transform bounding box from display coordinates to data coordinates
            bbox_data = bbox.transformed(ax.transData.inverted())
            widths[i] = bbox_data.width
            heights[i] = bbox_data.height
        except Exception as e:
            print(f"Warning: Could not get extent for label '{label}'. Error: {e}")
            # Assign a default or estimate if needed
            widths[i] = 0.1  # Example default
            heights[i] = 0.1  # Example default
        finally:
            text_obj.remove()  # Clean up the temporary text

    return widths, heights


# ==============================================================================
# 5. Optimization Logic
# ==============================================================================

# Corrected optimize_label_placement function


def optimize_label_placement(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    labels: List[str],
    radii: Sequence[float] = (0.1, 0.2),
    angles_deg: Sequence[float] = (0, 45, 90, 135, 180, 225, 270, 315),
    n_initializations: int = 10,
    n_local_iterations: int = 5,
    sigma: float = 0.2,
    overlap_penalty: float = 10000.0,  # Keep parameter from previous fix
    label_padding_factor: float = 1.1,  # Keep parameter from previous fix
    exclude_self_point_energy: bool = False,
) -> Optional[np.ndarray]:
    """
    Optimizes label positions to minimize energy using random starts and local search.
    (Args documentation omitted for brevity - see previous versions)
    """
    n = len(x)
    if n == 0 or len(labels) != n:
        print("Error: Empty input or mismatch between points and labels.")
        return None

    # --- Initialization ---
    base_matrix = np.zeros((n, 6))
    base_matrix[:, 0] = x
    base_matrix[:, 1] = y
    angles_rad = np.deg2rad(angles_deg)
    print("Calculating label dimensions...")
    try:
        label_widths, label_heights = get_label_dimensions(
            ax, labels, padding_factor=label_padding_factor
        )
    except Exception as e:
        print(f"Error getting label dimensions: {e}. Check Matplotlib backend/Axes.")
        return None
    print("Label dimensions calculated.")

    best_total_energy = float("inf")
    best_config_matrix = None
    best_point_box_E = None
    best_box_box_E = None

    # --- Phase 1: Multiple Random Initializations ---
    print(f"Running {n_initializations} random initializations...")
    for init_iter in range(n_initializations):
        current_matrix = base_matrix.copy()
        current_matrix[:, 2] = np.random.choice(radii, size=n)
        current_matrix[:, 3] = np.random.choice(angles_rad, size=n)
        current_matrix[:, 4] = label_widths
        current_matrix[:, 5] = label_heights

        # --- FIX: Correct order for energy calculation ---
        # 1. Calculate point-box energy
        point_box_E = compute_point_box_energy_matrix(current_matrix, sigma)

        # 2. Optionally zero the diagonal (self point-box interaction)
        if exclude_self_point_energy:
            np.fill_diagonal(point_box_E, 0)  # Now point_box_E exists

        # 3. Calculate box-box energy (passing overlap penalty)
        box_box_E = compute_box_box_energy_matrix(
            current_matrix, sigma, overlap_penalty
        )  # Already excludes self-interaction diagonal internally

        # 4. Compute total scalar energy
        total_energy = compute_total_energy(point_box_E, box_box_E)
        # --- End of FIX ---

        if total_energy < best_total_energy:
            best_total_energy = total_energy
            best_config_matrix = current_matrix.copy()
            # Store energy matrices corresponding to the best config so far
            best_point_box_E = point_box_E.copy()
            best_box_box_E = box_box_E.copy()
            # Optional print:
            # print(f"  Init {init_iter+1}: New best energy = {best_total_energy:.4f}")

    if best_config_matrix is None:
        print("Warning: No valid initial configuration found.")
        return None

    # Ensure energy matrices are initialized if only one init was run and successful
    if best_point_box_E is None or best_box_box_E is None:
        point_box_E = compute_point_box_energy_matrix(best_config_matrix, sigma)
        if exclude_self_point_energy:
            np.fill_diagonal(point_box_E, 0)
        box_box_E = compute_box_box_energy_matrix(
            best_config_matrix, sigma, overlap_penalty
        )  # Pass penalty here too
        best_point_box_E = point_box_E.copy()
        best_box_box_E = box_box_E.copy()
        best_total_energy = compute_total_energy(best_point_box_E, best_box_box_E)

    print(f"Best energy after initialization: {best_total_energy:.4f}")

    # --- Phase 2: Local Refinement (Using Delta E) ---
    print(f"Running {n_local_iterations} local refinement iterations...")
    current_best_config = best_config_matrix
    current_total_energy = best_total_energy
    current_point_energies = np.sum(best_point_box_E, axis=1)
    current_box_energies = np.sum(best_box_box_E, axis=1)
    _, current_all_x_min, current_all_x_max, current_all_y_min, current_all_y_max = (
        _calculate_box_geometry(current_best_config)
    )

    for iter_num in range(n_local_iterations):
        changed = False
        random_order = np.random.permutation(n)
        all_x_min = current_all_x_min.copy()
        all_x_max = current_all_x_max.copy()
        all_y_min = current_all_y_min.copy()
        all_y_max = current_all_y_max.copy()

        for i in random_order:  # Iterate through each label index
            label_i_best_energy = current_total_energy
            best_radius_i = current_best_config[i, 2]
            best_angle_i = current_best_config[i, 3]
            found_better_for_i = False
            E_pt_i_old = current_point_energies[i]
            E_box_i_old = current_box_energies[i]
            E_box_exerted_by_i_old = np.sum(best_box_box_E[:, i])

            for radius in radii:
                for angle_rad in angles_rad:
                    if (
                        radius == current_best_config[i, 2]
                        and angle_rad == current_best_config[i, 3]
                    ):
                        continue

                    # --- Create Test State (Calculate new geometry for box i) ---
                    pt_x_i, pt_y_i = (
                        current_best_config[i, 0],
                        current_best_config[i, 1],
                    )
                    w_i, h_i = current_best_config[i, 4], current_best_config[i, 5]
                    center_x_i = pt_x_i + radius * np.cos(angle_rad)
                    center_y_i = pt_y_i + radius * np.sin(angle_rad)
                    x_min_i, x_max_i = center_x_i - w_i / 2, center_x_i + w_i / 2
                    y_min_i, y_max_i = center_y_i - h_i / 2, center_y_i + h_i / 2

                    # --- Calculate *NEW* energy components involving only label i ---
                    # 1. Point-Box Energy for Row i
                    points_j = current_best_config[:, :2]
                    dist_i_pts = _compute_point_box_distances(
                        points_j,
                        np.array([x_min_i]),
                        np.array([x_max_i]),
                        np.array([y_min_i]),
                        np.array([y_max_i]),
                    ).flatten()
                    E_pt_row_i_new = gaussian_energy(dist_i_pts, mu=0, sigma=sigma)
                    if exclude_self_point_energy:
                        E_pt_row_i_new[i] = 0
                    E_pt_i_new = np.sum(E_pt_row_i_new)

                    # 2. Box-Box Energy for Row i and Column i
                    all_x_min[i], all_x_max[i] = x_min_i, x_max_i
                    all_y_min[i], all_y_max[i] = y_min_i, y_max_i
                    full_trial_distance_matrix = _compute_box_box_distances(
                        all_x_min, all_x_max, all_y_min, all_y_max
                    )
                    dist_i_boxes = full_trial_distance_matrix[i, :]
                    dist_boxes_i = full_trial_distance_matrix[:, i]
                    E_box_row_i_new = gaussian_energy(dist_i_boxes, mu=0, sigma=sigma)
                    E_box_col_i_new = gaussian_energy(dist_boxes_i, mu=0, sigma=sigma)
                    overlap_row = np.isclose(dist_i_boxes, 0.0)
                    overlap_col = np.isclose(dist_boxes_i, 0.0)
                    E_box_row_i_new[overlap_row] = overlap_penalty
                    E_box_col_i_new[overlap_col] = overlap_penalty
                    E_box_row_i_new[i] = 0
                    E_box_col_i_new[i] = 0
                    E_box_i_new = np.sum(E_box_row_i_new)
                    E_box_exerted_by_i_new = np.sum(E_box_col_i_new)

                    # --- Calculate Delta E and New Total Energy ---
                    delta_E_pt = E_pt_i_new - E_pt_i_old
                    delta_E_box = (E_box_i_new - E_box_i_old) + (
                        E_box_exerted_by_i_new - E_box_exerted_by_i_old
                    )
                    delta_E_total = delta_E_pt + delta_E_box
                    trial_total_energy = current_total_energy + delta_E_total

                    # --- Check if this move is the best for label i so far ---
                    if trial_total_energy < label_i_best_energy:
                        label_i_best_energy = trial_total_energy
                        best_radius_i = radius
                        best_angle_i = angle_rad
                        best_E_pt_row_i = E_pt_row_i_new
                        best_E_box_row_i = E_box_row_i_new
                        best_E_box_col_i = E_box_col_i_new
                        found_better_for_i = True

                    # --- Restore boundary arrays for the next trial angle/radius ---
                    all_x_min[i] = current_all_x_min[i]
                    all_x_max[i] = current_all_x_max[i]
                    all_y_min[i] = current_all_y_min[i]
                    all_y_max[i] = current_all_y_max[i]

            # --- Update label i in main config if a better position was found ---
            if found_better_for_i and label_i_best_energy < current_total_energy:
                current_best_config[i, 2] = best_radius_i
                current_best_config[i, 3] = best_angle_i
                current_total_energy = label_i_best_energy
                current_point_energies[i] = np.sum(best_E_pt_row_i)
                current_box_energies[i] = np.sum(best_E_box_row_i)
                best_point_box_E[i, :] = best_E_pt_row_i
                best_box_box_E[i, :] = best_E_box_row_i
                best_box_box_E[:, i] = best_E_box_col_i

                # Update the 'current' full geometry arrays
                pt_x_i, pt_y_i = current_best_config[i, 0], current_best_config[i, 1]
                w_i, h_i = current_best_config[i, 4], current_best_config[i, 5]
                center_x_i = pt_x_i + best_radius_i * np.cos(best_angle_i)
                center_y_i = pt_y_i + best_radius_i * np.sin(best_angle_i)
                current_all_x_min[i], current_all_x_max[i] = (
                    center_x_i - w_i / 2,
                    center_x_i + w_i / 2,
                )
                current_all_y_min[i], current_all_y_max[i] = (
                    center_y_i - h_i / 2,
                    center_y_i + h_i / 2,
                )
                all_x_min[i] = current_all_x_min[i]
                all_x_max[i] = current_all_x_max[i]
                all_y_min[i] = current_all_y_min[i]
                all_y_max[i] = current_all_y_max[i]

                changed = True

        if not changed:
            print(f"  Local refinement converged after iteration {iter_num + 1}.")
            break

    print(f"Final energy after refinement: {current_total_energy:.4f}")
    return current_best_config


# ==============================================================================
# 6. Plotting Function
# ==============================================================================


def plot_labels(ax: plt.Axes, final_matrix: np.ndarray, labels: List[str]):
    """
    Plots the final label configuration.

    Args:
        ax: Matplotlib Axes to plot on.
        final_matrix: The optimized configuration matrix (N, 6).
        labels: List of label strings.
    """
    if final_matrix is None:
        print("No configuration to plot.")
        return

    for i in range(final_matrix.shape[0]):
        point_x, point_y, radius, angle_rad, width, height = final_matrix[i]
        label_text = labels[i]

        # Calculate final label box center
        box_center_x = point_x + radius * np.cos(angle_rad)
        box_center_y = point_y + radius * np.sin(angle_rad)

        # Add label text
        ax.text(
            box_center_x,
            box_center_y,
            label_text,
            ha="center",
            va="center",
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8),
        )  # Add background box

        # Add connecting line
        ax.plot(
            [point_x, box_center_x],
            [point_y, box_center_y],
            "k-",
            linewidth=0.75,
            alpha=0.7,
        )
