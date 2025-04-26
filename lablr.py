import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
from numpy.typing import ArrayLike, NDArray
from energy import *
from tqdm import tqdm, trange


def optimize_label_positions(
    ax,
    anchors,
    labels,
    allowed_radii,
    allowed_angles,
    n_random_iterations=10,
    n_local_iterations=5,
):
    anchors = np.asarray(anchors)
    n_labels = len(labels)

    # Pre-compute influence extent once
    influence_extent = np.mean(allowed_radii)

    # Phase 1: Random initialization (keep this part the same)
    lowest_energy = np.inf
    best_label_pos = None

    for _ in trange(n_random_iterations, desc="Random Iterations"):
        label_pos = gen_random_label_pos(anchors, allowed_radii, allowed_angles)

        point_energies = calc_label_point_energies(
            anchors, label_pos, influence_extent=influence_extent
        )
        label_energies = calc_label_label_energies(
            label_pos, influence_extent=influence_extent
        )
        crossing_energies = calc_crossing_energies(anchors, label_pos)
        overlap_energies = calc_overlap_energies(anchors, label_pos, labels, ax=ax)

        energies = (
            point_energies + label_energies + crossing_energies + overlap_energies
        )
        total_energy = np.sum(energies)

        if total_energy < lowest_energy:
            lowest_energy = total_energy
            best_label_pos = label_pos

    # Phase 2: Iterative local optimization with caching
    if best_label_pos is None and n_labels > 0:
        best_label_pos = gen_random_label_pos(anchors, allowed_radii, allowed_angles)
        lowest_energy = calculate_total_energy(
            anchors, best_label_pos, labels, ax, allowed_radii
        )
    elif n_labels == 0:
        return np.array([]), np.array([]), 0.0

    # Pre-calculate energies that will be updated
    current_label_pos = best_label_pos.copy()

    # Pre-compute all possible positions for each label
    all_positions = {}
    for i in range(n_labels):
        positions = []
        for radius in allowed_radii:
            for angle in allowed_angles:
                pos = get_label_coords(anchors[i : i + 1], [radius], [angle])[0]
                positions.append(pos)
        all_positions[i] = np.array(positions)

    for _ in trange(n_local_iterations, desc="Local Iterations"):
        random_order = np.random.permutation(n_labels)

        for i in random_order:
            current_energy = lowest_energy
            best_position_idx = None

            # Store original position
            original_pos = current_label_pos[i].copy()

            # Try all pre-computed positions for label i
            for idx, new_pos in enumerate(all_positions[i]):
                current_label_pos[i] = new_pos

                # Calculate only the energy components that change
                energy = calculate_total_energy(
                    anchors, current_label_pos, labels, ax, allowed_radii
                )

                if energy < current_energy:
                    current_energy = energy
                    best_position_idx = idx

            # Update to best position found
            if best_position_idx is not None:
                current_label_pos[i] = all_positions[i][best_position_idx]
                lowest_energy = current_energy
            else:
                current_label_pos[i] = (
                    original_pos  # Restore original if no improvement
                )

    best_label_pos = current_label_pos

    # Calculate final energies
    point_energies = calc_label_point_energies(
        anchors, best_label_pos, influence_extent=influence_extent
    )
    label_energies = calc_label_label_energies(
        best_label_pos, influence_extent=influence_extent
    )
    crossing_energies = calc_crossing_energies(anchors, best_label_pos)
    overlap_energies = calc_overlap_energies(anchors, best_label_pos, labels, ax=ax)

    best_label_energies = (
        point_energies + label_energies + crossing_energies + overlap_energies
    )

    return best_label_pos, best_label_energies, lowest_energy


def get_label_coords(
    anchors: ArrayLike, radii: ArrayLike, angles: ArrayLike
) -> NDArray[np.float64]:
    """Calculate label positions using polar coordinate transformations from anchor points.

    Computes the Cartesian coordinates of labels positioned at specified radii and angles
    from given anchor points. Automatically converts angles from degrees to radians if
    they exceed 2π.

    Args:
        anchors: Array of shape (..., 2) containing anchor points in Cartesian coordinates.
                Each row should contain [x, y] coordinates.
        radii: Array of radii values, must be broadcastable with anchors.
        angles: Array of angles in radians (converted automatically if in degrees).
               Must be broadcastable with anchors.

    Returns:
        Array of shape (..., 2) containing the calculated label coordinates,
        where each row contains [x, y] coordinates of a label position.
    """
    # ensure anchors, radii, and angles are numpy arrays
    anchors = np.asarray(anchors)
    radii = np.asarray(radii)
    angles = np.asarray(angles)

    # convert angles to radians if they are in degrees
    if np.max(angles) > 2 * np.pi:
        angles = np.radians(angles)

    # split anchors into x0, y0
    x0 = anchors[..., 0]
    y0 = anchors[..., 1]

    # compute
    x = x0 + radii * np.cos(angles)
    y = y0 + radii * np.sin(angles)

    # stack into (..., 2)
    return np.stack((x, y), axis=-1)


def gen_random_label_pos(
    anchors: ArrayLike, allowed_radii: ArrayLike, allowed_angles: ArrayLike
) -> NDArray[np.float64]:
    """Generate random label positions by sampling from allowed radii and angles.

    Randomly selects radius and angle values from the provided allowed values for each
    anchor point, then computes the corresponding label positions using polar coordinate
    transformations.

    Args:
        anchors: Array of shape (n, 2) containing anchor points in Cartesian coordinates.
                Each row should contain [x, y] coordinates.
        allowed_radii: 1D array of allowed radius values to randomly sample from.
        allowed_angles: 1D array of allowed angle values in radians to randomly sample from.

    Returns:
        Array of shape (n, 2) containing randomly generated label coordinates,
        where each row contains [x, y] coordinates of a label position.
    """
    # sample random radius and angle
    radii = np.random.choice(allowed_radii, size=anchors.shape[0])
    angles = np.random.choice(allowed_angles, size=anchors.shape[0])

    label_pos = get_label_coords(anchors, radii, angles)

    return label_pos
