"""Smooth pairwise potentials over signed distance (in points)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Potential:
    """C1-smooth pairwise potential over signed distance ``d`` (in points).

    Parameters
    ----------
    target : float
        Signed distance (pt) where energy reaches its minimum (zero).
    tolerance : float
        Width of the potential well (pt). Larger values make the well softer.
    weight : float, default 1.0
        Relative energy weight of the non-overlapping well term.
    attractive : bool, default True
        ``True`` gives a two-sided well centered at ``target`` (for a label's
        own anchor). ``False`` gives a one-sided barrier that is zero for all
        ``d >= target`` (for repelling other labels, points, lines, and walls).
    overlap_stiffness : float, default 30.0
        Scale-invariant stiffness (``energy / pt²``) applied smoothly whenever
        ``d < 0`` (physical clipping). Independent of ``tolerance``.
    """

    target: float
    tolerance: float
    weight: float = 1.0
    attractive: bool = True
    overlap_stiffness: float = 30.0

    def __post_init__(self) -> None:
        if self.tolerance <= 0.0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")
        if self.weight < 0.0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")
        if self.overlap_stiffness < 0.0:
            raise ValueError(
                f"overlap_stiffness must be non-negative, got {self.overlap_stiffness}"
            )

    @classmethod
    def anchor(
        cls,
        distance: float = 8.0,
        tolerance: float | None = None,
        weight: float = 1.0,
    ) -> Potential:
        """Two-sided well tethering a label to its own anchor point."""
        tol = max(distance, 4.0) if tolerance is None else tolerance
        return cls(target=distance, tolerance=tol, weight=weight, attractive=True)

    @classmethod
    def repel(
        cls,
        padding: float = 4.0,
        tolerance: float | None = None,
        weight: float = 2.0,
    ) -> Potential:
        """One-sided barrier keeping objects separated by at least ``padding``."""
        tol = max(0.5 * padding, 1.5) if tolerance is None else tolerance
        return cls(target=padding, tolerance=tol, weight=weight, attractive=False)

    def evaluate(self, distance: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return ``(energy, dE/dd)``; non-finite inputs yield zero."""
        d = np.asarray(distance, dtype=np.float64)
        finite = np.isfinite(d)
        d_safe = np.where(finite, d, self.target)

        inv_tol = 1.0 / self.tolerance
        diff = d_safe - self.target
        if not self.attractive:
            diff = np.minimum(diff, 0.0)

        u = diff * inv_tol
        energy = 0.5 * self.weight * (u * u)
        deriv = self.weight * u * inv_tol

        if self.overlap_stiffness > 0.0:
            ov = np.minimum(d_safe, 0.0)
            energy = energy + 0.5 * self.overlap_stiffness * (ov * ov)
            deriv = deriv + self.overlap_stiffness * ov

        return np.where(finite, energy, 0.0), np.where(finite, deriv, 0.0)
