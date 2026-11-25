"""Matplotlib integration for lablr.

Measures text boxes, scatter/line marker radii, colors, and axes bounds in
physical typography points (1 pt = 1/72 in), then renders labels with
``ax.annotate(..., textcoords="offset points")`` and flush, color-matched leashes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import matplotlib.axes
import matplotlib.backend_bases
import matplotlib.collections
import matplotlib.colors
import matplotlib.text
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .optimizer import DEFAULT_ANGLES_DEG, LayoutMetrics, optimize_layout
from .potentials import Potential

FloatArray = NDArray[np.float64]


@dataclass
class PlacementResult:
    """Result returned by :func:`place_labels`."""

    annotations: list[matplotlib.text.Annotation]
    positions_data: FloatArray
    positions_pt: FloatArray
    offsets_pt: FloatArray
    label_sizes_pt: FloatArray
    energy_history: list[float]
    positions_history_pt: list[FloatArray]
    metrics: LayoutMetrics


def _points_per_pixel(ax: matplotlib.axes.Axes) -> float:
    return 72.0 / float(ax.figure.dpi)


def _get_renderer(ax: matplotlib.axes.Axes) -> matplotlib.backend_bases.RendererBase:
    return cast(Any, ax.figure.canvas).get_renderer()


def measure_labels_pt(
    ax: matplotlib.axes.Axes,
    labels: Sequence[str],
    text_kwargs: dict[str, Any] | None = None,
    bbox_kwargs: dict[str, Any] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Measure exact rendered ``(width, height)`` and center offset of labels in points."""
    n = len(labels)
    if n == 0:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty.copy()

    ax.figure.canvas.draw()
    renderer = _get_renderer(ax)
    scale = _points_per_pixel(ax)

    t_kw: dict[str, Any] = {"ha": "center", "va": "center", **(text_kwargs or {})}
    anchor_px = ax.transAxes.transform((0.5, 0.5))
    sizes_pt = np.zeros((n, 2), dtype=np.float64)
    center_offsets_pt = np.zeros((n, 2), dtype=np.float64)

    for i, label in enumerate(labels):
        if bbox_kwargs is not None:
            txt = ax.text(0.5, 0.5, label, transform=ax.transAxes, bbox=dict(bbox_kwargs), **t_kw)
            txt.update_bbox_position_size(renderer)
            patch = txt.get_bbox_patch()
            if patch is not None:
                ext = patch.get_window_extent(renderer)
                ec_rgba = matplotlib.colors.to_rgba(patch.get_edgecolor())
                lw = float(patch.get_linewidth()) if ec_rgba[3] > 0.0 else 0.0
            else:
                ext = txt.get_window_extent(renderer)
                lw = 0.0
        else:
            txt = ax.text(0.5, 0.5, label, transform=ax.transAxes, **t_kw)
            ext = txt.get_window_extent(renderer)
            lw = 0.0
        txt.remove()

        sizes_pt[i] = (max(ext.width * scale + lw, 1e-3), max(ext.height * scale + lw, 1e-3))
        center_offsets_pt[i] = (
            (0.5 * (ext.x0 + ext.x1) - anchor_px[0]) * scale,
            (0.5 * (ext.y0 + ext.y1) - anchor_px[1]) * scale,
        )

    return sizes_pt, center_offsets_pt


def _detect_plotted_points(
    ax: matplotlib.axes.Axes,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Extract plotted marker positions (pt), radii (pt), and opaque RGBA colors."""
    scale = _points_per_pixel(ax)
    pts_list: list[FloatArray] = []
    rad_list: list[FloatArray] = []
    col_list: list[FloatArray] = []

    for coll in ax.collections:
        if isinstance(coll, matplotlib.collections.PathCollection):
            offsets = np.asarray(coll.get_offsets(), dtype=np.float64)
            if offsets.ndim != 2 or offsets.shape[0] == 0:
                continue
            m = offsets.shape[0]
            pts_pt = coll.get_offset_transform().transform(offsets) * scale
            sizes = np.asarray(coll.get_sizes(), dtype=np.float64)
            s_arr = np.resize(sizes, m) if sizes.size > 0 else np.full(m, 38.0)
            lws = np.asarray(coll.get_linewidth(), dtype=np.float64)
            lw_arr = np.resize(lws, m) if lws.size > 0 else 0.0
            radii = 0.5 * np.sqrt(np.maximum(s_arr, 0.0)) + 0.5 * lw_arr

            fc = coll.get_facecolor()
            if len(fc) == 0:
                fc = coll.get_edgecolor()
            if len(fc) == 0:
                rgba = np.tile([0.2, 0.2, 0.2, 1.0], (m, 1))
            elif len(fc) == 1:
                rgba = np.tile(fc[0], (m, 1))
            else:
                rgba = np.asarray(fc[:m], dtype=np.float64)

            pts_list.append(pts_pt)
            rad_list.append(radii)
            col_list.append(rgba)

    for line in ax.lines:
        if line.get_marker() not in (None, "", "None", " "):
            xy = np.column_stack([line.get_xdata(), line.get_ydata()]).astype(np.float64)
            if xy.shape[0] == 0:
                continue
            m = xy.shape[0]
            r = 0.5 * float(line.get_markersize()) + 0.5 * float(line.get_markeredgewidth())
            c_rgba = matplotlib.colors.to_rgba(line.get_color())
            pts_list.append(line.get_transform().transform(xy) * scale)
            rad_list.append(np.full(m, r, dtype=np.float64))
            col_list.append(np.tile(c_rgba, (m, 1)))

    if not pts_list:
        return np.empty((0, 2)), np.empty((0,)), np.empty((0, 4))
    return np.vstack(pts_list), np.concatenate(rad_list), np.vstack(col_list)


def _match_points(
    anchors_pt: FloatArray,
    det_pts: FloatArray,
    det_radii: FloatArray,
    det_colors: FloatArray,
    point_radius: float | ArrayLike | None,
    avoid_other_points: bool,
) -> tuple[
    FloatArray,
    list[tuple[float, float, float, float] | None],
    FloatArray | None,
    FloatArray | None,
]:
    """Match anchors to plotted markers for radius & color, and collect unlabeled points."""
    n, m = anchors_pt.shape[0], det_pts.shape[0]
    radii = np.full(n, 3.5, dtype=np.float64)
    colors: list[tuple[float, float, float, float] | None] = [None] * n
    extra_pts: FloatArray | None = None
    extra_rad: FloatArray | None = None

    if n > 0 and m > 0:
        dist = np.hypot(*(anchors_pt[:, None, :] - det_pts[None, :, :]).T).T
        within = dist < 1.5
        matched = np.any(within, axis=1)
        if np.any(matched):
            rad_mat = np.where(within, det_radii[None, :], -np.inf)
            best_j = np.argmax(rad_mat, axis=1)
            for i in np.flatnonzero(matched):
                j = int(best_j[i])
                radii[i] = det_radii[j]
                rgba = det_colors[j]
                colors[i] = (float(rgba[0]), float(rgba[1]), float(rgba[2]), 1.0)
            if not np.all(matched):
                radii[~matched] = float(np.median(det_radii))
        else:
            radii[:] = float(np.median(det_radii))

        if avoid_other_points:
            unlabeled = np.min(dist, axis=0) >= 1.5
            if np.any(unlabeled):
                extra_pts = det_pts[unlabeled]
                extra_rad = det_radii[unlabeled]

    if point_radius is not None:
        radii = np.broadcast_to(np.asarray(point_radius, dtype=np.float64), (n,)).copy()

    return radii, colors, extra_pts, extra_rad


def _normalize_points(points: ArrayLike) -> FloatArray:
    if isinstance(points, tuple) and len(points) == 2:
        x, y = np.asarray(points[0], dtype=np.float64), np.asarray(points[1], dtype=np.float64)
        if x.ndim == 1 and y.ndim == 1 and x.shape == y.shape:
            return np.column_stack((x, y))
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2) or (x, y), got {arr.shape}")
    return arr


def place_labels(
    ax: matplotlib.axes.Axes,
    points: ArrayLike,
    labels: Sequence[str],
    *,
    distance: float = 8.0,
    padding: float = 4.0,
    tolerance: float | None = None,
    angles: Sequence[float] | None = DEFAULT_ANGLES_DEG,
    angle_weight: float = 0.12,
    keep_in_frame: bool = True,
    point_radius: float | ArrayLike | None = None,
    avoid_other_points: bool = True,
    text_kwargs: dict[str, Any] | None = None,
    bbox_kwargs: dict[str, Any] | None = None,
    leader_lines: bool = True,
    arrowprops: dict[str, Any] | None = None,
    anchor_potential: Potential | None = None,
    repulsion_potential: Potential | None = None,
    max_iter: int = 200,
    record_history: bool = False,
) -> PlacementResult:
    """Optimize and draw non-overlapping point labels on a Matplotlib Axes.

    All distances are in **typography points** (1 pt = 1/72 in), so layouts are
    invariant to axis scaling, aspect ratio, and export DPI. Call after plotting
    data and finalizing axis limits / ``tight_layout``.
    """
    pts_data = _normalize_points(points)
    labels_list = list(labels)
    n = pts_data.shape[0]
    if len(labels_list) != n:
        raise ValueError(f"Number of labels ({len(labels_list)}) != points ({n})")

    sizes_pt, center_offsets_pt = measure_labels_pt(
        ax, labels_list, text_kwargs=text_kwargs, bbox_kwargs=bbox_kwargs
    )
    scale = _points_per_pixel(ax)
    renderer = _get_renderer(ax)

    anchors_pt = ax.transData.transform(pts_data) * scale if n > 0 else np.empty((0, 2))
    ax_ext = ax.get_window_extent(renderer)
    bounds_pt = (
        (ax_ext.x0 * scale, ax_ext.y0 * scale, ax_ext.x1 * scale, ax_ext.y1 * scale)
        if keep_in_frame
        else None
    )

    det_pts, det_radii, det_colors = _detect_plotted_points(ax)
    radii_pt, point_colors, extra_pts_pt, extra_radii_pt = _match_points(
        anchors_pt, det_pts, det_radii, det_colors, point_radius, avoid_other_points
    )

    use_lines = leader_lines or (arrowprops is not None)
    layout = optimize_layout(
        anchors=anchors_pt,
        label_sizes=sizes_pt,
        point_radii=radii_pt,
        bounds=bounds_pt,
        distance=distance,
        padding=padding,
        tolerance=tolerance,
        angles=angles,
        angle_weight=angle_weight,
        anchor_potential=anchor_potential,
        repulsion_potential=repulsion_potential,
        avoid_lines=use_lines,
        extra_points=extra_pts_pt,
        extra_point_radii=extra_radii_pt,
        max_iter=max_iter,
        record_history=record_history,
    )

    t_kw: dict[str, Any] = {"ha": "center", "va": "center", **(text_kwargs or {})}
    offsets_pt = layout.centers - anchors_pt
    positions_data = (
        ax.transData.inverted().transform(layout.centers / scale)
        if n > 0
        else np.empty((0, 2), dtype=np.float64)
    )

    annotations: list[matplotlib.text.Annotation] = []
    for i in range(n):
        xytext = offsets_pt[i] - center_offsets_pt[i]

        pt_color = point_colors[i]
        b_kw: dict[str, Any] | None = None
        outline_color = pt_color
        if bbox_kwargs is not None:
            b_kw = dict(bbox_kwargs)
            ec = b_kw.get("edgecolor", b_kw.get("ec"))
            if ec is None and pt_color is not None:
                b_kw["edgecolor"] = pt_color
                outline_color = pt_color
            elif ec is not None and ec != "none":
                outline_color = ec

        ap: dict[str, Any] | None = None
        if use_lines:
            h_i = 0.5 * sizes_pt[i]
            rel_x = float(
                np.clip(
                    (anchors_pt[i, 0] - layout.centers[i, 0]) / (2.0 * h_i[0]) + 0.5, 0.04, 0.96
                )
            )
            rel_y = float(
                np.clip(
                    (anchors_pt[i, 1] - layout.centers[i, 1]) / (2.0 * h_i[1]) + 0.5, 0.04, 0.96
                )
            )
            ap = {
                "arrowstyle": "-",
                "color": outline_color if outline_color is not None else "0.35",
                "linewidth": 0.9,
                "shrinkA": 0.0,
                "shrinkB": float(radii_pt[i]),
                "relpos": (rel_x, rel_y),
            }
            if arrowprops is not None:
                ap.update(arrowprops)

        ann = ax.annotate(
            labels_list[i],
            xy=(float(pts_data[i, 0]), float(pts_data[i, 1])),
            xycoords="data",
            xytext=(float(xytext[0]), float(xytext[1])),
            textcoords="offset points",
            bbox=b_kw,
            arrowprops=ap,
            **t_kw,
        )
        annotations.append(ann)

    return PlacementResult(
        annotations=annotations,
        positions_data=positions_data,
        positions_pt=layout.centers,
        offsets_pt=offsets_pt,
        label_sizes_pt=sizes_pt,
        energy_history=layout.energy_history,
        positions_history_pt=layout.positions_history,
        metrics=layout.metrics,
    )
