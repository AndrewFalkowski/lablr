"""Exact 2D Signed Distance Functions (SDFs) and analytic gradients (pure NumPy).

Conventions:
  - ``d > 0``: separated (exact Euclidean gap between surfaces).
  - ``d == 0``: touching.
  - ``d < 0``: overlapping (negative minimum translation distance).
  - Gradients point in the direction that increases signed distance.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def box_point_sdf(
    delta: FloatArray,
    half_extents: FloatArray,
    tie_dir: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Exact signed distance and unit gradient from ``delta`` to origin box ``half_extents``."""
    delta = np.asarray(delta, dtype=np.float64)
    half_extents = np.asarray(half_extents, dtype=np.float64)

    abs_d = np.abs(delta)
    q = abs_d - half_extents
    qx, qy = q[..., 0], q[..., 1]

    ux, uy = np.maximum(qx, 0.0), np.maximum(qy, 0.0)
    out_norm = np.hypot(ux, uy)
    dist = out_norm + np.minimum(np.maximum(qx, qy), 0.0)

    tie_sx = 1.0 if tie_dir is None else np.where(tie_dir[..., 0] >= 0.0, 1.0, -1.0)
    tie_sy = 1.0 if tie_dir is None else np.where(tie_dir[..., 1] >= 0.0, 1.0, -1.0)
    sx = np.where(delta[..., 0] > 0.0, 1.0, np.where(delta[..., 0] < 0.0, -1.0, tie_sx))
    sy = np.where(delta[..., 1] > 0.0, 1.0, np.where(delta[..., 1] < 0.0, -1.0, tie_sy))

    is_out = out_norm > 0.0
    safe_n = np.where(is_out, out_norm, 1.0)
    x_shallower = qx > qy

    gx = np.where(is_out, sx * (ux / safe_n), np.where(x_shallower, sx, 0.0))
    gy = np.where(is_out, sy * (uy / safe_n), np.where(x_shallower, 0.0, sy))

    if tie_dir is not None:
        zero_both = (abs_d[..., 0] < 1e-12) & (abs_d[..., 1] < 1e-12) & (~is_out)
        gx = np.where(zero_both, tie_dir[..., 0], gx)
        gy = np.where(zero_both, tie_dir[..., 1], gy)

    return dist, np.stack((gx, gy), axis=-1)


def _antisymmetric_tie_dirs(n: int) -> FloatArray:
    """Deterministic unit vectors ``T[i, j] == -T[j, i]`` for coincident boxes."""
    idx = np.arange(n)
    lo, hi = np.minimum(idx[:, None], idx[None, :]), np.maximum(idx[:, None], idx[None, :])
    angles = (lo * 2.399963229728653 + hi * 1.618033988749895) % (2.0 * np.pi)
    sign = np.where(idx[:, None] < idx[None, :], 1.0, -1.0)
    return np.stack((sign * np.cos(angles), sign * np.sin(angles)), axis=-1)


def box_box_sdf(
    centers_a: FloatArray,
    half_a: FloatArray,
    centers_b: FloatArray | None = None,
    half_b: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Pairwise exact SDF and gradient w.r.t. ``centers_a`` (self if ``b`` is None)."""
    ca = np.asarray(centers_a, dtype=np.float64)
    ha = np.asarray(half_a, dtype=np.float64)
    self_mode = centers_b is None
    cb = ca if self_mode else np.asarray(centers_b, dtype=np.float64)
    hb = ha if half_b is None else np.asarray(half_b, dtype=np.float64)
    if ca.shape[0] == 0 or cb.shape[0] == 0:
        return np.empty((ca.shape[0], cb.shape[0])), np.empty((ca.shape[0], cb.shape[0], 2))

    tie = _antisymmetric_tie_dirs(ca.shape[0]) if self_mode else None
    dist, grad = box_point_sdf(
        ca[:, None, :] - cb[None, :, :], ha[:, None, :] + hb[None, :, :], tie_dir=tie
    )
    if self_mode:
        diag = np.arange(ca.shape[0])
        dist[diag, diag] = np.inf
        grad[diag, diag, :] = 0.0
    return dist, grad


def box_circle_sdf(
    centers: FloatArray,
    half_extents: FloatArray,
    points: FloatArray,
    radii: FloatArray | float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Exact SDF and gradient from N boxes to M circular markers."""
    c = np.asarray(centers, dtype=np.float64)
    h = np.asarray(half_extents, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    if c.shape[0] == 0 or p.shape[0] == 0:
        return np.empty((c.shape[0], p.shape[0])), np.empty((c.shape[0], p.shape[0], 2))
    r = np.broadcast_to(np.asarray(radii, dtype=np.float64), (p.shape[0],))
    dist_pt, grad = box_point_sdf(c[:, None, :] - p[None, :, :], h[:, None, :])
    return dist_pt - r[None, :], grad


def box_frame_sdf(
    centers: FloatArray,
    half_extents: FloatArray,
    bounds: tuple[float, float, float, float],
) -> tuple[FloatArray, FloatArray]:
    """Signed clearance from N boxes to the 4 walls ``(xmin, ymin, xmax, ymax)``."""
    c = np.asarray(centers, dtype=np.float64)
    h = np.asarray(half_extents, dtype=np.float64)
    xmin, ymin, xmax, ymax = bounds
    dist = np.stack(
        (
            (c[:, 0] - h[:, 0]) - xmin,
            xmax - (c[:, 0] + h[:, 0]),
            (c[:, 1] - h[:, 1]) - ymin,
            ymax - (c[:, 1] + h[:, 1]),
        ),
        axis=-1,
    )
    normals = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    return dist, np.broadcast_to(normals[None, :, :], (c.shape[0], 4, 2))


def circle_segment_sdf(
    seg_starts: FloatArray,
    seg_ends: FloatArray,
    points: FloatArray,
    radii: FloatArray | float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Exact SDF from N segments ``(starts -> ends)`` to M circles with gradient w.r.t. ``ends``."""
    a = np.asarray(seg_starts, dtype=np.float64)
    b = np.asarray(seg_ends, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    n, m = a.shape[0], p.shape[0]
    if n == 0 or m == 0:
        return np.empty((n, m)), np.empty((n, m, 2))

    r = np.broadcast_to(np.asarray(radii, dtype=np.float64), (m,))
    v = b - a
    len_sq = np.sum(v * v, axis=-1)
    has_len = len_sq > 1e-18
    safe_sq = np.where(has_len, len_sq, 1.0)

    w = p[None, :, :] - a[:, None, :]
    t = np.where(
        has_len[:, None],
        np.clip(np.sum(w * v[:, None, :], axis=-1) / safe_sq[:, None], 0.0, 1.0),
        0.0,
    )
    delta = (a[:, None, :] + t[..., None] * v[:, None, :]) - p[None, :, :]
    raw = np.hypot(delta[..., 0], delta[..., 1])

    v_len = np.sqrt(safe_sq)
    perp_x = np.where(has_len, -v[:, 1] / v_len, 0.0)[:, None]
    perp_y = np.where(has_len, v[:, 0] / v_len, 1.0)[:, None]
    safe_r = np.where(raw > 1e-12, raw, 1.0)
    nx = np.where(raw > 1e-12, delta[..., 0] / safe_r, perp_x)
    ny = np.where(raw > 1e-12, delta[..., 1] / safe_r, perp_y)

    return raw - r[None, :], np.stack((t * nx, t * ny), axis=-1)


def _box_segment_sdf_flat(
    c: FloatArray, h: FloatArray, a: FloatArray, b: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Exact SDF and gradients for P aligned ``(box_p, segment_p)`` pairs."""
    cx, cy = c[:, 0], c[:, 1]
    hx, hy = h[:, 0], h[:, 1]
    ax_, ay_ = a[:, 0], a[:, 1]
    vx, vy = b[:, 0] - ax_, b[:, 1] - ay_

    len_sq = vx * vx + vy * vy
    safe_sq = np.where(len_sq > 1e-18, len_sq, 1.0)
    has_vx, has_vy = np.abs(vx) > 1e-12, np.abs(vy) > 1e-12
    safe_vx = np.where(has_vx, vx, 1.0)
    safe_vy = np.where(has_vy, vy, 1.0)

    dx_p, dx_m = (cx + hx) - ax_, (cx - hx) - ax_
    dy_p, dy_m = (cy + hy) - ay_, (cy - hy) - ay_
    sp, sm = vx + vy, vx - vy

    t_all = np.stack(
        (
            np.zeros_like(cx),
            np.ones_like(cx),
            (dx_p * vx + dy_p * vy) / safe_sq,
            (dx_p * vx + dy_m * vy) / safe_sq,
            (dx_m * vx + dy_p * vy) / safe_sq,
            (dx_m * vx + dy_m * vy) / safe_sq,
            np.where(has_vx, dx_p / safe_vx, 0.0),
            np.where(has_vx, dx_m / safe_vx, 0.0),
            np.where(has_vy, dy_p / safe_vy, 0.0),
            np.where(has_vy, dy_m / safe_vy, 0.0),
            np.where(has_vx, (cx - ax_) / safe_vx, 0.0),
            np.where(has_vy, (cy - ay_) / safe_vy, 0.0),
            np.where(
                np.abs(sp) > 1e-12,
                ((cx - ax_) + (cy - ay_) + (hx - hy)) / np.where(np.abs(sp) > 1e-12, sp, 1.0),
                0.0,
            ),
            np.where(
                np.abs(sm) > 1e-12,
                ((cx - ax_) - (cy - ay_) + (hx - hy)) / np.where(np.abs(sm) > 1e-12, sm, 1.0),
                0.0,
            ),
        ),
        axis=-1,
    )
    t_all = np.where(len_sq[:, None] > 1e-18, np.clip(t_all, 0.0, 1.0), 0.0)

    qx = ax_[:, None] + t_all * vx[:, None]
    qy = ay_[:, None] + t_all * vy[:, None]
    delta = np.stack((cx[:, None] - qx, cy[:, None] - qy), axis=-1)
    d_all, g_all = box_point_sdf(delta, h[:, None, :])

    best = np.argmin(d_all, axis=-1)
    idx = np.arange(len(cx))
    dist = d_all[idx, best]
    g_box = g_all[idx, best, :]
    g_end = -t_all[idx, best, None] * g_box
    return dist, g_box, g_end


def box_segment_sdf(
    box_centers: FloatArray,
    box_half_extents: FloatArray,
    seg_starts: FloatArray,
    seg_ends: FloatArray,
    cutoff: float | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Exact SDF and gradients between N boxes and M segments.

    When ``cutoff`` is given, pairs whose AABBs are separated by at least
    ``cutoff`` are skipped (filled with ``+inf`` and zero gradients)—100% exact
    for any one-sided repulsive potential with ``target <= cutoff``.
    """
    c = np.asarray(box_centers, dtype=np.float64)
    h = np.asarray(box_half_extents, dtype=np.float64)
    a = np.asarray(seg_starts, dtype=np.float64)
    b = np.asarray(seg_ends, dtype=np.float64)
    n, m = c.shape[0], a.shape[0]
    if n == 0 or m == 0:
        return np.empty((n, m)), np.empty((n, m, 2)), np.empty((n, m, 2))

    if cutoff is not None:
        seg_min = np.minimum(a, b)
        seg_max = np.maximum(a, b)
        box_min = c - h - cutoff
        box_max = c + h + cutoff
        close = (
            (box_max[:, None, 0] >= seg_min[None, :, 0])
            & (box_min[:, None, 0] <= seg_max[None, :, 0])
            & (box_max[:, None, 1] >= seg_min[None, :, 1])
            & (box_min[:, None, 1] <= seg_max[None, :, 1])
        )
        dist = np.full((n, m), np.inf, dtype=np.float64)
        g_box = np.zeros((n, m, 2), dtype=np.float64)
        g_end = np.zeros((n, m, 2), dtype=np.float64)
        ii, jj = np.nonzero(close)
        if len(ii) > 0:
            d_f, gb_f, ge_f = _box_segment_sdf_flat(c[ii], h[ii], a[jj], b[jj])
            dist[ii, jj] = d_f
            g_box[ii, jj] = gb_f
            g_end[ii, jj] = ge_f
        return dist, g_box, g_end

    ii, jj = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    d_f, gb_f, ge_f = _box_segment_sdf_flat(
        c[ii.ravel()], h[ii.ravel()], a[jj.ravel()], b[jj.ravel()]
    )
    return d_f.reshape(n, m), gb_f.reshape(n, m, 2), ge_f.reshape(n, m, 2)


def segments_intersect(
    starts_a: FloatArray,
    ends_a: FloatArray,
    starts_b: FloatArray,
    ends_b: FloatArray,
    eps: float = 1e-9,
) -> NDArray[np.bool_]:
    """Strict pairwise segment intersection matrix of shape ``(len(a), len(b))``."""
    sa, ea = np.asarray(starts_a, dtype=np.float64), np.asarray(ends_a, dtype=np.float64)
    sb, eb = np.asarray(starts_b, dtype=np.float64), np.asarray(ends_b, dtype=np.float64)
    if sa.shape[0] == 0 or sb.shape[0] == 0:
        return np.zeros((sa.shape[0], sb.shape[0]), dtype=bool)

    da, db = ea - sa, eb - sb
    sa_to_sb = sb[None, :, :] - sa[:, None, :]
    sa_to_eb = eb[None, :, :] - sa[:, None, :]
    sb_to_ea = ea[:, None, :] - sb[None, :, :]

    c1 = da[:, None, 0] * sa_to_sb[..., 1] - da[:, None, 1] * sa_to_sb[..., 0]
    c2 = da[:, None, 0] * sa_to_eb[..., 1] - da[:, None, 1] * sa_to_eb[..., 0]
    c3 = db[None, :, 0] * (-sa_to_sb[..., 1]) - db[None, :, 1] * (-sa_to_sb[..., 0])
    c4 = db[None, :, 0] * sb_to_ea[..., 1] - db[None, :, 1] * sb_to_ea[..., 0]
    return ((c1 * c2) < -eps) & ((c3 * c4) < -eps)


def count_segment_crossings(starts: FloatArray, ends: FloatArray) -> int:
    """Count unordered pairs of crossing segments."""
    if len(starts) <= 1:
        return 0
    mat = segments_intersect(starts, ends, starts, ends)
    np.fill_diagonal(mat, False)
    return int(np.sum(mat) // 2)
