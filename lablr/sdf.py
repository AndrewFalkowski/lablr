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
    smooth_eps: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Exact signed distance and unit gradient from ``delta`` to origin box ``half_extents``.

    When ``smooth_eps > 0``, rounds box corners up to ``0.85 * min(hx, hy)`` and smooths
    axial sign transitions so overlapping boxes slide smoothly past corners during warm-up.
    """
    delta = np.asarray(delta, dtype=np.float64)
    half_extents = np.asarray(half_extents, dtype=np.float64)

    if smooth_eps > 0.0:
        s = min(float(smooth_eps), 1.0)
        r_c = 0.85 * s * np.minimum(half_extents[..., 0], half_extents[..., 1])
        inner_h = half_extents - r_c[..., None]
        ax_eps = 0.75 * s
        hypot_d = np.hypot(delta, ax_eps)
        q = (hypot_d - ax_eps) - inner_h
        qx, qy = q[..., 0], q[..., 1]

        ux, uy = np.maximum(qx, 0.0), np.maximum(qy, 0.0)
        out_norm = np.hypot(ux, uy)
        dist = out_norm + np.minimum(np.maximum(qx, qy), 0.0) - r_c

        is_out = out_norm > 1e-12
        safe_n = np.where(is_out, out_norm, 1.0)
        x_shallower = qx > qy
        ds = delta / hypot_d
        gx = np.where(is_out, ux / safe_n, np.where(x_shallower, 1.0, 0.0)) * ds[..., 0]
        gy = np.where(is_out, uy / safe_n, np.where(x_shallower, 0.0, 1.0)) * ds[..., 1]
        return dist, np.stack((gx, gy), axis=-1)

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


def _pair_tie_dirs(ii: NDArray[np.intp], jj: NDArray[np.intp]) -> FloatArray:
    """Deterministic unit vectors ``T[i, j] == -T[j, i]`` for coincident self-pair indices."""
    lo, hi = np.minimum(ii, jj), np.maximum(ii, jj)
    angles = (lo * 2.399963229728653 + hi * 1.618033988749895) % (2.0 * np.pi)
    sign = np.where(ii < jj, 1.0, -1.0)
    return np.stack((sign * np.cos(angles), sign * np.sin(angles)), axis=-1)


def _antisymmetric_tie_dirs(n: int) -> FloatArray:
    """Deterministic unit vectors ``T[i, j] == -T[j, i]`` for coincident boxes."""
    idx = np.arange(n)
    return _pair_tie_dirs(idx[:, None], idx[None, :])


def box_box_sdf(
    centers_a: FloatArray,
    half_a: FloatArray,
    centers_b: FloatArray | None = None,
    half_b: FloatArray | None = None,
    cutoff: float | None = None,
    smooth_eps: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Pairwise exact SDF and gradient w.r.t. ``centers_a`` (self if ``b`` is None)."""
    ca = np.asarray(centers_a, dtype=np.float64)
    ha = np.asarray(half_a, dtype=np.float64)
    self_mode = centers_b is None
    cb = ca if self_mode else np.asarray(centers_b, dtype=np.float64)
    hb = ha if half_b is None else np.asarray(half_b, dtype=np.float64)
    n, m = ca.shape[0], cb.shape[0]
    if n == 0 or m == 0:
        return np.empty((n, m)), np.empty((n, m, 2))

    if cutoff is not None:
        h_sum = ha[:, None, :] + hb[None, :, :]
        delta_all = ca[:, None, :] - cb[None, :, :]
        close = (np.abs(delta_all[..., 0]) <= h_sum[..., 0] + cutoff) & (
            np.abs(delta_all[..., 1]) <= h_sum[..., 1] + cutoff
        )
        dist = np.full((n, m), np.inf, dtype=np.float64)
        grad = np.zeros((n, m, 2), dtype=np.float64)
        if self_mode:
            ii, jj = np.nonzero(np.triu(close, k=1))
            if len(ii) > 0:
                tie = _pair_tie_dirs(ii, jj) if smooth_eps <= 0.0 else None
                df, gf = box_point_sdf(
                    ca[ii] - cb[jj], ha[ii] + hb[jj], tie_dir=tie, smooth_eps=smooth_eps
                )
                dist[ii, jj] = df
                dist[jj, ii] = df
                grad[ii, jj] = gf
                grad[jj, ii] = -gf
        else:
            ii, jj = np.nonzero(close)
            if len(ii) > 0:
                df, gf = box_point_sdf(ca[ii] - cb[jj], ha[ii] + hb[jj], smooth_eps=smooth_eps)
                dist[ii, jj] = df
                grad[ii, jj] = gf
        return dist, grad

    tie = _antisymmetric_tie_dirs(n) if (self_mode and smooth_eps <= 0.0) else None
    dist, grad = box_point_sdf(
        ca[:, None, :] - cb[None, :, :],
        ha[:, None, :] + hb[None, :, :],
        tie_dir=tie,
        smooth_eps=smooth_eps,
    )
    if self_mode:
        diag = np.arange(n)
        dist[diag, diag] = np.inf
        grad[diag, diag, :] = 0.0
    return dist, grad


def box_circle_sdf(
    centers: FloatArray,
    half_extents: FloatArray,
    points: FloatArray,
    radii: FloatArray | float = 0.0,
    cutoff: float | None = None,
    smooth_eps: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Exact SDF and gradient from N boxes to M circular markers."""
    c = np.asarray(centers, dtype=np.float64)
    h = np.asarray(half_extents, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    n, m = c.shape[0], p.shape[0]
    if n == 0 or m == 0:
        return np.empty((n, m)), np.empty((n, m, 2))
    r = np.broadcast_to(np.asarray(radii, dtype=np.float64), (m,))

    if cutoff is not None:
        ext = h[:, None, :] + (r[None, :, None] + cutoff)
        delta_all = c[:, None, :] - p[None, :, :]
        close = (np.abs(delta_all[..., 0]) <= ext[..., 0]) & (
            np.abs(delta_all[..., 1]) <= ext[..., 1]
        )
        dist = np.full((n, m), np.inf, dtype=np.float64)
        grad = np.zeros((n, m, 2), dtype=np.float64)
        ii, jj = np.nonzero(close)
        if len(ii) > 0:
            df, gf = box_point_sdf(c[ii] - p[jj], h[ii], smooth_eps=smooth_eps)
            dist[ii, jj] = df - r[jj]
            grad[ii, jj] = gf
        return dist, grad

    dist_pt, grad = box_point_sdf(
        c[:, None, :] - p[None, :, :], h[:, None, :], smooth_eps=smooth_eps
    )
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


def _point_segment_flat(
    p: FloatArray, a: FloatArray, b: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Exact distance and gradients w.r.t. ``p`` and ``b`` for P aligned ``(p, a -> b)`` pairs."""
    v = b - a
    len_sq = np.sum(v * v, axis=-1)
    has_len = len_sq > 1e-18
    safe_sq = np.where(has_len, len_sq, 1.0)
    t = np.where(has_len, np.clip(np.sum((p - a) * v, axis=-1) / safe_sq, 0.0, 1.0), 0.0)

    delta = p - (a + t[:, None] * v)
    raw = np.hypot(delta[:, 0], delta[:, 1])
    v_len = np.sqrt(safe_sq)
    perp_x = np.where(has_len, v[:, 1] / v_len, 0.0)
    perp_y = np.where(has_len, -v[:, 0] / v_len, -1.0)
    safe_r = np.where(raw > 1e-12, raw, 1.0)
    nx = np.where(raw > 1e-12, delta[:, 0] / safe_r, perp_x)
    ny = np.where(raw > 1e-12, delta[:, 1] / safe_r, perp_y)

    grad_p = np.stack((nx, ny), axis=-1)
    grad_b = -t[:, None] * grad_p
    return raw, grad_p, grad_b


def circle_segment_sdf(
    seg_starts: FloatArray,
    seg_ends: FloatArray,
    points: FloatArray,
    radii: FloatArray | float = 0.0,
    cutoff: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Exact SDF from N segments ``(starts -> ends)`` to M circles with gradient w.r.t. ``ends``."""
    a = np.asarray(seg_starts, dtype=np.float64)
    b = np.asarray(seg_ends, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    n, m = a.shape[0], p.shape[0]
    if n == 0 or m == 0:
        return np.empty((n, m)), np.empty((n, m, 2))
    r = np.broadcast_to(np.asarray(radii, dtype=np.float64), (m,))

    if cutoff is not None:
        smin, smax = np.minimum(a, b), np.maximum(a, b)
        prad = r[None, :] + cutoff
        close = (
            (smax[:, None, 0] + prad >= p[None, :, 0])
            & (smin[:, None, 0] - prad <= p[None, :, 0])
            & (smax[:, None, 1] + prad >= p[None, :, 1])
            & (smin[:, None, 1] - prad <= p[None, :, 1])
        )
        dist = np.full((n, m), np.inf, dtype=np.float64)
        grad = np.zeros((n, m, 2), dtype=np.float64)
        ii, jj = np.nonzero(close)
        if len(ii) > 0:
            raw_f, _, gb_f = _point_segment_flat(p[jj], a[ii], b[ii])
            dist[ii, jj] = raw_f - r[jj]
            grad[ii, jj] = gb_f
        return dist, grad

    ii, jj = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    raw_f, _, gb_f = _point_segment_flat(p[jj.ravel()], a[ii.ravel()], b[ii.ravel()])
    return raw_f.reshape(n, m) - r[None, :], gb_f.reshape(n, m, 2)


def _box_segment_sdf_flat(
    c: FloatArray, h: FloatArray, a: FloatArray, b: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Exact 12-candidate SDF and gradients for P aligned ``(box_p, segment_p)`` pairs."""
    cx, cy = c[:, 0], c[:, 1]
    hx, hy = h[:, 0], h[:, 1]
    ax_, ay_ = a[:, 0], a[:, 1]
    vx, vy = b[:, 0] - ax_, b[:, 1] - ay_

    len_sq = vx * vx + vy * vy
    safe_sq = np.where(len_sq > 1e-18, len_sq, 1.0)
    has_vx, has_vy = np.abs(vx) > 1e-12, np.abs(vy) > 1e-12
    safe_vx = np.where(has_vx, vx, 1.0)
    safe_vy = np.where(has_vy, vy, 1.0)

    d0x, d0y = cx - ax_, cy - ay_
    dx_p, dx_m = d0x + hx, d0x - hx
    dy_p, dy_m = d0y + hy, d0y - hy
    sp, sm = vx + vy, vx - vy
    has_sp, has_sm = np.abs(sp) > 1e-12, np.abs(sm) > 1e-12
    safe_sp = np.where(has_sp, sp, 1.0)
    safe_sm = np.where(has_sm, sm, 1.0)
    dh = hx - hy

    t_all = np.stack(
        (
            np.zeros_like(cx),
            np.ones_like(cx),
            (dx_p * vx + dy_p * vy) / safe_sq,
            (dx_p * vx + dy_m * vy) / safe_sq,
            (dx_m * vx + dy_p * vy) / safe_sq,
            (dx_m * vx + dy_m * vy) / safe_sq,
            np.where(has_vx, d0x / safe_vx, 0.0),
            np.where(has_vy, d0y / safe_vy, 0.0),
            np.where(has_sp, (d0x + d0y + dh) / safe_sp, 0.0),
            np.where(has_sp, (d0x + d0y - dh) / safe_sp, 0.0),
            np.where(has_sm, (d0x - d0y + dh) / safe_sm, 0.0),
            np.where(has_sm, (d0x - d0y - dh) / safe_sm, 0.0),
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

    When ``cutoff`` is given, pairs whose AABBs are separated by more than
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
        seg_min, seg_max = np.minimum(a, b), np.maximum(a, b)
        box_min, box_max = c - h - cutoff, c + h + cutoff
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


def _segment_segment_sdf_flat(
    a1: FloatArray, b1: FloatArray, a2: FloatArray, b2: FloatArray, eps: float = 1e-9
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Exact C1-across-zero 2D SDF and endpoint gradients for P aligned segment pairs."""
    r0, _, gb2_0 = _point_segment_flat(a1, a2, b2)
    r1, gb1_1, gb2_1 = _point_segment_flat(b1, a2, b2)
    r2, _, gb1_2 = _point_segment_flat(a2, a1, b1)
    r3, gb2_3, gb1_3 = _point_segment_flat(b2, a1, b1)

    zeros = np.zeros_like(b1)
    r_all = np.stack((r0, r1, r2, r3), axis=-1)
    gb1_all = np.stack((zeros, gb1_1, gb1_2, gb1_3), axis=1)
    gb2_all = np.stack((gb2_0, gb2_1, zeros, gb2_3), axis=1)

    best = np.argmin(r_all, axis=-1)
    p_idx = np.arange(len(a1))
    r_min = r_all[p_idx, best]
    gb1 = gb1_all[p_idx, best]
    gb2 = gb2_all[p_idx, best]

    d1, d2 = b1 - a1, b2 - a2
    a1_a2, a1_b2, a2_b1 = a2 - a1, b2 - a1, b1 - a2
    c1 = d1[:, 0] * a1_a2[:, 1] - d1[:, 1] * a1_a2[:, 0]
    c2 = d1[:, 0] * a1_b2[:, 1] - d1[:, 1] * a1_b2[:, 0]
    c3 = d2[:, 0] * (-a1_a2[:, 1]) - d2[:, 1] * (-a1_a2[:, 0])
    c4 = d2[:, 0] * a2_b1[:, 1] - d2[:, 1] * a2_b1[:, 0]
    crossed = ((c1 * c2) < -eps) & ((c3 * c4) < -eps)
    sgn = np.where(crossed, -1.0, 1.0)

    return sgn * r_min, sgn[:, None] * gb1, sgn[:, None] * gb2


def segment_segment_sdf(
    starts_a: FloatArray,
    ends_a: FloatArray,
    starts_b: FloatArray | None = None,
    ends_b: FloatArray | None = None,
    cutoff: float | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Exact 2D SDF between segments ``(starts_a -> ends_a)`` and ``(starts_b -> ends_b)``.

    Positive when separated, zero when touching, and negative (minimum endpoint/pivot
    translation distance to uncross) when strictly intersecting, with gradients w.r.t.
    ``ends_a`` and ``ends_b``.
    """
    sa = np.asarray(starts_a, dtype=np.float64)
    ea = np.asarray(ends_a, dtype=np.float64)
    self_mode = starts_b is None
    sb = sa if self_mode else np.asarray(starts_b, dtype=np.float64)
    eb = ea if ends_b is None else np.asarray(ends_b, dtype=np.float64)
    n, m = sa.shape[0], sb.shape[0]
    if n == 0 or m == 0:
        return np.empty((n, m)), np.empty((n, m, 2)), np.empty((n, m, 2))

    if cutoff is not None:
        smin_a, smax_a = np.minimum(sa, ea), np.maximum(sa, ea)
        smin_b, smax_b = np.minimum(sb, eb), np.maximum(sb, eb)
        close = (
            (smax_a[:, None, 0] + cutoff >= smin_b[None, :, 0])
            & (smin_a[:, None, 0] - cutoff <= smax_b[None, :, 0])
            & (smax_a[:, None, 1] + cutoff >= smin_b[None, :, 1])
            & (smin_a[:, None, 1] - cutoff <= smax_b[None, :, 1])
        )
        dist = np.full((n, m), np.inf, dtype=np.float64)
        g_ea = np.zeros((n, m, 2), dtype=np.float64)
        g_eb = np.zeros((n, m, 2), dtype=np.float64)
        if self_mode:
            ii, jj = np.nonzero(np.triu(close, k=1))
            if len(ii) > 0:
                df, g1, g2 = _segment_segment_sdf_flat(sa[ii], ea[ii], sb[jj], eb[jj])
                dist[ii, jj] = df
                dist[jj, ii] = df
                g_ea[ii, jj] = g1
                g_eb[ii, jj] = g2
                g_ea[jj, ii] = g2
                g_eb[jj, ii] = g1
        else:
            ii, jj = np.nonzero(close)
            if len(ii) > 0:
                df, g1, g2 = _segment_segment_sdf_flat(sa[ii], ea[ii], sb[jj], eb[jj])
                dist[ii, jj] = df
                g_ea[ii, jj] = g1
                g_eb[ii, jj] = g2
        return dist, g_ea, g_eb

    ii, jj = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    df, g1, g2 = _segment_segment_sdf_flat(
        sa[ii.ravel()], ea[ii.ravel()], sb[jj.ravel()], eb[jj.ravel()]
    )
    dist = df.reshape(n, m)
    g_ea = g1.reshape(n, m, 2)
    g_eb = g2.reshape(n, m, 2)
    if self_mode:
        diag = np.arange(n)
        dist[diag, diag] = np.inf
        g_ea[diag, diag, :] = 0.0
        g_eb[diag, diag, :] = 0.0
    return dist, g_ea, g_eb


def count_segment_crossings(starts: FloatArray, ends: FloatArray) -> int:
    """Count unordered pairs of crossing segments."""
    if len(starts) <= 1:
        return 0
    mat = segments_intersect(starts, ends, starts, ends)
    np.fill_diagonal(mat, False)
    return int(np.sum(mat) // 2)
