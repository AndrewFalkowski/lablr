"""Simplified, AABB-accelerated FIRE 2.0 SDF layout optimizer (pure NumPy)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from .potentials import Potential
from .sdf import (
    box_box_sdf,
    box_circle_sdf,
    box_frame_sdf,
    box_point_sdf,
    box_segment_sdf,
    circle_segment_sdf,
    count_segment_crossings,
    segment_segment_sdf,
    segments_intersect,
)

FloatArray = NDArray[np.float64]
DEFAULT_ANGLES_DEG: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)


@dataclass(frozen=True)
class LayoutMetrics:
    """Geometric quality diagnostics for a label layout (in points)."""

    n_label_overlaps: int
    n_point_overlaps: int
    n_frame_clips: int
    n_line_overlaps: int
    n_crossings: int
    min_label_clearance: float
    min_point_clearance: float
    min_frame_clearance: float
    final_energy: float


@dataclass
class LayoutResult:
    """Output of :func:`optimize_layout` in physical points coordinates."""

    centers: FloatArray
    half_extents: FloatArray
    energy_history: list[float]
    positions_history: list[FloatArray]
    metrics: LayoutMetrics


def _normalize_angles_rad(angles_deg: Sequence[float] | None) -> FloatArray | None:
    if angles_deg is None or len(angles_deg) == 0:
        return None
    arr = np.deg2rad(np.asarray(angles_deg, dtype=np.float64)) % (2.0 * np.pi)
    return np.unique(np.round(arr, 12))


def _box_edge_anchor_offset(unit_dirs: FloatArray, half_extent: FloatArray) -> FloatArray:
    """Offset from nearest box edge/corner facing ``-unit_dirs`` to box center."""
    sx = np.where(np.abs(unit_dirs[:, 0]) > 1e-6, np.sign(unit_dirs[:, 0]), 0.0)
    sy = np.where(np.abs(unit_dirs[:, 1]) > 1e-6, np.sign(unit_dirs[:, 1]), 0.0)
    return np.column_stack((sx * half_extent[0], sy * half_extent[1]))


def _nearest_box_attach_points(
    anchors: FloatArray, centers: FloatArray, half_extents: FloatArray
) -> FloatArray:
    """Exact nearest point on each box perimeter to its own anchor point."""
    return centers + np.clip(anchors - centers, -half_extents, half_extents)


def _racetrack_candidates(
    anchor: FloatArray,
    half_extent: FloatArray,
    radius: float,
    ring_offsets: Sequence[float],
    angles_rad: FloatArray | None = None,
    n_racetrack: int = 16,
) -> FloatArray:
    """Sample candidates along exact SDF level-set racetracks plus primary compass slots."""
    hx, hy = float(half_extent[0]), float(half_extent[1])
    rad_arr = radius + np.maximum(np.asarray(ring_offsets, dtype=np.float64), 0.0)
    r_cnt = len(rad_arr)
    arc = 0.5 * np.pi * rad_arr

    seg_lens = np.column_stack(
        (
            np.full(r_cnt, hy),
            arc,
            np.full(r_cnt, 2.0 * hx),
            arc,
            np.full(r_cnt, 2.0 * hy),
            arc,
            np.full(r_cnt, 2.0 * hx),
            arc,
            np.full(r_cnt, hy),
        )
    )
    cum = np.concatenate((np.zeros((r_cnt, 1)), np.cumsum(seg_lens, axis=1)), axis=1)
    s = np.linspace(0.0, 1.0, n_racetrack, endpoint=False)[None, :] * cum[:, -1:]
    seg = np.sum(s[:, :, None] >= cum[:, None, 1:-1], axis=-1)
    u = s - cum[np.arange(r_cnt)[:, None], seg]

    r_b = np.broadcast_to(rad_arr[:, None], s.shape)
    safe_r = np.maximum(r_b, 1e-9)
    px, py = np.zeros_like(s), np.zeros_like(s)

    m = seg == 0
    px[m], py[m] = (hx + r_b)[m], u[m]
    m = seg == 1
    th = u[m] / safe_r[m]
    px[m], py[m] = hx + r_b[m] * np.cos(th), hy + r_b[m] * np.sin(th)
    m = seg == 2
    px[m], py[m] = hx - u[m], (hy + r_b)[m]
    m = seg == 3
    th = 0.5 * np.pi + u[m] / safe_r[m]
    px[m], py[m] = -hx + r_b[m] * np.cos(th), hy + r_b[m] * np.sin(th)
    m = seg == 4
    px[m], py[m] = (-hx - r_b)[m], hy - u[m]
    m = seg == 5
    th = np.pi + u[m] / safe_r[m]
    px[m], py[m] = -hx + r_b[m] * np.cos(th), -hy + r_b[m] * np.sin(th)
    m = seg == 6
    px[m], py[m] = -hx + u[m], (-hy - r_b)[m]
    m = seg == 7
    th = 1.5 * np.pi + u[m] / safe_r[m]
    px[m], py[m] = hx + r_b[m] * np.cos(th), -hy + r_b[m] * np.sin(th)
    m = seg == 8
    px[m], py[m] = (hx + r_b)[m], -hy + u[m]

    rt = anchor[None, :] + np.stack((px.ravel(), py.ravel()), axis=-1)
    if angles_rad is None or len(angles_rad) == 0:
        return rt

    unit_dirs = np.column_stack((np.cos(angles_rad), np.sin(angles_rad)))
    box_off = _box_edge_anchor_offset(unit_dirs, half_extent)
    comp = np.vstack([anchor[None, :] + box_off + unit_dirs * float(r_val) for r_val in rad_arr])
    return np.vstack((comp, rt))


def _angular_well_energy_and_grad(
    deltas: FloatArray,
    half_extents: FloatArray,
    radii: FloatArray,
    target_distance: float,
    fade_scale: float,
    angles_rad: FloatArray | None,
    weight: float,
) -> tuple[FloatArray, FloatArray]:
    """C1-smooth, distance-faded compass well on the nearest-edge vector ``u``."""
    if weight <= 0.0 or angles_rad is None or len(angles_rad) == 0:
        return np.zeros(deltas.shape[0]), np.zeros_like(deltas)

    sorted_a = np.sort(angles_rad)
    gaps = np.diff(np.append(sorted_a, sorted_a[0] + 2.0 * np.pi))
    sigma = max(0.5 * float(np.min(gaps)), 1e-3)

    abs_d = np.abs(deltas)
    u = np.sign(deltas) * np.maximum(abs_d - half_extents, 0.0)
    du_ddelta = (abs_d > half_extents).astype(np.float64)

    rho_sq = np.sum(u * u, axis=-1)
    active = rho_sq > 1e-12
    safe_rho_sq = np.where(active, rho_sq, 1.0)
    rho = np.sqrt(safe_rho_sq)
    phi = np.arctan2(u[:, 1], u[:, 0])

    diff = np.arctan2(
        np.sin(phi[:, None] - sorted_a[None, :]),
        np.cos(phi[:, None] - sorted_a[None, :]),
    )
    psi = diff[np.arange(len(deltas)), np.argmin(np.abs(diff), axis=1)]
    psi_c = np.clip(psi, -sigma, sigma)
    freq = np.pi / sigma

    g_phi = 0.5 * weight * (1.0 - np.cos(freq * psi_c))
    dg_dphi = 0.5 * weight * freq * np.sin(freq * psi_c)

    inv_fade = 1.0 / max(fade_scale, 1e-3)
    s = np.maximum(rho - radii - max(target_distance, 0.0), 0.0) * inv_fade
    fade = np.exp(-0.5 * s * s)
    dfade_drho = -s * fade * inv_fade

    energy = np.where(active, fade * g_phi, 0.0)
    grad_t = np.stack((-u[:, 1], u[:, 0]), axis=-1) * (fade * dg_dphi / safe_rho_sq)[:, None]
    grad_r = u * (g_phi * dfade_drho / rho)[:, None]
    grad = np.where(active[:, None], (grad_t + grad_r) * du_ddelta, 0.0)
    return energy, grad


def _face_centering_energy_and_grad(
    deltas: FloatArray,
    half_extents: FloatArray,
    weight: float,
    r0: float = 2.0,
) -> tuple[FloatArray, FloatArray]:
    """Tiny C1-smooth centering pull along flat box faces so uncrowded labels center neatly."""
    if weight <= 0.0 or len(deltas) == 0:
        return np.zeros(len(deltas)), np.zeros_like(deltas)

    dx, dy = deltas[:, 0], deltas[:, 1]
    hx, hy = np.maximum(half_extents[:, 0], 1.0), np.maximum(half_extents[:, 1], 1.0)
    ax, ay = np.abs(dx), np.abs(dy)
    sx, sy = np.sign(dx), np.sign(dy)

    ox, oy = np.maximum(ax - hx, 0.0), np.maximum(ay - hy, 0.0)
    r0_sq = r0 * r0
    dx_den, dy_den = ox * ox + r0_sq, oy * oy + r0_sq
    rx, ry = (ox * ox) / dx_den, (oy * oy) / dy_den
    drx_dox = (2.0 * ox * r0_sq) / (dx_den * dx_den)
    dry_doy = (2.0 * oy * r0_sq) / (dy_den * dy_den)

    ux, uy = np.clip(dx / hx, -1.0, 1.0), np.clip(dy / hy, -1.0, 1.0)
    in_x, in_y = ax < hx, ay < hy
    gx = np.where(in_x, 0.5 * (1.0 - np.cos(np.pi * ux)), 1.0)
    gy = np.where(in_y, 0.5 * (1.0 - np.cos(np.pi * uy)), 1.0)
    dgx_ddx = np.where(in_x, (0.5 * np.pi / hx) * np.sin(np.pi * ux), 0.0)
    dgy_ddy = np.where(in_y, (0.5 * np.pi / hy) * np.sin(np.pi * uy), 0.0)

    wx = ry * (1.0 - rx)
    wy = rx * (1.0 - ry)
    energy = weight * (wx * gx + wy * gy)
    grad_x = weight * (ry * (-drx_dox * sx) * gx + wx * dgx_ddx + (drx_dox * sx) * (1.0 - ry) * gy)
    grad_y = weight * ((dry_doy * sy) * (1.0 - rx) * gx + rx * (-dry_doy * sy) * gy + wy * dgy_ddy)
    return energy, np.stack((grad_x, grad_y), axis=-1)


class _Scene:
    """Unified SDF scene evaluator for energies, analytic gradients, and candidate scoring."""

    def __init__(
        self,
        anchors: FloatArray,
        half_extents: FloatArray,
        radii: FloatArray,
        anchor_pot: Potential,
        repel_pot: Potential,
        bounds: tuple[float, float, float, float] | None = None,
        extra_points: FloatArray | None = None,
        extra_radii: FloatArray | float | None = None,
        obstacle_boxes: tuple[FloatArray, FloatArray] | None = None,
        avoid_lines: bool = True,
        angles_rad: FloatArray | None = None,
        angle_weight: float = 0.12,
    ) -> None:
        self.anchors = anchors
        self.half_extents = half_extents
        self.radii = radii
        self.n = anchors.shape[0]
        self.anchor_pot = anchor_pot
        self.repel_pot = repel_pot
        self.bounds = bounds
        self.extra_points = extra_points
        self.extra_radii: FloatArray | float = 0.0 if extra_radii is None else extra_radii
        self.obstacle_boxes = obstacle_boxes
        self.avoid_lines = avoid_lines
        self.angles_rad = angles_rad
        self.angle_weight = angle_weight

        if extra_points is not None and len(extra_points) > 0:
            ex_r = np.broadcast_to(
                np.asarray(self.extra_radii, dtype=np.float64), (len(extra_points),)
            )
            self._all_points = np.vstack((anchors, extra_points))
            self._all_radii = np.concatenate((radii, ex_r))
        else:
            self._all_points = anchors
            self._all_radii = radii

    def evaluate(
        self,
        centers: FloatArray,
        anchor_pot: Potential | None = None,
        repel_pot: Potential | None = None,
        smooth_eps: float = 0.0,
    ) -> tuple[float, FloatArray, FloatArray]:
        """Compute ``(total_energy, grad, per_label_energy)`` in one AABB-pruned pass."""
        a_pot = self.anchor_pot if anchor_pot is None else anchor_pot
        r_pot = self.repel_pot if repel_pot is None else repel_pot
        cutoff = r_pot.target

        n = self.n
        idx = np.arange(n)
        per_e = np.zeros(n, dtype=np.float64)
        grad = np.zeros((n, 2), dtype=np.float64)
        if n == 0:
            return 0.0, grad, per_e

        # 1. Own anchor tether (O(N))
        d_own, g_own = box_point_sdf(
            centers - self.anchors, self.half_extents, smooth_eps=smooth_eps
        )
        e_own, de_own = a_pot.evaluate(d_own - self.radii)
        per_e += e_own
        grad += de_own[:, None] * g_own

        # 2. Other markers & unlabeled background points (AABB-pruned)
        if len(self._all_points) > 1:
            d_pts, g_pts = box_circle_sdf(
                centers,
                self.half_extents,
                self._all_points,
                self._all_radii,
                cutoff=cutoff,
                smooth_eps=smooth_eps,
            )
            d_pts[idx, idx] = np.inf
            e_pts, de_pts = r_pot.evaluate(d_pts)
            per_e += np.sum(e_pts, axis=1)
            grad += np.sum(de_pts[..., None] * g_pts, axis=1)

        # 3. Label <-> Label (symmetric AABB-pruned)
        if n > 1:
            d_lbl, g_lbl = box_box_sdf(
                centers, self.half_extents, cutoff=cutoff, smooth_eps=smooth_eps
            )
            e_lbl, de_lbl = r_pot.evaluate(d_lbl)
            per_e += 0.5 * np.sum(e_lbl, axis=1)
            grad += np.sum(de_lbl[..., None] * g_lbl, axis=1)

        # 4. Leader lines <-> boxes, markers, and crossing leader lines
        if self.avoid_lines:
            local = self.anchors - centers
            if smooth_eps > 0.0:
                h = self.half_extents
                hp = np.hypot(local + h, smooth_eps)
                hm = np.hypot(local - h, smooth_eps)
                clipped = 0.5 * ((local + h - hp) + (local - h + hm))
                dclip = 0.5 * ((1.0 - (local + h) / hp) + (1.0 + (local - h) / hm))
                attach = centers + clipped
                attach_jac = 1.0 - dclip
            else:
                attach = centers + np.clip(local, -self.half_extents, self.half_extents)
                attach_jac = (np.abs(local) >= self.half_extents).astype(np.float64)

            if n > 1:
                d_bl, g_box, g_end = box_segment_sdf(
                    centers, self.half_extents, self.anchors, attach, cutoff=cutoff
                )
                d_bl[idx, idx] = np.inf
                e_bl, de_bl = r_pot.evaluate(d_bl)
                per_e += 0.5 * np.sum(e_bl, axis=1) + 0.5 * np.sum(e_bl, axis=0)
                grad += np.sum(de_bl[..., None] * g_box, axis=1)
                grad += np.sum(de_bl[..., None] * g_end, axis=0) * attach_jac

                cross_stiff = max(1.5 * r_pot.overlap_stiffness, 12.0)
                cross_pot = Potential(
                    target=0.0,
                    tolerance=1.0,
                    weight=0.0,
                    attractive=False,
                    overlap_stiffness=cross_stiff,
                )
                d_ss, g_ss, _ = segment_segment_sdf(self.anchors, attach, cutoff=0.0)
                e_ss, de_ss = cross_pot.evaluate(d_ss)
                per_e += 0.5 * np.sum(e_ss, axis=1)
                grad += np.sum(de_ss[..., None] * g_ss, axis=1) * attach_jac

            if len(self._all_points) > 1:
                d_lp, g_lp = circle_segment_sdf(
                    self.anchors, attach, self._all_points, self._all_radii, cutoff=cutoff
                )
                d_lp[idx, idx] = np.inf
                e_lp, de_lp = r_pot.evaluate(d_lp)
                per_e += np.sum(e_lp, axis=1)
                grad += np.sum(de_lp[..., None] * g_lp, axis=1) * attach_jac

        # 5. Frame walls
        if self.bounds is not None:
            d_f, g_f = box_frame_sdf(centers, self.half_extents, self.bounds)
            e_f, de_f = r_pot.evaluate(d_f)
            per_e += np.sum(e_f, axis=1)
            grad += np.sum(de_f[..., None] * g_f, axis=1)

        # 6. Fixed obstacle boxes
        if self.obstacle_boxes is not None and len(self.obstacle_boxes[0]) > 0:
            d_obs, g_obs = box_box_sdf(
                centers,
                self.half_extents,
                self.obstacle_boxes[0],
                self.obstacle_boxes[1],
                cutoff=cutoff,
                smooth_eps=smooth_eps,
            )
            e_obs, de_obs = r_pot.evaluate(d_obs)
            per_e += np.sum(e_obs, axis=1)
            grad += np.sum(de_obs[..., None] * g_obs, axis=1)

        # 7. Distance-faded compass preference + tiny face-centering pull
        if self.angles_rad is not None and self.angle_weight > 0.0:
            deltas = centers - self.anchors
            e_ang, g_ang = _angular_well_energy_and_grad(
                deltas,
                self.half_extents,
                self.radii,
                a_pot.target,
                a_pot.tolerance,
                self.angles_rad,
                self.angle_weight,
            )
            e_ctr, g_ctr = _face_centering_energy_and_grad(
                deltas, self.half_extents, 0.1 * self.angle_weight
            )
            per_e += e_ang + e_ctr
            grad += g_ang + g_ctr

        return float(np.sum(per_e)), grad, per_e

    def energy_and_grad(
        self,
        centers: FloatArray,
        anchor_pot: Potential | None = None,
        repel_pot: Potential | None = None,
        smooth_eps: float = 0.0,
    ) -> tuple[float, FloatArray]:
        """Evaluate total SDF energy and exact analytic gradient w.r.t. ``centers``."""
        energy, grad, _ = self.evaluate(
            centers, anchor_pot=anchor_pot, repel_pot=repel_pot, smooth_eps=smooth_eps
        )
        return energy, grad

    def score_candidates(
        self,
        label_idx: int,
        cands: FloatArray,
        other_centers: FloatArray | None = None,
        other_half: FloatArray | None = None,
        other_anchors: FloatArray | None = None,
        outward_dir: FloatArray | None = None,
        outward_bias: float = 0.0,
    ) -> FloatArray:
        """Score K candidate centers for ``label_idx`` against active scene objects."""
        k = cands.shape[0]
        h_rep = np.broadcast_to(self.half_extents[label_idx : label_idx + 1], (k, 2))
        scores = np.zeros(k, dtype=np.float64)
        cutoff = self.repel_pot.target

        d_own, _ = box_point_sdf(cands - self.anchors[label_idx][None, :], h_rep)
        scores += self.anchor_pot.evaluate(d_own - self.radii[label_idx])[0]

        if len(self._all_points) > 1:
            d_pts, _ = box_circle_sdf(
                cands, h_rep, self._all_points, self._all_radii, cutoff=cutoff
            )
            d_pts[:, label_idx] = np.inf
            scores += np.sum(self.repel_pot.evaluate(d_pts)[0], axis=1)

        has_other = other_centers is not None and other_half is not None and len(other_centers) > 0
        if has_other and other_centers is not None and other_half is not None:
            d_lbl, _ = box_box_sdf(cands, h_rep, other_centers, other_half, cutoff=cutoff)
            scores += np.sum(self.repel_pot.evaluate(d_lbl)[0], axis=1)

        if self.avoid_lines:
            a_rep = np.broadcast_to(self.anchors[label_idx : label_idx + 1], (k, 2))
            cand_attach = _nearest_box_attach_points(a_rep, cands, h_rep)
            if len(self._all_points) > 1:
                d_lp, _ = circle_segment_sdf(
                    a_rep, cand_attach, self._all_points, self._all_radii, cutoff=cutoff
                )
                d_lp[:, label_idx] = np.inf
                scores += np.sum(self.repel_pot.evaluate(d_lp)[0], axis=1)
            if (
                has_other
                and other_centers is not None
                and other_half is not None
                and other_anchors is not None
            ):
                other_attach = _nearest_box_attach_points(other_anchors, other_centers, other_half)
                d_bl, _, _ = box_segment_sdf(
                    cands, h_rep, other_anchors, other_attach, cutoff=cutoff
                )
                scores += np.sum(self.repel_pot.evaluate(d_bl)[0], axis=1)
                d_lb, _, _ = box_segment_sdf(
                    other_centers, other_half, a_rep, cand_attach, cutoff=cutoff
                )
                scores += np.sum(self.repel_pot.evaluate(d_lb)[0], axis=0)
                cross = segments_intersect(a_rep, cand_attach, other_anchors, other_attach)
                scores += 20.0 * np.sum(cross, axis=1)

        if self.bounds is not None:
            d_f, _ = box_frame_sdf(cands, h_rep, self.bounds)
            scores += np.sum(self.repel_pot.evaluate(d_f)[0], axis=1)

        if self.obstacle_boxes is not None and len(self.obstacle_boxes[0]) > 0:
            d_obs, _ = box_box_sdf(
                cands, h_rep, self.obstacle_boxes[0], self.obstacle_boxes[1], cutoff=cutoff
            )
            scores += np.sum(self.repel_pot.evaluate(d_obs)[0], axis=1)

        if self.angles_rad is not None and self.angle_weight > 0.0:
            rel = cands - self.anchors[label_idx][None, :]
            e_ang, _ = _angular_well_energy_and_grad(
                rel,
                h_rep,
                np.full(k, float(self.radii[label_idx]), dtype=np.float64),
                self.anchor_pot.target,
                self.anchor_pot.tolerance,
                self.angles_rad,
                self.angle_weight,
            )
            e_ctr, _ = _face_centering_energy_and_grad(rel, h_rep, 0.02 * self.angle_weight)
            scores += e_ang + e_ctr

        if outward_dir is not None and outward_bias > 0.0:
            rel = cands - self.anchors[label_idx][None, :]
            norm = np.maximum(np.hypot(rel[:, 0], rel[:, 1]), 1e-9)
            cos_sim = (rel[:, 0] * outward_dir[0] + rel[:, 1] * outward_dir[1]) / norm
            scores += outward_bias * (1.0 - cos_sim)

        return scores

    def metrics(self, centers: FloatArray, final_energy: float, eps: float = 1e-6) -> LayoutMetrics:
        """Compute geometric overlap, clipping, and crossing diagnostics."""
        n = self.n
        if n == 0:
            return LayoutMetrics(0, 0, 0, 0, 0, np.inf, np.inf, np.inf, final_energy)
        idx = np.arange(n)

        if n > 1:
            d_lbl, _ = box_box_sdf(centers, self.half_extents)
            n_lbl_ov = int(np.sum(d_lbl < -eps) // 2)
            min_lbl = float(np.min(d_lbl))
        else:
            n_lbl_ov, min_lbl = 0, float("inf")

        d_pts, _ = box_circle_sdf(centers, self.half_extents, self._all_points, self._all_radii)
        n_pt_ov = int(np.sum(d_pts < -eps))
        min_pt = float(np.min(d_pts))

        if self.bounds is not None:
            d_f, _ = box_frame_sdf(centers, self.half_extents, self.bounds)
            n_frame = int(np.sum(np.any(d_f < -eps, axis=1)))
            min_frame = float(np.min(d_f))
        else:
            n_frame, min_frame = 0, float("inf")

        n_line_ov = 0
        attach = _nearest_box_attach_points(self.anchors, centers, self.half_extents)
        if self.avoid_lines:
            v_seg = attach - self.anchors
            if n > 1:
                d_bl, _, _ = box_segment_sdf(
                    centers, self.half_extents, self.anchors, attach, cutoff=self.repel_pot.target
                )
                d_bl[idx, idx] = np.inf
                n_line_ov += int(np.sum(d_bl < -eps))

            if len(self._all_points) > 1:
                toward = (
                    np.sum(
                        (self._all_points[None, :, :] - self.anchors[:, None, :])
                        * v_seg[:, None, :],
                        axis=-1,
                    )
                    > 1e-9
                )
                d_lp, _ = circle_segment_sdf(
                    self.anchors, attach, self._all_points, self._all_radii, cutoff=0.0
                )
                d_lp[idx, idx] = np.inf
                n_line_ov += int(np.sum((d_lp < -eps) & toward))

        n_cross = count_segment_crossings(self.anchors, attach) if self.avoid_lines else 0
        return LayoutMetrics(
            n_label_overlaps=n_lbl_ov,
            n_point_overlaps=n_pt_ov,
            n_frame_clips=n_frame,
            n_line_overlaps=n_line_ov,
            n_crossings=n_cross,
            min_label_clearance=min_lbl,
            min_point_clearance=min_pt,
            min_frame_clearance=min_frame,
            final_energy=final_energy,
        )


def _outward_density_field(
    anchors: FloatArray,
    half_extents: FloatArray,
    all_points: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    sigma = max(2.2 * float(np.mean(np.hypot(half_extents[:, 0], half_extents[:, 1]))), 25.0)
    diff = anchors[:, None, :] - all_points[None, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    dist = np.sqrt(dist_sq)
    valid = dist > 1e-9
    w = np.where(valid, np.exp(-0.5 * dist_sq / (sigma * sigma)), 0.0)
    crowd = np.sum(w, axis=1)
    away = np.sum(diff * (w / np.where(valid, dist, 1.0))[..., None], axis=1)
    norms = np.hypot(away[:, 0], away[:, 1])
    outward = np.where(
        (norms > 1e-6)[:, None], away / np.where(norms > 1e-6, norms, 1.0)[:, None], 0.0
    )
    return outward, crowd


def _fire_minimize(
    x0: FloatArray,
    value_and_grad: Callable[[FloatArray], tuple[float, FloatArray]],
    max_iter: int = 200,
    grad_tol: float = 1e-4,
    dt_init: float = 0.35,
    dt_max: float = 3.5,
    max_step: float = 8.0,
    record_history: bool = False,
) -> tuple[FloatArray, list[float], list[FloatArray]]:
    """Pure-NumPy FIRE 2.0 (Fast Inertial Relaxation Engine, Guenole et al. 2020)."""
    x = x0.copy()
    v = np.zeros_like(x)
    prev_step = np.zeros_like(x)
    dt = dt_init
    dt_min = 0.005
    alpha_start = 0.25
    alpha = alpha_start
    f_inc, f_dec, f_alpha = 1.1, 0.5, 0.99
    n_delay = 5
    n_pos = 0

    f, g = value_and_grad(x)
    best_x, best_f = x.copy(), f
    e_hist = [f]
    p_hist = [x.copy()] if record_history else []

    for it in range(max_iter):
        max_g = float(np.max(np.hypot(g[:, 0], g[:, 1]))) if len(g) > 0 else 0.0
        if max_g < grad_tol:
            break

        if it > 0:
            power = -float(np.sum(g * v))
            if power > 0.0:
                n_pos += 1
                if n_pos > n_delay:
                    dt = min(dt * f_inc, dt_max)
                    alpha *= f_alpha
                v_norm = float(np.linalg.norm(v))
                g_norm = max(float(np.linalg.norm(g)), 1e-12)
                v = (1.0 - alpha) * v - alpha * (v_norm / g_norm) * g
            else:
                n_pos = 0
                dt = max(dt * f_dec, dt_min)
                alpha = alpha_start
                x = x - 0.5 * prev_step
                v.fill(0.0)

        v = v - dt * g
        step = dt * v
        disp = np.hypot(step[:, 0], step[:, 1])
        max_d = float(np.max(disp)) if len(disp) > 0 else 0.0
        if max_d > max_step:
            scale = max_step / max_d
            step *= scale
            v *= scale

        x = x + step
        prev_step = step
        f, g = value_and_grad(x)
        if f < best_f:
            best_f = f
            best_x = x.copy()
        e_hist.append(f)
        if record_history:
            p_hist.append(x.copy())

    if best_f < f:
        x = best_x
        e_hist.append(best_f)
        if record_history:
            p_hist.append(x.copy())

    return x, e_hist, p_hist


def _size_aware_swap(
    anchors: FloatArray, centers: FloatArray, half_extents: FloatArray, i: int, j: int
) -> tuple[FloatArray, FloatArray]:
    """Swap attachment target positions of labels ``i`` and ``j``, adjusting for box sizes."""
    att = _nearest_box_attach_points(anchors[[i, j]], centers[[i, j]], half_extents[[i, j]])
    dir_i = att[1] - anchors[i]
    dir_j = att[0] - anchors[j]
    sx_i = float(np.sign(dir_i[0])) if abs(dir_i[0]) > 2.0 else 0.0
    sy_i = float(np.sign(dir_i[1])) if abs(dir_i[1]) > 2.0 else 0.0
    sx_j = float(np.sign(dir_j[0])) if abs(dir_j[0]) > 2.0 else 0.0
    sy_j = float(np.sign(dir_j[1])) if abs(dir_j[1]) > 2.0 else 0.0
    new_ci = att[1] + np.array([sx_i * half_extents[i, 0], sy_i * half_extents[i, 1]])
    new_cj = att[0] + np.array([sx_j * half_extents[j, 0], sy_j * half_extents[j, 1]])
    return new_ci, new_cj


def _escape_and_uncross(
    scene: _Scene, centers: FloatArray, ring_offsets: tuple[float, ...]
) -> tuple[FloatArray, bool]:
    """Single-label escape jumps for stressed labels followed by energy-guarded 2-opt uncrossing."""
    n = scene.n
    changed = False
    ext_rings = (*ring_offsets, ring_offsets[-1] + 22.0)

    _, _, per_e = scene.evaluate(centers)
    stressed = np.flatnonzero(per_e > 0.35)
    for i_raw in stressed[np.argsort(-per_e[stressed])][:20]:
        i = int(i_raw)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        cands = np.vstack(
            (
                centers[i : i + 1],
                _racetrack_candidates(
                    scene.anchors[i],
                    scene.half_extents[i],
                    float(scene.radii[i]),
                    ext_rings,
                    scene.angles_rad,
                ),
            )
        )
        scores = scene.score_candidates(
            i,
            cands,
            centers[mask] if n > 1 else None,
            scene.half_extents[mask] if n > 1 else None,
            scene.anchors[mask] if n > 1 else None,
        )
        best = int(np.argmin(scores))
        if best != 0 and scores[best] < scores[0] - 0.1:
            centers[i] = cands[best]
            changed = True

    if scene.avoid_lines and n > 1:
        for _ in range(6):
            attach = _nearest_box_attach_points(scene.anchors, centers, scene.half_extents)
            mat = segments_intersect(scene.anchors, attach, scene.anchors, attach)
            np.fill_diagonal(mat, False)
            pairs = np.argwhere(np.triu(mat, k=1))
            if len(pairs) == 0:
                break
            cur_cross = len(pairs)
            cur_total = scene.energy_and_grad(centers)[0] + 25.0 * cur_cross
            did_swap = False
            for i_raw, j_raw in pairs:
                i, j = int(i_raw), int(j_raw)
                cand_swaps = (
                    (centers[j].copy(), centers[i].copy()),
                    _size_aware_swap(scene.anchors, centers, scene.half_extents, i, j),
                )
                best_trial: tuple[FloatArray, float] | None = None
                best_total = cur_total + 40.0
                for ci_new, cj_new in cand_swaps:
                    trial = centers.copy()
                    trial[i], trial[j] = ci_new, cj_new
                    att_t = _nearest_box_attach_points(scene.anchors, trial, scene.half_extents)
                    nxt_cross = count_segment_crossings(scene.anchors, att_t)
                    if nxt_cross < cur_cross:
                        tot_t = scene.energy_and_grad(trial)[0] + 25.0 * nxt_cross
                        if tot_t < best_total:
                            best_total = tot_t
                            best_trial = (trial, tot_t)
                if best_trial is not None:
                    centers, cur_total = best_trial
                    did_swap = changed = True
                    break
            if not did_swap:
                break

    return centers, changed


def optimize_layout(
    anchors: FloatArray,
    label_sizes: FloatArray,
    point_radii: FloatArray | float = 3.5,
    bounds: tuple[float, float, float, float] | None = None,
    distance: float = 8.0,
    padding: float = 4.0,
    tolerance: float | None = None,
    angles: Sequence[float] | None = DEFAULT_ANGLES_DEG,
    angle_weight: float = 0.12,
    anchor_potential: Potential | None = None,
    repulsion_potential: Potential | None = None,
    avoid_lines: bool = True,
    extra_points: FloatArray | None = None,
    extra_point_radii: FloatArray | float | None = None,
    obstacle_boxes: tuple[FloatArray, FloatArray] | None = None,
    max_iter: int = 200,
    record_history: bool = False,
) -> LayoutResult:
    """Optimize 2D label positions using exact SDFs, capsule-smooth FIRE 2.0, and 2-opt escape."""
    anchors = np.asarray(anchors, dtype=np.float64)
    label_sizes = np.asarray(label_sizes, dtype=np.float64)
    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError(f"anchors must have shape (N, 2), got {anchors.shape}")
    if label_sizes.shape != anchors.shape:
        raise ValueError(f"label_sizes must have shape {anchors.shape}, got {label_sizes.shape}")

    n = anchors.shape[0]
    half_extents = 0.5 * label_sizes
    radii = np.broadcast_to(np.asarray(point_radii, dtype=np.float64), (n,)).copy()
    a_pot = anchor_potential or Potential.anchor(distance=distance, tolerance=tolerance)
    r_pot = repulsion_potential or Potential.repel(padding=padding)
    angles_rad = _normalize_angles_rad(angles)

    extra_pts = None if extra_points is None else np.asarray(extra_points, dtype=np.float64)
    extra_rad = (
        None if extra_point_radii is None else np.asarray(extra_point_radii, dtype=np.float64)
    )

    scene = _Scene(
        anchors=anchors,
        half_extents=half_extents,
        radii=radii,
        anchor_pot=a_pot,
        repel_pot=r_pot,
        bounds=bounds,
        extra_points=extra_pts,
        extra_radii=extra_rad,
        obstacle_boxes=obstacle_boxes,
        avoid_lines=avoid_lines,
        angles_rad=angles_rad,
        angle_weight=angle_weight,
    )

    if n == 0:
        empty = np.empty((0, 2), dtype=np.float64)
        return LayoutResult(
            empty, empty, [0.0], [empty] if record_history else [], scene.metrics(empty, 0.0)
        )

    target = max(a_pot.target, 0.0)
    tol = max(a_pot.tolerance, 6.0)
    ring_offsets = (target, target + tol, target + 2.2 * tol, target + 3.6 * tol)

    # 1. Outward-aware greedy initialization along exact SDF level-set racetracks
    outward_dirs, crowd = _outward_density_field(anchors, half_extents, scene._all_points)
    centers = np.zeros((n, 2), dtype=np.float64)
    placed: list[int] = []
    for i_raw in np.argsort(crowd, kind="stable"):
        i = int(i_raw)
        cands = _racetrack_candidates(
            anchors[i], half_extents[i], float(radii[i]), ring_offsets, angles_rad
        )
        has_out = float(np.hypot(*outward_dirs[i])) > 0.5 and float(crowd[i]) > 0.6
        scores = scene.score_candidates(
            i,
            cands,
            centers[placed] if placed else None,
            half_extents[placed] if placed else None,
            anchors[placed] if placed else None,
            outward_dir=outward_dirs[i] if has_out else None,
            outward_bias=0.35 if has_out else 0.0,
        )
        centers[i] = cands[int(np.argmin(scores))]
        placed.append(i)

    # 2. Capsule-smoothed FIRE 2.0 warm-up (if any initial tension exists) -> exact FIRE 2.0 polish
    _, _, per0 = scene.evaluate(centers)
    e_soft: list[float] = []
    p_soft: list[FloatArray] = []
    if float(np.max(per0)) > 0.25:
        soft_a = replace(a_pot, overlap_stiffness=min(a_pot.overlap_stiffness, 5.0))
        soft_r = replace(r_pot, overlap_stiffness=min(r_pot.overlap_stiffness, 5.0))
        centers, e_soft, p_soft = _fire_minimize(
            centers,
            lambda c: scene.energy_and_grad(c, anchor_pot=soft_a, repel_pot=soft_r, smooth_eps=1.0),
            max_iter=min(60, max(25, max_iter // 3)),
            grad_tol=2e-2,
            record_history=record_history,
        )

    centers, energy_hist, pos_hist = _fire_minimize(
        centers, scene.energy_and_grad, max_iter=max_iter, record_history=record_history
    )
    if e_soft:
        energy_hist = e_soft + energy_hist[1:]
        if record_history:
            pos_hist = p_soft + pos_hist[1:]

    # 3. Stressed-label escape jumps + size-aware 2-opt uncrossing + exact FIRE 2.0 polish
    for _ in range(2):
        centers, changed = _escape_and_uncross(scene, centers, ring_offsets)
        if not changed:
            break
        centers, e2, p2 = _fire_minimize(
            centers,
            scene.energy_and_grad,
            max_iter=max(100, max_iter // 2),
            record_history=record_history,
        )
        energy_hist.extend(e2[1:])
        if record_history:
            pos_hist.extend(p2[1:])

    final_e, _ = scene.energy_and_grad(centers)
    energy_hist[-1] = final_e
    return LayoutResult(
        centers=centers,
        half_extents=half_extents,
        energy_history=energy_hist,
        positions_history=pos_hist,
        metrics=scene.metrics(centers, final_e),
    )
