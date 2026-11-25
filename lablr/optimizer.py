"""Simplified, AABB-accelerated L-BFGS SDF layout optimizer (pure NumPy)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from .potentials import Potential
from .sdf import (
    box_box_sdf,
    box_circle_sdf,
    box_frame_sdf,
    box_segment_sdf,
    circle_segment_sdf,
    count_segment_crossings,
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


def _expand_candidate_angles(angles_rad: FloatArray | None) -> FloatArray:
    """Include both primary compass angles and intermediate midpoints for discrete search."""
    if angles_rad is None:
        return np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    if len(angles_rad) >= 16:
        return angles_rad
    sorted_a = np.sort(angles_rad)
    next_a = np.append(sorted_a[1:], sorted_a[0] + 2.0 * np.pi)
    mids = (0.5 * (sorted_a + next_a)) % (2.0 * np.pi)
    return np.unique(np.round(np.concatenate([sorted_a, mids]), 12))


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


def _angular_well_energy_and_grad(
    deltas: FloatArray,
    half_extents: FloatArray,
    radii: FloatArray,
    target_distance: float,
    fade_scale: float,
    angles_rad: FloatArray,
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


class _Scene:
    """Single source of truth for SDF energies, analytic gradients, and candidate scoring."""

    def __init__(
        self,
        anchors: FloatArray,
        half_extents: FloatArray,
        radii: FloatArray,
        anchor_pot: Potential,
        repel_pot: Potential,
        bounds: tuple[float, float, float, float] | None = None,
        extra_points: FloatArray | None = None,
        extra_radii: FloatArray | None = None,
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
        self.extra_radii = 0.0 if extra_radii is None else extra_radii
        self.obstacle_boxes = obstacle_boxes
        self.avoid_lines = avoid_lines
        self.angles_rad = angles_rad
        self.angle_weight = angle_weight

    def energy_and_grad(
        self,
        centers: FloatArray,
        anchor_pot: Potential | None = None,
        repel_pot: Potential | None = None,
    ) -> tuple[float, FloatArray]:
        """Evaluate total SDF energy and exact analytic gradient w.r.t. ``centers``."""
        a_pot = self.anchor_pot if anchor_pot is None else anchor_pot
        r_pot = self.repel_pot if repel_pot is None else repel_pot
        cutoff = r_pot.target

        n = self.n
        idx = np.arange(n)
        energy = 0.0
        grad = np.zeros((n, 2), dtype=np.float64)

        extra_pts = self.extra_points
        has_extra = extra_pts is not None and len(extra_pts) > 0
        obs_boxes = self.obstacle_boxes

        # 1. Label <-> Points
        d_pts, g_pts = box_circle_sdf(centers, self.half_extents, self.anchors, self.radii)
        e_own, de_own = a_pot.evaluate(d_pts[idx, idx])
        energy += float(np.sum(e_own))
        grad += de_own[:, None] * g_pts[idx, idx, :]

        if n > 1:
            d_other = d_pts.copy()
            d_other[idx, idx] = np.inf
            e_o, de_o = r_pot.evaluate(d_other)
            energy += float(np.sum(e_o))
            grad += np.sum(de_o[..., None] * g_pts, axis=1)

        if has_extra and extra_pts is not None:
            d_ex, g_ex = box_circle_sdf(centers, self.half_extents, extra_pts, self.extra_radii)
            e_ex, de_ex = r_pot.evaluate(d_ex)
            energy += float(np.sum(e_ex))
            grad += np.sum(de_ex[..., None] * g_ex, axis=1)

        # 2. Label <-> Label
        if n > 1:
            d_lbl, g_lbl = box_box_sdf(centers, self.half_extents)
            e_lbl, de_lbl = r_pot.evaluate(d_lbl)
            energy += 0.5 * float(np.sum(e_lbl))
            grad += np.sum(de_lbl[..., None] * g_lbl, axis=1)

        # 3. Leader lines (AABB-accelerated) <-> Other labels & points
        if self.avoid_lines:
            local = self.anchors - centers
            attach = centers + np.clip(local, -self.half_extents, self.half_extents)
            attach_jac = (np.abs(local) >= self.half_extents).astype(np.float64)

            if n > 1:
                d_bl, g_box, g_end = box_segment_sdf(
                    centers, self.half_extents, self.anchors, attach, cutoff=cutoff
                )
                d_bl[idx, idx] = np.inf
                e_bl, de_bl = r_pot.evaluate(d_bl)
                energy += float(np.sum(e_bl))
                grad += np.sum(de_bl[..., None] * g_box, axis=1)
                grad += np.sum(de_bl[..., None] * g_end, axis=0) * attach_jac

                d_lp, g_lp = circle_segment_sdf(self.anchors, attach, self.anchors, self.radii)
                d_lp[idx, idx] = np.inf
                e_lp, de_lp = r_pot.evaluate(d_lp)
                energy += float(np.sum(e_lp))
                grad += np.sum(de_lp[..., None] * g_lp, axis=1) * attach_jac

            if has_extra and extra_pts is not None:
                d_lex, g_lex = circle_segment_sdf(self.anchors, attach, extra_pts, self.extra_radii)
                e_lex, de_lex = r_pot.evaluate(d_lex)
                energy += float(np.sum(e_lex))
                grad += np.sum(de_lex[..., None] * g_lex, axis=1) * attach_jac

        # 4. Frame walls
        if self.bounds is not None:
            d_f, g_f = box_frame_sdf(centers, self.half_extents, self.bounds)
            e_f, de_f = r_pot.evaluate(d_f)
            energy += float(np.sum(e_f))
            grad += np.sum(de_f[..., None] * g_f, axis=1)

        # 5. Fixed obstacle boxes
        if obs_boxes is not None and len(obs_boxes[0]) > 0:
            d_obs, g_obs = box_box_sdf(centers, self.half_extents, obs_boxes[0], obs_boxes[1])
            e_obs, de_obs = r_pot.evaluate(d_obs)
            energy += float(np.sum(e_obs))
            grad += np.sum(de_obs[..., None] * g_obs, axis=1)

        # 6. Distance-faded compass preference
        if self.angles_rad is not None and self.angle_weight > 0.0:
            e_ang, g_ang = _angular_well_energy_and_grad(
                centers - self.anchors,
                self.half_extents,
                self.radii,
                a_pot.target,
                a_pot.tolerance,
                self.angles_rad,
                self.angle_weight,
            )
            energy += float(np.sum(e_ang))
            grad += g_ang

        return energy, grad

    def total_energy_with_crossings(self, centers: FloatArray) -> float:
        """Total continuous energy plus discrete crossing penalty (for coupled moves)."""
        e, _ = self.energy_and_grad(centers)
        if self.avoid_lines and self.n > 1:
            attach = _nearest_box_attach_points(self.anchors, centers, self.half_extents)
            e += 15.0 * count_segment_crossings(self.anchors, attach)
        return e

    def score_candidates(
        self,
        label_idx: int,
        cands: FloatArray,
        other_centers: FloatArray | None,
        other_half: FloatArray | None,
        other_anchors: FloatArray | None,
        outward_dir: FloatArray | None = None,
        outward_bias: float = 0.0,
    ) -> FloatArray:
        """Score K candidate centers for ``label_idx`` against all active scene objects."""
        k = cands.shape[0]
        h_rep = np.broadcast_to(self.half_extents[label_idx : label_idx + 1], (k, 2))
        scores = np.zeros(k, dtype=np.float64)
        cutoff = self.repel_pot.target

        extra_pts = self.extra_points
        has_extra = extra_pts is not None and len(extra_pts) > 0
        obs_boxes = self.obstacle_boxes

        d_pts, _ = box_circle_sdf(cands, h_rep, self.anchors, self.radii)
        scores += self.anchor_pot.evaluate(d_pts[:, label_idx])[0]
        if self.n > 1:
            d_other = d_pts.copy()
            d_other[:, label_idx] = np.inf
            scores += np.sum(self.repel_pot.evaluate(d_other)[0], axis=1)

        if has_extra and extra_pts is not None:
            d_ex, _ = box_circle_sdf(cands, h_rep, extra_pts, self.extra_radii)
            scores += np.sum(self.repel_pot.evaluate(d_ex)[0], axis=1)

        has_other = other_centers is not None and other_half is not None and len(other_centers) > 0
        if has_other and other_centers is not None and other_half is not None:
            d_lbl, _ = box_box_sdf(cands, h_rep, other_centers, other_half)
            scores += np.sum(self.repel_pot.evaluate(d_lbl)[0], axis=1)

        if self.avoid_lines:
            a_rep = np.broadcast_to(self.anchors[label_idx : label_idx + 1], (k, 2))
            cand_attach = _nearest_box_attach_points(a_rep, cands, h_rep)
            if self.n > 1:
                d_lp, _ = circle_segment_sdf(a_rep, cand_attach, self.anchors, self.radii)
                d_lp[:, label_idx] = np.inf
                scores += np.sum(self.repel_pot.evaluate(d_lp)[0], axis=1)
            if has_extra and extra_pts is not None:
                d_lex, _ = circle_segment_sdf(a_rep, cand_attach, extra_pts, self.extra_radii)
                scores += np.sum(self.repel_pot.evaluate(d_lex)[0], axis=1)
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
                scores += 15.0 * np.sum(cross, axis=1)

        if self.bounds is not None:
            d_f, _ = box_frame_sdf(cands, h_rep, self.bounds)
            scores += np.sum(self.repel_pot.evaluate(d_f)[0], axis=1)

        if obs_boxes is not None and len(obs_boxes[0]) > 0:
            d_obs, _ = box_box_sdf(cands, h_rep, obs_boxes[0], obs_boxes[1])
            scores += np.sum(self.repel_pot.evaluate(d_obs)[0], axis=1)

        if self.angles_rad is not None and self.angle_weight > 0.0:
            r_rep = np.full(k, float(self.radii[label_idx]), dtype=np.float64)
            e_ang, _ = _angular_well_energy_and_grad(
                cands - self.anchors[label_idx][None, :],
                h_rep,
                r_rep,
                self.anchor_pot.target,
                self.anchor_pot.tolerance,
                self.angles_rad,
                self.angle_weight,
            )
            scores += e_ang

        if outward_dir is not None and outward_bias > 0.0:
            rel = cands - self.anchors[label_idx][None, :]
            norm = np.maximum(np.hypot(rel[:, 0], rel[:, 1]), 1e-9)
            cos_sim = (rel[:, 0] * outward_dir[0] + rel[:, 1] * outward_dir[1]) / norm
            scores += outward_bias * (1.0 - cos_sim)

        return scores

    def candidate_ring(
        self, label_idx: int, ring_offsets: tuple[float, ...], angles_rad: FloatArray
    ) -> FloatArray:
        """Place box so its nearest edge/corner is at exact SDF gap ``off`` along ``angles_rad``."""
        unit_dirs = np.column_stack((np.cos(angles_rad), np.sin(angles_rad)))
        box_off = _box_edge_anchor_offset(unit_dirs, self.half_extents[label_idx])
        anchor = self.anchors[label_idx]
        radius = float(self.radii[label_idx])
        return np.vstack(
            [
                anchor[None, :] + box_off + unit_dirs * (radius + max(off, 0.0))
                for off in ring_offsets
            ]
        )

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

        extra_pts = self.extra_points
        has_extra = extra_pts is not None and len(extra_pts) > 0

        d_pts, _ = box_circle_sdf(centers, self.half_extents, self.anchors, self.radii)
        n_pt_ov = int(np.sum(d_pts < -eps))
        min_pt = float(np.min(d_pts))
        if has_extra and extra_pts is not None:
            d_ex, _ = box_circle_sdf(centers, self.half_extents, extra_pts, self.extra_radii)
            n_pt_ov += int(np.sum(d_ex < -eps))
            min_pt = min(min_pt, float(np.min(d_ex)))

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

                toward = (
                    np.sum(
                        (self.anchors[None, :, :] - self.anchors[:, None, :]) * v_seg[:, None, :],
                        axis=-1,
                    )
                    > 1e-9
                )
                d_lp, _ = circle_segment_sdf(self.anchors, attach, self.anchors, self.radii)
                d_lp[idx, idx] = np.inf
                n_line_ov += int(np.sum((d_lp < -eps) & toward))
            if has_extra and extra_pts is not None:
                toward_ex = (
                    np.sum(
                        (extra_pts[None, :, :] - self.anchors[:, None, :]) * v_seg[:, None, :],
                        axis=-1,
                    )
                    > 1e-9
                )
                d_lex, _ = circle_segment_sdf(self.anchors, attach, extra_pts, self.extra_radii)
                n_line_ov += int(np.sum((d_lex < -eps) & toward_ex))

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
    extra_points: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    all_pts = (
        np.vstack([anchors, extra_points])
        if extra_points is not None and len(extra_points) > 0
        else anchors
    )
    sigma = max(2.2 * float(np.mean(np.hypot(half_extents[:, 0], half_extents[:, 1]))), 25.0)
    diff = anchors[:, None, :] - all_pts[None, :, :]
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


def _lbfgs_minimize(
    x0: FloatArray,
    value_and_grad,
    max_iter: int = 200,
    grad_tol: float = 1e-4,
    memory: int = 10,
    record_history: bool = False,
) -> tuple[FloatArray, list[float], list[FloatArray]]:
    """Pure-NumPy L-BFGS with Armijo backtracking line search."""
    x = x0.copy()
    f, g = value_and_grad(x)
    e_hist = [f]
    p_hist = [x.copy()] if record_history else []
    s_list: list[FloatArray] = []
    y_list: list[FloatArray] = []
    rho_list: list[float] = []

    for _ in range(max_iter):
        g_flat = g.ravel()
        g_norm = float(np.linalg.norm(g_flat))
        if g_norm < grad_tol:
            break

        q = g_flat.copy()
        alphas: list[float] = []
        for s_k, y_k, rho_k in zip(
            reversed(s_list), reversed(y_list), reversed(rho_list), strict=True
        ):
            a_k = rho_k * float(np.dot(s_k, q))
            alphas.append(a_k)
            q = q - a_k * y_k

        if s_list:
            yy = float(np.dot(y_list[-1], y_list[-1]))
            sy = float(np.dot(s_list[-1], y_list[-1]))
            r = (sy / yy if yy > 1e-18 else 1.0) * q
        else:
            r = q * min(1.0, 2.0 / max(float(np.max(np.abs(g_flat))), 1e-12))

        for s_k, y_k, rho_k, a_k in zip(s_list, y_list, rho_list, reversed(alphas), strict=True):
            r = r + s_k * (a_k - rho_k * float(np.dot(y_k, r)))

        p = -r.reshape(x.shape)
        slope = float(np.dot(g_flat, p.ravel()))
        if slope >= -1e-14:
            s_list.clear()
            y_list.clear()
            rho_list.clear()
            p = -g * min(1.0, 2.0 / max(float(np.max(np.abs(g_flat))), 1e-12))
            slope = float(np.dot(g_flat, p.ravel()))

        step, accepted = 1.0, False
        x_new, f_new, g_new = x, f, g
        for _ls in range(25):
            xt = x + step * p
            ft, gt = value_and_grad(xt)
            if ft <= f + 1e-4 * step * slope:
                x_new, f_new, g_new, accepted = xt, ft, gt, True
                break
            step *= 0.5

        if not accepted:
            s_list.clear()
            y_list.clear()
            rho_list.clear()
            p_sd = -g / max(g_norm, 1e-12)
            for sd_step in (1.0, 0.25, 0.05, 0.01, 0.002):
                xt = x + sd_step * p_sd
                ft, gt = value_and_grad(xt)
                if ft < f:
                    x_new, f_new, g_new, accepted = xt, ft, gt, True
                    break
            if not accepted:
                break

        s_vec = (x_new - x).ravel()
        y_vec = (g_new - g).ravel()
        sy = float(np.dot(s_vec, y_vec))
        if sy > 1e-12:
            if len(s_list) == memory:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)
            s_list.append(s_vec)
            y_list.append(y_vec)
            rho_list.append(1.0 / sy)

        x, f, g = x_new, f_new, g_new
        e_hist.append(f)
        if record_history:
            p_hist.append(x.copy())
        if float(np.linalg.norm(s_vec)) < 1e-5:
            break

    return x, e_hist, p_hist


def _uncross_pass(
    anchors: FloatArray, centers: FloatArray, half_extents: FloatArray, max_sweeps: int = 6
) -> bool:
    """2-opt swap pass to uncross intersecting leader lines."""
    n = centers.shape[0]
    if n <= 1:
        return False
    swapped_any = False
    for _ in range(max_sweeps):
        attach = _nearest_box_attach_points(anchors, centers, half_extents)
        mat = segments_intersect(anchors, attach, anchors, attach)
        np.fill_diagonal(mat, False)
        pairs = np.argwhere(np.triu(mat, k=1))
        if len(pairs) == 0:
            break
        cur = len(pairs)
        did_swap = False
        for i, j in pairs:
            ci, cj = centers[i].copy(), centers[j].copy()
            old_len = np.hypot(*(attach[i] - anchors[i])) + np.hypot(*(attach[j] - anchors[j]))
            centers[i], centers[j] = cj, ci
            new_attach = _nearest_box_attach_points(anchors, centers, half_extents)
            new_len = np.hypot(*(new_attach[i] - anchors[i])) + np.hypot(
                *(new_attach[j] - anchors[j])
            )
            nxt = count_segment_crossings(anchors, new_attach)
            if nxt < cur and new_len < old_len + 10.0:
                cur, attach, did_swap, swapped_any = nxt, new_attach, True, True
            else:
                centers[i], centers[j] = ci, cj
        if not did_swap:
            break
    return swapped_any


def _close_point_clusters(anchors: FloatArray, half_extents: FloatArray) -> list[list[int]]:
    """Find connected components of size >= 3 whose anchors are tightly packed."""
    n = anchors.shape[0]
    if n < 3:
        return []
    mean_h = float(np.mean(2.0 * half_extents[:, 1]))
    mean_w = float(np.mean(2.0 * half_extents[:, 0]))
    thresh = max(2.2 * mean_h, 0.75 * mean_w, 26.0)

    dist = np.hypot(*(anchors[:, None, :] - anchors[None, :, :]).T).T
    adj = (dist < thresh) & (dist > 0.0)
    visited = np.zeros(n, dtype=bool)
    clusters: list[list[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        comp, stack = [], [start]
        visited[start] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.flatnonzero(adj[u] & (~visited)):
                visited[v] = True
                stack.append(int(v))
        if len(comp) >= 3:
            clusters.append(comp)
    return clusters


def _spread_cyclic_angles(angles: FloatArray, min_gap: float) -> FloatArray:
    """Nudge sorted cyclic angles apart so adjacent points have >= ``min_gap`` separation."""
    m = len(angles)
    if m <= 1:
        return angles.copy()
    gap = min(min_gap, 0.9 * (2.0 * np.pi / m))
    out = angles.copy()
    for _ in range(12):
        for k in range(m):
            nxt = (k + 1) % m
            diff = out[nxt] - out[k] + (2.0 * np.pi if nxt == 0 else 0.0)
            if diff < gap:
                push = 0.5 * (gap - diff)
                out[k] -= push
                out[nxt] += push
    return out % (2.0 * np.pi)


def _domino_bump_pass(
    scene: _Scene,
    centers: FloatArray,
    cand_angles: FloatArray,
    ring_offsets: tuple[float, ...],
    per_label_scores: FloatArray,
    max_attempts: int = 8,
) -> bool:
    """Break 2- and 3-label deadlocks via coupled chain-bump moves."""
    n = scene.n
    if n < 2:
        return False

    worst = np.flatnonzero(per_label_scores > 0.5)
    if len(worst) == 0:
        return False
    order = worst[np.argsort(-per_label_scores[worst])][:max_attempts]

    anchor_dist = np.hypot(*(scene.anchors[:, None, :] - scene.anchors[None, :, :]).T).T
    box_dist, _ = box_box_sdf(centers, scene.half_extents)
    base_energy = scene.total_energy_with_crossings(centers)
    moved_any = False

    for i in order:
        i = int(i)
        closeness = np.minimum(anchor_dist[i], np.maximum(box_dist[i], 0.0))
        closeness[i] = np.inf
        nbrs = np.argsort(closeness)[: min(3, n - 1)]

        cands_i = scene.candidate_ring(i, ring_offsets, cand_angles)
        best_move = None
        best_e = base_energy - 0.05

        for j_raw in nbrs:
            j = int(j_raw)
            if not np.isfinite(closeness[j]):
                continue

            mask_no_ij = np.ones(n, dtype=bool)
            mask_no_ij[i] = False
            mask_no_ij[j] = False
            scores_i = scene.score_candidates(
                i,
                cands_i,
                centers[mask_no_ij] if n > 2 else None,
                scene.half_extents[mask_no_ij] if n > 2 else None,
                scene.anchors[mask_no_ij] if n > 2 else None,
            )
            bi = int(np.argmin(scores_i))
            if scores_i[bi] >= per_label_scores[i] - 0.05:
                continue
            new_ci = cands_i[bi]
            if float(np.hypot(*(new_ci - centers[i]))) < 2.0:
                continue

            trial = centers.copy()
            trial[i] = new_ci
            cands_j = scene.candidate_ring(j, ring_offsets, cand_angles)
            mask_no_j = np.ones(n, dtype=bool)
            mask_no_j[j] = False
            scores_j = scene.score_candidates(
                j,
                cands_j,
                trial[mask_no_j],
                scene.half_extents[mask_no_j],
                scene.anchors[mask_no_j],
            )
            bj = int(np.argmin(scores_j))
            trial[j] = cands_j[bj]

            k_move = None
            if scores_j[bj] > 1.0 and n > 2:
                d_jk = box_box_sdf(trial, scene.half_extents)[0][j]
                d_jk[i] = np.inf
                d_jk[j] = np.inf
                k = int(np.argmin(d_jk))
                if d_jk[k] < scene.repel_pot.target:
                    cands_k = scene.candidate_ring(k, ring_offsets, cand_angles)
                    mask_no_k = np.ones(n, dtype=bool)
                    mask_no_k[k] = False
                    scores_k = scene.score_candidates(
                        k,
                        cands_k,
                        trial[mask_no_k],
                        scene.half_extents[mask_no_k],
                        scene.anchors[mask_no_k],
                    )
                    bk = int(np.argmin(scores_k))
                    trial[k] = cands_k[bk]
                    k_move = (k, trial[k].copy())
            elif scores_i[bi] + scores_j[bj] > per_label_scores[i] + per_label_scores[j] + 1.0:
                continue

            e_trial = scene.total_energy_with_crossings(trial)
            if e_trial < best_e:
                best_e = e_trial
                best_move = (i, new_ci.copy(), j, trial[j].copy(), k_move)

        if best_move is not None:
            ii, ci, jj, cj, km = best_move
            centers[ii], centers[jj] = ci, cj
            if km is not None:
                centers[km[0]] = km[1]
            base_energy = best_e
            moved_any = True

    return moved_any


def _cluster_fanout_pass(
    scene: _Scene, centers: FloatArray, record_history: bool = False
) -> tuple[FloatArray, bool]:
    """Rip up tightly packed clusters of 3+ points and test a coordinated radial fan-out."""
    clusters = _close_point_clusters(scene.anchors, scene.half_extents)
    if not clusters:
        return centers, False

    cur_e = scene.total_energy_with_crossings(centers)
    improved_any = False

    for comp in clusters:
        comp_arr = np.array(comp, dtype=np.intp)
        pts = scene.anchors[comp_arr]
        centroid = np.mean(pts, axis=0)
        rel = pts - centroid[None, :]
        raw_angles = np.arctan2(rel[:, 1], rel[:, 0]) % (2.0 * np.pi)

        cyc_order = np.argsort(raw_angles, kind="stable")
        sorted_ids = comp_arr[cyc_order]
        spread = _spread_cyclic_angles(raw_angles[cyc_order], min_gap=2.0 * np.pi / len(comp))

        cluster_rad = float(np.max(np.hypot(rel[:, 0], rel[:, 1])))
        perim_needed = float(np.sum(2.0 * scene.half_extents[comp_arr, 1] + scene.repel_pot.target))
        fan_rad = max(cluster_rad, perim_needed / (2.0 * np.pi))
        base_gap = max(scene.anchor_pot.target, 6.0)

        trial = centers.copy()
        for idx_in_c, lbl_id in enumerate(sorted_ids):
            ang = float(spread[idx_in_c])
            u = np.array([[np.cos(ang), np.sin(ang)]])
            box_off = _box_edge_anchor_offset(u, scene.half_extents[lbl_id])[0]
            proj_out = max(fan_rad - float(np.dot(scene.anchors[lbl_id] - centroid, u[0])), 0.0)
            gap = float(scene.radii[lbl_id]) + base_gap + proj_out
            trial[lbl_id] = scene.anchors[lbl_id] + box_off + u[0] * gap

        polished, _, _ = _lbfgs_minimize(
            trial, scene.energy_and_grad, max_iter=90, record_history=False
        )
        e_new = scene.total_energy_with_crossings(polished)
        if e_new < cur_e - 0.1:
            centers = polished
            cur_e = e_new
            improved_any = True

    return centers, improved_any


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
    """Optimize 2D label positions using exact SDFs, continuation L-BFGS, and coupled escape."""
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

    cand_angles = _expand_candidate_angles(angles_rad)
    target = max(a_pot.target, 0.0)
    tol = max(a_pot.tolerance, 6.0)
    ring_offsets = (target, target + tol, target + 2.2 * tol, target + 3.6 * tol)

    # 1. Perimeter-first outward-aware initialization
    outward_dirs, crowd = _outward_density_field(anchors, half_extents, extra_pts)
    centers = np.zeros((n, 2), dtype=np.float64)
    placed: list[int] = []
    for i in np.argsort(crowd, kind="stable"):
        i = int(i)
        cands = scene.candidate_ring(i, ring_offsets, cand_angles)
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

    # 2. Soft-to-hard continuation L-BFGS
    soft_a = replace(a_pot, overlap_stiffness=min(a_pot.overlap_stiffness, 2.5))
    soft_r = replace(r_pot, overlap_stiffness=min(r_pot.overlap_stiffness, 2.5))
    centers, _, p_soft = _lbfgs_minimize(
        centers,
        lambda c: scene.energy_and_grad(c, anchor_pot=soft_a, repel_pot=soft_r),
        max_iter=25,
        record_history=record_history,
    )
    centers, energy_hist, pos_hist = _lbfgs_minimize(
        centers, scene.energy_and_grad, max_iter=max_iter, record_history=record_history
    )
    if record_history:
        pos_hist = p_soft + pos_hist[1:]

    # 3. Coordinated cluster rip-up & radial fan-out for tight knots of 3+ points
    centers, fanned = _cluster_fanout_pass(scene, centers, record_history=record_history)
    if fanned:
        e_fan, _ = scene.energy_and_grad(centers)
        energy_hist.append(e_fan)
        if record_history:
            pos_hist.append(centers.copy())

    # 4. Two rounds of single-label jumps + coupled domino bumps + 2-opt uncross + L-BFGS polish
    for _ in range(2):
        cur_scores = np.empty(n, dtype=np.float64)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            cur_scores[i] = scene.score_candidates(
                i,
                centers[i : i + 1],
                centers[mask] if n > 1 else None,
                half_extents[mask] if n > 1 else None,
                anchors[mask] if n > 1 else None,
            )[0]

        worst = np.flatnonzero(cur_scores > 0.25)
        order_esc = worst[np.argsort(-cur_scores[worst])][:24]
        changed = False
        for i in order_esc:
            i = int(i)
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            cands = np.vstack(
                [centers[i : i + 1], scene.candidate_ring(i, ring_offsets, cand_angles)]
            )
            scores = scene.score_candidates(
                i,
                cands,
                centers[mask] if n > 1 else None,
                half_extents[mask] if n > 1 else None,
                anchors[mask] if n > 1 else None,
            )
            best = int(np.argmin(scores))
            if best != 0 and scores[best] < scores[0] - 0.05:
                centers[i] = cands[best]
                cur_scores[i] = scores[best]
                changed = True

        if _domino_bump_pass(scene, centers, cand_angles, ring_offsets, cur_scores):
            changed = True

        if avoid_lines and _uncross_pass(anchors, centers, half_extents):
            changed = True

        if not changed:
            break

        centers, e2, p2 = _lbfgs_minimize(
            centers,
            scene.energy_and_grad,
            max_iter=max(80, max_iter // 2),
            record_history=record_history,
        )
        energy_hist.extend(e2[1:])
        if record_history:
            pos_hist.extend(p2[1:])

    return LayoutResult(
        centers=centers,
        half_extents=half_extents,
        energy_history=energy_hist,
        positions_history=pos_hist,
        metrics=scene.metrics(centers, energy_hist[-1]),
    )
