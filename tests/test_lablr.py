"""Tests for lablr SDFs, potentials, FIRE 2.0 optimizer, and Matplotlib frontend."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from lablr import (
    DEFAULT_ANGLES_DEG,
    Potential,
    box_box_sdf,
    box_circle_sdf,
    box_frame_sdf,
    box_point_sdf,
    box_segment_sdf,
    circle_segment_sdf,
    measure_labels_pt,
    optimize_layout,
    place_labels,
    segment_segment_sdf,
)
from lablr.optimizer import _normalize_angles_rad, _Scene


def test_sdfs_exact_values_and_aabb_cutoff():
    half = np.array([10.0, 5.0])
    d, g = box_point_sdf(np.array([14.0, 2.0]), half)
    assert np.isclose(d, 4.0) and np.allclose(g, [1.0, 0.0])

    d, g = box_point_sdf(np.array([13.0, 9.0]), half)
    assert np.isclose(d, 5.0) and np.allclose(g, [0.6, 0.8])

    d, g = box_point_sdf(np.array([2.0, 3.0]), half)
    assert np.isclose(d, -2.0) and np.allclose(g, [0.0, 1.0])

    # Coincident box-box antisymmetry & box-frame clearance
    d_bb, g_bb = box_box_sdf(
        np.array([[50.0, 50.0], [50.0, 50.0]]), np.array([[20.0, 7.0], [20.0, 7.0]])
    )
    assert np.isinf(d_bb[0, 0]) and np.isclose(d_bb[0, 1], -14.0)
    assert np.allclose(g_bb[0, 1], -g_bb[1, 0])
    d_fr, _ = box_frame_sdf(
        np.array([[10.0, 10.0]]), np.array([[4.0, 2.0]]), (0.0, 0.0, 20.0, 15.0)
    )
    assert np.allclose(d_fr[0], [6.0, 6.0, 8.0, 3.0])

    # Segment SDFs with and without AABB cutoff match exactly on close pairs
    starts, ends = np.array([[0.0, 0.0]]), np.array([[20.0, 0.0]])
    d_cs, g_end = circle_segment_sdf(
        starts, ends, np.array([[10.0, 5.0], [10.0, 0.5]]), np.array([2.0, 2.0])
    )
    assert np.isclose(d_cs[0, 0], 3.0) and np.allclose(g_end[0, 0], [0.0, -0.5])
    assert np.isclose(d_cs[0, 1], -1.5)

    boxes_c = np.array([[10.0, 6.0], [10.0, 0.5], [100.0, 100.0]])
    boxes_h = np.array([[4.0, 2.0], [4.0, 2.0], [4.0, 2.0]])
    d_full, gb_full, ge_full = box_segment_sdf(boxes_c, boxes_h, starts, ends)
    d_cut, gb_cut, ge_cut = box_segment_sdf(boxes_c, boxes_h, starts, ends, cutoff=5.0)
    assert np.allclose(d_full[:2], d_cut[:2])
    assert np.allclose(gb_full[:2], gb_cut[:2])
    assert np.allclose(ge_full[:2], ge_cut[:2])
    assert np.isinf(d_cut[2, 0])

    # Regression check: 12-candidate box_segment_sdf matches dense ground truth across penetrations
    rng = np.random.default_rng(42)
    ts = np.linspace(0.0, 1.0, 4001)
    for _ in range(40):
        c_r = rng.uniform(-20.0, 20.0, size=(1, 2))
        h_r = rng.uniform(5.0, 30.0, size=(1, 2))
        a_r = c_r + rng.uniform(-35.0, 35.0, size=(1, 2))
        b_r = c_r + rng.uniform(-35.0, 35.0, size=(1, 2))
        d_exact, _, _ = box_segment_sdf(c_r, h_r, a_r, b_r)
        pts_t = a_r + ts[:, None] * (b_r - a_r)
        d_gt = float(np.min(box_point_sdf(c_r - pts_t, h_r)[0]))
        assert d_exact[0, 0] <= d_gt + 1e-12 and np.isclose(d_exact[0, 0], d_gt, atol=2e-2)

    # AABB cutoffs on box_box_sdf, box_circle_sdf, circle_segment_sdf & segment_segment_sdf
    c_multi = np.array([[0.0, 0.0], [12.0, 0.0], [200.0, 200.0]])
    h_multi = np.array([[5.0, 3.0], [5.0, 3.0], [5.0, 3.0]])
    dbb_full, gbb_full = box_box_sdf(c_multi, h_multi)
    dbb_cut, gbb_cut = box_box_sdf(c_multi, h_multi, cutoff=4.0)
    assert np.isclose(dbb_full[0, 1], dbb_cut[0, 1]) and np.isinf(dbb_cut[0, 2])
    assert np.allclose(gbb_full[0, 1], gbb_cut[0, 1])
    assert np.allclose(gbb_cut[0, 1], -gbb_cut[1, 0])

    dbc_full, gbc_full = box_circle_sdf(c_multi, h_multi, np.array([[8.0, 0.0]]), 1.5)
    dbc_cut, gbc_cut = box_circle_sdf(c_multi, h_multi, np.array([[8.0, 0.0]]), 1.5, cutoff=4.0)
    assert np.allclose(dbc_full[:2], dbc_cut[:2]) and np.isinf(dbc_cut[2, 0])
    assert np.allclose(gbc_full[:2], gbc_cut[:2])

    dcs_cut, gcs_cut = circle_segment_sdf(
        starts, ends, np.array([[10.0, 5.0], [100.0, 100.0]]), np.array([2.0, 2.0]), cutoff=5.0
    )
    assert np.isclose(dcs_cut[0, 0], 3.0) and np.isinf(dcs_cut[0, 1])
    assert np.allclose(gcs_cut[0, 0], [0.0, -0.5])

    # Crossed vs uncrossed segment_segment_sdf continuity & finite-difference gradient
    sa = np.array([[0.0, 4.0], [0.0, 4.0]])
    ea = np.array([[0.0, -1.0], [0.0, 1.5]])
    sb = np.array([[-5.0, 0.0]])
    eb = np.array([[5.0, 0.0]])
    dss, gea, geb = segment_segment_sdf(sa, ea, sb, eb, cutoff=3.0)
    assert np.isclose(dss[0, 0], -1.0) and np.isclose(dss[1, 0], 1.5)
    assert np.allclose(gea[0, 0], [0.0, 1.0]) and np.allclose(gea[1, 0], [0.0, 1.0])
    assert np.allclose(geb[0, 0], [0.0, -0.5]) and np.allclose(geb[1, 0], [0.0, -0.5])


def test_scene_analytic_gradient_matches_finite_differences():
    rng = np.random.default_rng(7)
    n = 6
    anchors = rng.uniform(20.0, 180.0, size=(n, 2))
    centers = anchors + rng.uniform(-18.0, 18.0, size=(n, 2))
    half = rng.uniform(8.0, 22.0, size=(n, 2))
    radii = rng.uniform(2.0, 6.0, size=(n,))

    scene = _Scene(
        anchors=anchors,
        half_extents=half,
        radii=radii,
        anchor_pot=Potential.anchor(distance=8.0, tolerance=6.0),
        repel_pot=Potential.repel(padding=4.0),
        bounds=(0.0, 0.0, 200.0, 200.0),
        extra_points=rng.uniform(20.0, 180.0, size=(4, 2)),
        extra_radii=np.full(4, 3.0),
        obstacle_boxes=(np.array([[100.0, 100.0]]), np.array([[15.0, 10.0]])),
        avoid_lines=True,
        angles_rad=_normalize_angles_rad(DEFAULT_ANGLES_DEG),
        angle_weight=0.25,
    )

    for smooth_eps in (0.0, 1.0):
        _, grad = scene.energy_and_grad(centers, smooth_eps=smooth_eps)

        eps = 1e-6
        num_grad = np.zeros_like(grad)
        for i in range(n):
            for axis in range(2):
                cp, cm = centers.copy(), centers.copy()
                cp[i, axis] += eps
                cm[i, axis] -= eps
                ep, _ = scene.energy_and_grad(cp, smooth_eps=smooth_eps)
                em, _ = scene.energy_and_grad(cm, smooth_eps=smooth_eps)
                num_grad[i, axis] = (ep - em) / (2.0 * eps)

        assert np.allclose(grad, num_grad, rtol=1e-4, atol=1e-4)


def test_potential_and_single_point_target():
    well = Potential(target=10.0, tolerance=5.0, weight=2.0, attractive=True)
    e, de = well.evaluate(np.array([-2.0, 0.0, 5.0, 10.0, 15.0, np.inf]))
    assert np.isclose(e[3], 0.0) and np.isclose(de[3], 0.0)
    assert de[2] < 0.0 and de[4] > 0.0 and e[5] == 0.0

    rep = Potential.repel(padding=4.0)
    er, der = rep.evaluate(np.array([-1.0, 2.0, 4.0, 10.0]))
    assert er[0] > er[1] > 0.0 and er[2] == 0.0 and der[3] == 0.0

    with pytest.raises(ValueError):
        Potential(target=5.0, tolerance=0.0)

    anchors = np.array([[100.0, 100.0]])
    sizes = np.array([[60.0, 14.0]])
    radii = np.array([4.0])
    res = optimize_layout(anchors, sizes, point_radii=radii, bounds=(0, 0, 300, 300), distance=12.0)
    d, _ = box_circle_sdf(res.centers, res.half_extents, anchors, radii)
    assert np.isclose(d[0, 0], 12.0, atol=5e-2)


def test_coincident_points_and_fixed_obstacles():
    anchors = np.array([[150.0, 150.0]] * 5)
    sizes = np.tile([36.0, 14.0], (5, 1))
    res = optimize_layout(anchors, sizes, point_radii=4.0, bounds=(0, 0, 300, 300))
    assert res.metrics.n_label_overlaps == 0
    assert res.metrics.n_point_overlaps == 0

    obs_c, obs_h = np.array([[120.0, 115.0]]), np.array([[20.0, 15.0]])
    res2 = optimize_layout(
        np.array([[100.0, 100.0]]),
        np.array([[30.0, 12.0]]),
        point_radii=3.0,
        bounds=(0.0, 0.0, 300.0, 300.0),
        obstacle_boxes=(obs_c, obs_h),
    )
    assert box_box_sdf(res2.centers, res2.half_extents, obs_c, obs_h)[0][0, 0] >= 0.0


def _rendered_annotation_bboxes_pt(fig, annotations):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = 72.0 / fig.dpi
    centers, halfs = [], []
    for ann in annotations:
        patch = ann.get_bbox_patch()
        ext = (
            patch.get_window_extent(renderer)
            if patch is not None
            else ann.get_window_extent(renderer)
        )
        centers.append([0.5 * (ext.x0 + ext.x1) * scale, 0.5 * (ext.y0 + ext.y1) * scale])
        halfs.append([0.5 * ext.width * scale, 0.5 * ext.height * scale])
    return np.array(centers), np.array(halfs)


def test_matplotlib_dense_scatter_dpi_invariant_and_flush_matched_leashes():
    np.random.seed(1)
    points = np.random.rand(40, 2)
    labels = [f"Label {i}" for i in range(40)]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    sc = ax.scatter(points[:, 0], points[:, 1], s=50, color="tab:blue", zorder=3)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    fig.tight_layout()

    text_kw = {"fontsize": 9}
    bbox_kw = {"boxstyle": "round,pad=0.2", "facecolor": "white"}
    res = place_labels(
        ax, points, labels, text_kwargs=text_kw, bbox_kwargs=bbox_kw, record_history=True
    )

    assert text_kw == {"fontsize": 9}
    assert bbox_kw == {"boxstyle": "round,pad=0.2", "facecolor": "white"}
    assert res.metrics.n_label_overlaps == 0
    assert res.metrics.n_point_overlaps == 0
    assert res.metrics.n_frame_clips == 0
    assert res.metrics.n_crossings == 0

    expected_rgb = np.asarray(sc.get_facecolor())[0, :3]
    ann0 = res.annotations[0]
    assert ann0.arrowprops is not None
    assert np.isclose(ann0.arrowprops["shrinkB"], 0.5 * np.sqrt(50.0) + 0.5, atol=0.5)
    bbox_patch = ann0.get_bbox_patch()
    arrow_patch = ann0.arrow_patch
    assert bbox_patch is not None
    assert arrow_patch is not None
    assert np.allclose(bbox_patch.get_edgecolor()[:3], expected_rgb, atol=1e-2)
    assert np.allclose(arrow_patch.get_edgecolor()[:3], expected_rgb, atol=1e-2)

    c100, h100 = _rendered_annotation_bboxes_pt(fig, res.annotations)
    assert np.min(box_box_sdf(c100, h100)[0]) >= -0.5

    fig.set_dpi(300)
    c300, h300 = _rendered_annotation_bboxes_pt(fig, res.annotations)
    assert np.allclose(c100, c300, atol=0.5)
    assert np.allclose(h100, h300, atol=0.5)
    assert np.min(box_box_sdf(c300, h300)[0]) >= -0.5
    plt.close(fig)


def test_anisotropic_log_scale_and_edge_cases():
    rng = np.random.default_rng(42)
    x = 10.0 ** rng.uniform(0, 4, size=18)
    y = rng.uniform(0.001, 0.01, size=18)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.scatter(x, y, s=80, color="tab:green", zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 2e4)
    ax.set_ylim(0.0, 0.012)
    fig.tight_layout()

    res = place_labels(
        ax,
        (x, y),
        [f"Sample {i}" for i in range(18)],
        text_kwargs={"fontsize": 9},
        bbox_kwargs={"boxstyle": "round,pad=0.2", "facecolor": "white"},
    )
    assert res.metrics.n_label_overlaps == 0
    assert res.metrics.n_point_overlaps == 0
    assert res.metrics.n_frame_clips == 0

    assert len(place_labels(ax, np.empty((0, 2)), []).annotations) == 0
    s_plain, _ = measure_labels_pt(ax, ["Label 0"], text_kwargs={"fontsize": 10})
    s_boxed, _ = measure_labels_pt(
        ax, ["Label 0"], text_kwargs={"fontsize": 10}, bbox_kwargs={"boxstyle": "round,pad=0.3"}
    )
    assert s_boxed[0, 0] > s_plain[0, 0] + 3.0
    plt.close(fig)
