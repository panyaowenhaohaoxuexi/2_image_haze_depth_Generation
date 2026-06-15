import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_asm_haze import (
    apply_atmospheric_scattering,
    depth_to_meters,
    find_paired_ir_path,
)


def test_metric_depth_ignores_depth_scale():
    raw_depth = np.array([[1.0, 2.0]], dtype=np.float32)

    depth_m, effective_scale = depth_to_meters(
        raw_depth,
        is_metric_depth=True,
        depth_scale=80.0,
    )

    np.testing.assert_allclose(depth_m, raw_depth)
    assert effective_scale == 1.0


def test_relative_depth_uses_depth_scale():
    raw_depth = np.array([[0.25, 0.5]], dtype=np.float32)

    depth_m, effective_scale = depth_to_meters(
        raw_depth,
        is_metric_depth=False,
        depth_scale=80.0,
    )

    np.testing.assert_allclose(depth_m, np.array([[20.0, 40.0]], dtype=np.float32))
    assert effective_scale == 80.0


def test_asm_uses_sky_soft_mask_and_koschmieder_beta():
    image = np.ones((1, 2, 3), dtype=np.float32) * 0.2
    depth_m = np.array([[10.0, 10.0]], dtype=np.float32)
    sky_mask = np.array([[0.0, 1.0]], dtype=np.float32)
    atmospheric_light = np.array([0.9, 0.9, 0.9], dtype=np.float32)

    hazy, t_final = apply_atmospheric_scattering(
        image,
        depth_m,
        sky_mask,
        atmospheric_light,
        visibility_m=100.0,
        t_sky=0.07,
    )

    expected_ground_t = np.exp(-(3.912 / 100.0) * 10.0)
    np.testing.assert_allclose(t_final[0, 0], expected_ground_t, rtol=1e-6)
    np.testing.assert_allclose(t_final[0, 1], 0.07, rtol=1e-6)
    assert hazy[0, 1, 0] > hazy[0, 0, 0]


def test_same_name_pairing_rule(tmp_path):
    vis_path = tmp_path / "vis" / "abc.jpg"
    ir_dir = tmp_path / "ir"
    vis_path.parent.mkdir()
    ir_dir.mkdir()
    (ir_dir / "abc.jpg").write_bytes(b"x")

    paired = find_paired_ir_path(vis_path, ir_dir)

    assert paired == ir_dir / "abc.jpg"
