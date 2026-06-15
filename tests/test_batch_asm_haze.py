import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_asm_haze import (
    apply_atmospheric_scattering,
    depth_to_meters,
    find_paired_ir_path,
    process_one_image,
    save_fogmap,
    smooth_depth_for_haze,
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


def test_fogmap_is_one_minus_t_grayscale_label(tmp_path):
    t_final = np.array(
        [
            [0.0, 0.25],
            [0.5, 1.0],
        ],
        dtype=np.float32,
    )
    out_path = tmp_path / "fogmap.png"

    save_fogmap(out_path, t_final)

    with Image.open(out_path) as image:
        assert image.mode == "L"
        saved = np.asarray(image)
    expected = ((np.clip(1.0 - t_final, 0.0, 1.0) * 255.0 + 0.5)).astype(np.uint8)
    np.testing.assert_array_equal(saved, expected)


def test_smooth_depth_for_haze_reduces_local_structure():
    depth_m = np.zeros((21, 21), dtype=np.float32)
    depth_m[:, :10] = 10.0
    depth_m[:, 10:] = 80.0
    depth_m[10, 10] = 5.0

    unsmoothed = smooth_depth_for_haze(depth_m, sigma=0.0)
    smoothed = smooth_depth_for_haze(depth_m, sigma=3.0)

    np.testing.assert_array_equal(unsmoothed, depth_m)
    assert smoothed.dtype == np.float32
    assert smoothed[10, 10] > depth_m[10, 10]
    assert smoothed[:, 0].mean() < smoothed[:, -1].mean()


def test_process_one_image_writes_three_uniform_outputs(tmp_path):
    class DummyDepthEstimator:
        def predict(self, image, target_hw):
            return np.ones(target_hw, dtype=np.float32) * 10.0

    vis_dir = tmp_path / "vis"
    ir_dir = tmp_path / "ir"
    out_dir = tmp_path / "out"
    debug_dir = out_dir / "debug"
    fogmap_dir = out_dir / "fogmap"
    vis_dir.mkdir()
    ir_dir.mkdir()
    Image.fromarray(np.full((8, 8, 3), 80, dtype=np.uint8), mode="RGB").save(
        vis_dir / "sample.jpg"
    )

    args = Namespace(
        is_metric_depth=True,
        depth_scale=80.0,
        disable_sky_mask=True,
        dark_channel_patch=3,
        visibilities=[200.0, 100.0, 50.0],
        debug_visibility=100.0,
        t_sky=0.07,
        save_fogmap=True,
        depth_smooth_sigma=3.0,
    )

    process_one_image(
        vis_path=vis_dir / "sample.jpg",
        ir_dir=ir_dir,
        out_dir=out_dir,
        debug_dir=debug_dir,
        fogmap_dir=fogmap_dir,
        depth_estimator=DummyDepthEstimator(),
        args=args,
    )

    fog_means = {}
    for tag in ["V200", "V100", "V50"]:
        assert (out_dir / f"sample_{tag}.jpg").exists()
        fog_path = fogmap_dir / f"sample_{tag}_fogmap.png"
        assert fog_path.exists()
        with Image.open(fog_path) as fog_image:
            assert fog_image.mode == "L"
            fog_array = np.asarray(fog_image)
        assert fog_array.ndim == 2
        assert fog_array.min() == fog_array.max()
        fog_means[tag] = float(fog_array.mean())

    assert len(list(out_dir.glob("sample_*.jpg"))) == 3
    assert fog_means["V50"] > fog_means["V100"] > fog_means["V200"]
    assert not (out_dir / "sample_V200_nf.jpg").exists()
    assert not (out_dir / "sample_V100_nf.jpg").exists()
    assert not (out_dir / "sample_V50_nf.jpg").exists()
