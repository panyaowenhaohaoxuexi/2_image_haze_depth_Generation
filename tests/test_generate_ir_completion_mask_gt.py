import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_ir_completion_mask_gt import (
    build_mask_from_fogmap,
    fog_threshold_from_t_occ,
    parse_args,
    process_transmission_root,
)


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        assert image.mode == "L"
        return np.asarray(image)


def write_gray(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def test_fog_threshold_from_t_occ():
    assert fog_threshold_from_t_occ(0.05) == 242
    assert fog_threshold_from_t_occ(0.10) == 230


def test_t_occ_validation():
    with pytest.raises(ValueError):
        fog_threshold_from_t_occ(-0.1)
    with pytest.raises(ValueError):
        fog_threshold_from_t_occ(1.1)


def test_parse_args_uses_code_defaults():
    args = parse_args([])

    assert str(args.transmission_root).endswith("Transmission_Map_GT")
    assert str(args.out_root).endswith("IR_Completion_Mask_GT")
    assert args.severity_names == ["mist", "middle", "dense"]
    assert args.t_occ == 0.05
    assert args.morph_kernel == 7
    assert args.min_area == 64
    assert args.overwrite is False


def test_build_mask_from_fogmap_binary_values(tmp_path):
    fogmap_path = tmp_path / "Transmission_Map_GT" / "mist" / "a.png"
    out_path = tmp_path / "IR_Completion_Mask_GT" / "mist" / "a.png"
    write_gray(
        fogmap_path,
        np.array(
            [
                [0, 229],
                [230, 255],
            ],
            dtype=np.uint8,
        ),
    )

    wrote = build_mask_from_fogmap(
        fogmap_path,
        out_path,
        t_occ=0.10,
        morph_kernel=0,
        min_area=0,
        overwrite=False,
    )

    assert wrote is True
    mask = read_gray(out_path)
    np.testing.assert_array_equal(
        mask,
        np.array(
            [
                [0, 0],
                [255, 255],
            ],
            dtype=np.uint8,
        ),
    )
    assert set(np.unique(mask).tolist()).issubset({0, 255})


def test_no_overwrite_by_default(tmp_path):
    fogmap_path = tmp_path / "Transmission_Map_GT" / "mist" / "a.png"
    out_path = tmp_path / "IR_Completion_Mask_GT" / "mist" / "a.png"
    write_gray(fogmap_path, np.full((2, 2), 255, dtype=np.uint8))
    write_gray(out_path, np.full((2, 2), 123, dtype=np.uint8))

    wrote = build_mask_from_fogmap(
        fogmap_path,
        out_path,
        t_occ=0.05,
        morph_kernel=0,
        min_area=0,
        overwrite=False,
    )

    assert wrote is False
    saved = read_gray(out_path)
    assert np.all(saved == 123)


def test_binary_values_after_postprocessing(tmp_path):
    fogmap_path = tmp_path / "Transmission_Map_GT" / "dense" / "a.png"
    out_path = tmp_path / "IR_Completion_Mask_GT" / "dense" / "a.png"
    fog = np.zeros((9, 9), dtype=np.uint8)
    fog[2:7, 2:7] = 255
    fog[4, 4] = 0
    fog[0, 0] = 255
    write_gray(fogmap_path, fog)

    wrote = build_mask_from_fogmap(
        fogmap_path,
        out_path,
        t_occ=0.05,
        morph_kernel=3,
        min_area=4,
        overwrite=False,
    )

    assert wrote is True
    mask = read_gray(out_path)
    assert set(np.unique(mask).tolist()).issubset({0, 255})


def test_process_severity_dirs(tmp_path):
    transmission_root = tmp_path / "Transmission_Map_GT"
    out_root = tmp_path / "IR_Completion_Mask_GT"
    write_gray(transmission_root / "mist" / "a.png", np.full((2, 2), 255, dtype=np.uint8))
    write_gray(transmission_root / "dense" / "a.png", np.full((2, 2), 255, dtype=np.uint8))

    processed = process_transmission_root(
        transmission_root,
        out_root,
        severity_names=["mist", "middle", "dense"],
        t_occ=0.05,
        morph_kernel=0,
        min_area=0,
        overwrite=False,
    )

    assert processed == 2
    assert (out_root / "mist" / "a.png").exists()
    assert (out_root / "dense" / "a.png").exists()
    assert not (out_root / "middle" / "a.png").exists()
