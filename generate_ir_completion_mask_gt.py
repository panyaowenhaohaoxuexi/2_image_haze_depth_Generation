"""
Generate binary infrared completion mask GT from existing Transmission_Map_GT.

The script only reads fog concentration maps and writes a separate binary mask tree.
It does not read hazy images or rerun depth estimation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


DEFAULT_SEVERITY_NAMES = ["mist", "middle", "dense"]

CONFIG = {
    # 已有雾气浓度图根目录，内部应包含 mist/middle/dense 子目录。
    "TRANSMISSION_ROOT": r"F:\Dehaze_Paper\2_Dataset\1_main_benchmark\FLIR\train\Transmission_Map_GT",
    # 红外补全掩码输出根目录，脚本会在其中生成 mist/middle/dense 子目录。
    "OUT_ROOT": r"F:\Dehaze_Paper\2_Dataset\1_main_benchmark\FLIR\train\IR_Completion_Mask_GT",
    # 需要处理的雾强子目录名称，需与 Transmission_Map_GT 下的子目录一致。
    "SEVERITY_NAMES": DEFAULT_SEVERITY_NAMES,
    # 透射率失效阈值：t_final <= T_OCC 的像素会被标为 255（红外补全区域）。
    # 0.05 对应 fogmap >= 242；0.10 对应 fogmap >= 230。值越大，补全区域越多。
    "T_OCC": 0.05,
    # 形态学开闭操作核大小，用于去除孤立噪声并填补小空洞。
    # <=0 表示关闭；偶数会自动加 1 变成奇数；值越大，掩码越平滑。
    "MORPH_KERNEL": 7,
    # 小连通域过滤面积阈值：面积小于该值的 255 区域会被删除。
    # 0 表示关闭；值越大，越容易删除零碎补全区域。
    "MIN_AREA": 64,
    # 是否覆盖已经存在的输出 mask。False 表示已有文件会跳过，不会改动。
    "OVERWRITE": False,
}


def fog_threshold_from_t_occ(t_occ: float) -> int:
    t_occ = float(t_occ)
    if not 0.0 <= t_occ <= 1.0:
        raise ValueError(f"t_occ must be in [0, 1], got {t_occ}")
    return int((1.0 - t_occ) * 255.0 + 0.5)


def normalize_morph_kernel(morph_kernel: int) -> int:
    morph_kernel = int(morph_kernel)
    if morph_kernel <= 0:
        return 0
    if morph_kernel % 2 == 0:
        return morph_kernel + 1
    return morph_kernel


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if int(min_area) <= 0:
        return mask

    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    cleaned = np.zeros_like(binary, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            cleaned[labels == label] = 255
    return cleaned.astype(np.uint8)


def build_mask_from_fogmap(
    fogmap_path: Path,
    out_path: Path,
    *,
    t_occ: float,
    morph_kernel: int,
    min_area: int,
    overwrite: bool,
) -> bool:
    if out_path.exists() and not overwrite:
        print(f"[SKIP] exists: {out_path}")
        return False

    fog_threshold = fog_threshold_from_t_occ(t_occ)
    with Image.open(fogmap_path) as image:
        fog = np.asarray(image.convert("L"), dtype=np.uint8)

    mask = (fog >= fog_threshold).astype(np.uint8) * 255

    kernel_size = normalize_morph_kernel(morph_kernel)
    if kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    mask = remove_small_components(mask, min_area)
    mask = ((mask > 0).astype(np.uint8) * 255)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(out_path)
    print(f"[INFO] saved: {out_path}")
    return True


def process_transmission_root(
    transmission_root: Path,
    out_root: Path,
    *,
    severity_names: list[str],
    t_occ: float,
    morph_kernel: int,
    min_area: int,
    overwrite: bool,
) -> int:
    fog_threshold = fog_threshold_from_t_occ(t_occ)
    morph_kernel = normalize_morph_kernel(morph_kernel)
    print(f"[INFO] t_occ={float(t_occ):.4f}, fog_threshold={fog_threshold}")
    print(f"[INFO] output root: {out_root}")

    processed_count = 0
    for severity in severity_names:
        in_dir = transmission_root / severity
        out_dir = out_root / severity
        if not in_dir.exists():
            print(f"[WARN] severity directory missing, skip: {in_dir}")
            continue
        fogmaps = sorted(path for path in in_dir.glob("*.png") if path.is_file())
        print(f"[INFO] severity={severity}, images={len(fogmaps)}")
        for fogmap_path in fogmaps:
            out_path = out_dir / fogmap_path.name
            if build_mask_from_fogmap(
                fogmap_path,
                out_path,
                t_occ=t_occ,
                morph_kernel=morph_kernel,
                min_area=min_area,
                overwrite=overwrite,
            ):
                processed_count += 1

    print(f"[INFO] processed files: {processed_count}")
    return processed_count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate IR completion mask GT from existing Transmission_Map_GT."
    )
    parser.add_argument(
        "--transmission_root",
        default=CONFIG["TRANSMISSION_ROOT"],
        help="Existing Transmission_Map_GT root containing severity subdirectories.",
    )
    parser.add_argument(
        "--out_root",
        default=CONFIG["OUT_ROOT"],
        help="Output IR_Completion_Mask_GT root.",
    )
    parser.add_argument(
        "--severity_names",
        nargs="+",
        default=CONFIG["SEVERITY_NAMES"],
        help="Severity subdirectories to process. Default: mist middle dense.",
    )
    parser.add_argument(
        "--t_occ",
        type=float,
        default=CONFIG["T_OCC"],
        help="Pixels with t_final <= t_occ are marked as IR completion regions.",
    )
    parser.add_argument(
        "--morph_kernel",
        type=int,
        default=CONFIG["MORPH_KERNEL"],
        help="Morphology kernel size. <=0 disables morphology; even values are rounded up.",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=CONFIG["MIN_AREA"],
        help="Remove connected components smaller than this area. <=0 disables filtering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=CONFIG["OVERWRITE"],
        help="Overwrite existing output masks.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    process_transmission_root(
        Path(args.transmission_root),
        Path(args.out_root),
        severity_names=list(args.severity_names),
        t_occ=args.t_occ,
        morph_kernel=args.morph_kernel,
        min_area=args.min_area,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
