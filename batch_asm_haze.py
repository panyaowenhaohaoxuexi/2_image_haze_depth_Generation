"""
基于 ASM + Depth Anything V2 + 红外天空掩膜的批量加雾工具。

物理模型：
    I(x) = J(x) * t(x) + A * (1 - t(x))
    t(x) = exp(-beta * d(x))
    beta = 3.912 / V

其中 J 为清晰图，I 为加雾图，A 为大气光，d 为米制深度，V 为气象能见度（米）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - 运行主流程时会给出明确错误
    cv2 = None

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover - 运行主流程时会给出明确错误
    ndimage = None


CONFIG = {
    "VIS_DIR": "vis",
    "IR_DIR": "ir",
    "OUT_DIR": "out_haze",
    "MODEL_NAME_OR_PATH": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "IS_METRIC_DEPTH": True,
    "VISIBILITIES": [200.0, 100.0, 50.0],
    "T_SKY": 0.07,
    "DEPTH_SCALE": 80.0,
    "DEPTH_SMOOTH_SIGMA": 9.0,
    "DARK_CHANNEL_PATCH": 15,
    "DARK_CHANNEL_TOP_PERCENT": 0.001,
    "DEBUG_VISIBILITY": 100.0,
    "DEVICE": "auto",
    "SKY_DILATE_PIXELS": 5,
    "SKY_BLUR_SIGMA": 3.0,
    "SKY_FALLBACK_PERCENTILE": 92.0,
    "ATMOSPHERIC_LIGHT_MIN": 0.8,
    "ATMOSPHERIC_LIGHT_MAX": 0.95,
    "DISABLE_SKY_MASK": False,
    "SAVE_FOGMAP": False,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def require_runtime_dependencies() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if ndimage is None:
        missing.append("scipy")
    if missing:
        raise RuntimeError(
            "缺少运行依赖: "
            + ", ".join(missing)
            + "。请先执行 `pip install -r requirements.txt`。"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch ASM haze synthesis with Depth Anything V2 and IR sky masks."
    )
    parser.add_argument("--vis_dir", default=CONFIG["VIS_DIR"], help="可见光图文件夹")
    parser.add_argument("--ir_dir", default=CONFIG["IR_DIR"], help="红外图文件夹")
    parser.add_argument("--out_dir", default=CONFIG["OUT_DIR"], help="输出文件夹")
    parser.add_argument(
        "--model_name_or_path",
        default=CONFIG["MODEL_NAME_OR_PATH"],
        help="HuggingFace 模型 ID 或本地权重目录",
    )
    parser.add_argument(
        "--is_metric_depth",
        dest="is_metric_depth",
        action="store_true",
        default=CONFIG["IS_METRIC_DEPTH"],
        help="True: metric 深度版直接输出米；False: 相对深度乘 depth_scale",
    )
    parser.add_argument(
        "--no-is_metric_depth",
        dest="is_metric_depth",
        action="store_false",
        help="使用相对深度版模型时关闭 metric 模式，启用 depth_scale 缩放",
    )
    parser.add_argument(
        "--visibilities",
        nargs="+",
        type=float,
        default=CONFIG["VISIBILITIES"],
        help="能见度档位（米），例如 200 100 50",
    )
    parser.add_argument("--t_sky", type=float, default=CONFIG["T_SKY"])
    parser.add_argument("--depth_scale", type=float, default=CONFIG["DEPTH_SCALE"])
    parser.add_argument(
        "--depth_smooth_sigma",
        type=float,
        default=CONFIG["DEPTH_SMOOTH_SIGMA"],
        help="Gaussian sigma for low-frequency haze depth; 0 disables smoothing.",
    )
    parser.add_argument(
        "--dark_channel_patch",
        type=int,
        default=CONFIG["DARK_CHANNEL_PATCH"],
        help="暗通道局部 min filter patch size",
    )
    parser.add_argument(
        "--debug_visibility",
        type=float,
        default=CONFIG["DEBUG_VISIBILITY"],
        help="保存透射率 debug 图时使用的能见度",
    )
    parser.add_argument(
        "--device",
        default=CONFIG["DEVICE"],
        choices=["auto", "cuda", "cpu"],
        help="深度推理设备",
    )
    parser.add_argument(
        "--disable_sky_mask",
        action="store_true",
        default=CONFIG["DISABLE_SKY_MASK"],
        help="不使用红外/深度天空掩膜，完全按深度透射率加雾",
    )
    parser.add_argument(
        "--save_fogmap",
        action="store_true",
        default=CONFIG["SAVE_FOGMAP"],
        help="Save single-channel uint8 fog concentration maps where 0=clear and 255=dense fog.",
    )
    return parser.parse_args()


def list_images(vis_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in vis_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_paired_ir_path(vis_path: Path, ir_dir: Path) -> Path | None:
    """默认同名配对：vis/abc.jpg -> ir/abc.jpg。

    如果你的命名是 xxx_rgb.jpg / xxx_ir.jpg，可改为：
        ir_name = vis_path.name.replace("_rgb", "_ir")
        candidate = ir_dir / ir_name
    """
    candidate = ir_dir / vis_path.name
    if candidate.exists():
        return candidate
    return None


def load_rgb_float(path: Path) -> tuple[np.ndarray, Image.Image]:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array, image


def save_rgb_float(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    uint8 = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(uint8, mode="RGB").save(path, quality=95)


def save_gray_float(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    uint8 = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(uint8, mode="L").save(path)


def save_fogmap(path: Path, t_final: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fog_map = 1.0 - np.asarray(t_final, dtype=np.float32)
    uint8 = (np.clip(fog_map, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(uint8, mode="L").save(path)


def depth_to_meters(
    raw_depth: np.ndarray,
    *,
    is_metric_depth: bool,
    depth_scale: float,
) -> tuple[np.ndarray, float]:
    """将模型输出转为米制深度。

    - metric 版 Depth Anything V2：模型输出单位已是米，depth_scale 强制视为 1.0。
    - 相对深度版 Depth Anything V2：depth_m = relative_depth * depth_scale。
    """
    raw_depth = np.asarray(raw_depth, dtype=np.float32)
    raw_depth = np.nan_to_num(raw_depth, nan=0.0, posinf=0.0, neginf=0.0)
    raw_depth = np.maximum(raw_depth, 0.0)
    if is_metric_depth:
        return raw_depth, 1.0
    return raw_depth * float(depth_scale), float(depth_scale)


def smooth_depth_for_haze(depth_m: np.ndarray, *, sigma: float) -> np.ndarray:
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if float(sigma) <= 0.0:
        return depth_m.copy()
    require_runtime_dependencies()
    return cv2.GaussianBlur(
        depth_m,
        ksize=(0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REPLICATE,
    ).astype(np.float32)


class DepthEstimator:
    def __init__(self, model_name_or_path: str, device: str) -> None:
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "缺少 torch/transformers，无法加载 Depth Anything V2。"
                "请先执行 `pip install -r requirements.txt`。"
            ) from exc

        if device == "auto":
            use_cuda = torch.cuda.is_available()
        else:
            use_cuda = device == "cuda"
        if use_cuda and not torch.cuda.is_available():
            print("[WARN] 指定 cuda 但当前不可用，自动回退 CPU。")
            use_cuda = False

        self.device_name = "cuda" if use_cuda else "cpu"
        pipeline_device = 0 if use_cuda else -1
        print(f"[INFO] Loading depth model: {model_name_or_path}")
        print(f"[INFO] Depth device: {self.device_name}")
        self.pipe = pipeline(
            task="depth-estimation",
            model=model_name_or_path,
            device=pipeline_device,
        )

    def predict(self, image: Image.Image, target_hw: tuple[int, int]) -> np.ndarray:
        result = self.pipe(image)
        depth = result.get("predicted_depth", result.get("depth"))
        if depth is None:
            raise RuntimeError("Depth pipeline 未返回 predicted_depth/depth。")

        if hasattr(depth, "detach"):
            depth_np = depth.detach().cpu().float().numpy()
        elif isinstance(depth, Image.Image):
            depth_np = np.asarray(depth, dtype=np.float32)
        else:
            depth_np = np.asarray(depth, dtype=np.float32)

        depth_np = np.squeeze(depth_np).astype(np.float32)
        height, width = target_hw
        if depth_np.shape != (height, width):
            require_runtime_dependencies()
            depth_np = cv2.resize(depth_np, (width, height), interpolation=cv2.INTER_CUBIC)
        return depth_np


def generate_ir_sky_mask_soft(
    ir_path: Path,
    target_hw: tuple[int, int],
    *,
    dilate_pixels: int = CONFIG["SKY_DILATE_PIXELS"],
    blur_sigma: float = CONFIG["SKY_BLUR_SIGMA"],
) -> np.ndarray:
    require_runtime_dependencies()
    height, width = target_hw
    ir_gray = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
    if ir_gray is None:
        raise RuntimeError(f"红外图读取失败: {ir_path}")
    if ir_gray.shape != (height, width):
        ir_gray = cv2.resize(ir_gray, (width, height), interpolation=cv2.INTER_LINEAR)

    _, binary = cv2.threshold(ir_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    class_one = binary > 0
    class_zero = ~class_one

    upper = np.zeros((height, width), dtype=bool)
    upper[: max(1, height // 2), :] = True
    sky_mask = _choose_sky_class(class_zero, class_one, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = sky_mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    filled = ndimage.binary_fill_holes(mask_u8 > 0)

    if dilate_pixels > 0:
        ksize = 2 * int(dilate_pixels) + 1
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        filled = cv2.dilate(filled.astype(np.uint8), dilate_kernel, iterations=1) > 0

    soft = cv2.GaussianBlur(
        filled.astype(np.float32),
        ksize=(0, 0),
        sigmaX=float(blur_sigma),
        sigmaY=float(blur_sigma),
    )
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def _choose_sky_class(mask_a: np.ndarray, mask_b: np.ndarray, upper: np.ndarray) -> np.ndarray:
    score_a = _upper_class_score(mask_a, upper)
    score_b = _upper_class_score(mask_b, upper)
    return mask_a if score_a >= score_b else mask_b


def _upper_class_score(mask: np.ndarray, upper: np.ndarray) -> float:
    area = float(mask.sum())
    if area <= 0.0:
        return -1.0
    upper_overlap = float((mask & upper).sum()) / area
    upper_coverage = float((mask & upper).sum()) / float(upper.sum())
    area_ratio = area / float(mask.size)
    area_penalty = abs(area_ratio - 0.35)
    return upper_overlap + 0.5 * upper_coverage - 0.25 * area_penalty


def fallback_sky_mask_from_depth(
    depth_m: np.ndarray,
    *,
    percentile: float = CONFIG["SKY_FALLBACK_PERCENTILE"],
    blur_sigma: float = CONFIG["SKY_BLUR_SIGMA"],
) -> np.ndarray:
    require_runtime_dependencies()
    threshold = np.percentile(depth_m, percentile)
    mask = depth_m >= threshold
    soft = cv2.GaussianBlur(
        mask.astype(np.float32),
        ksize=(0, 0),
        sigmaX=float(blur_sigma),
        sigmaY=float(blur_sigma),
    )
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def estimate_atmospheric_light(
    image: np.ndarray,
    sky_mask_soft: np.ndarray | None = None,
    *,
    patch_size: int = CONFIG["DARK_CHANNEL_PATCH"],
    top_percent: float = CONFIG["DARK_CHANNEL_TOP_PERCENT"],
    light_min: float = CONFIG["ATMOSPHERIC_LIGHT_MIN"],
    light_max: float = CONFIG["ATMOSPHERIC_LIGHT_MAX"],
) -> np.ndarray:
    require_runtime_dependencies()
    patch_size = max(3, int(patch_size))
    if patch_size % 2 == 0:
        patch_size += 1

    dark_base = image.min(axis=2)
    dark_channel = ndimage.minimum_filter(dark_base, size=patch_size, mode="nearest")
    flat_dark = dark_channel.reshape(-1)
    pixel_count = flat_dark.size
    top_k = max(1, int(pixel_count * float(top_percent)))

    candidate_indices = np.argpartition(flat_dark, pixel_count - top_k)[-top_k:]
    if sky_mask_soft is not None:
        sky_flat = sky_mask_soft.reshape(-1)
        sky_candidates = candidate_indices[sky_flat[candidate_indices] >= 0.5]
        if sky_candidates.size >= max(10, top_k // 20):
            candidate_indices = sky_candidates

    flat_image = image.reshape(-1, 3)
    candidate_rgb = flat_image[candidate_indices]
    luminance = candidate_rgb.mean(axis=1)
    atmospheric_light = candidate_rgb[int(np.argmax(luminance))]
    return np.clip(atmospheric_light, light_min, light_max).astype(np.float32)


def apply_atmospheric_scattering(
    image: np.ndarray,
    depth_m: np.ndarray,
    sky_mask_soft: np.ndarray,
    atmospheric_light: np.ndarray,
    *,
    visibility_m: float,
    t_sky: float,
) -> tuple[np.ndarray, np.ndarray]:
    """严格使用 ASM 合成雾图。

    beta = 3.912 / V，t_depth = exp(-beta * d)。
    天空区使用 t_final = (1 - sky_mask_soft) * t_depth + sky_mask_soft * t_sky。
    """
    beta = 3.912 / float(visibility_m)
    t_depth = np.exp(-beta * depth_m).astype(np.float32)
    sky = np.clip(sky_mask_soft.astype(np.float32), 0.0, 1.0)
    t_final = (1.0 - sky) * t_depth + sky * float(t_sky)
    t_final = np.clip(t_final, 0.0, 1.0).astype(np.float32)
    hazy = image * t_final[..., None] + atmospheric_light.reshape(1, 1, 3) * (
        1.0 - t_final[..., None]
    )
    return np.clip(hazy, 0.0, 1.0).astype(np.float32), t_final


def normalize_for_debug(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    low = float(np.percentile(image, 2.0))
    high = float(np.percentile(image, 98.0))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def visibility_tag(visibility: float) -> str:
    if float(visibility).is_integer():
        return f"V{int(visibility)}"
    return f"V{str(visibility).replace('.', 'p')}"


def process_one_image(
    *,
    vis_path: Path,
    ir_dir: Path,
    out_dir: Path,
    debug_dir: Path,
    fogmap_dir: Path,
    depth_estimator: DepthEstimator,
    args: argparse.Namespace,
) -> None:
    image, pil_image = load_rgb_float(vis_path)
    height, width = image.shape[:2]

    raw_depth = depth_estimator.predict(pil_image, (height, width))
    depth_m, effective_scale = depth_to_meters(
        raw_depth,
        is_metric_depth=args.is_metric_depth,
        depth_scale=args.depth_scale,
    )
    depth_for_haze = smooth_depth_for_haze(depth_m, sigma=args.depth_smooth_sigma)
    if args.is_metric_depth:
        print("[INFO] metric depth: depth_scale=1.0 effective")
    else:
        print(f"[INFO] relative depth: depth_m = relative_depth * depth_scale ({effective_scale})")

    if args.disable_sky_mask:
        print("[INFO] sky mask disabled: 使用纯深度透射率，不强制天空 t_sky。")
        sky_mask_soft = np.zeros((height, width), dtype=np.float32)
    else:
        ir_path = find_paired_ir_path(vis_path, ir_dir)
        if ir_path is None:
            print(f"[WARN] 找不到配对红外图: {vis_path.name}，回退为纯深度阈值天空处理。")
            sky_mask_soft = fallback_sky_mask_from_depth(depth_for_haze)
        else:
            try:
                sky_mask_soft = generate_ir_sky_mask_soft(ir_path, (height, width))
            except Exception as exc:
                print(f"[WARN] 红外天空掩膜失败: {ir_path} ({exc})，回退为纯深度阈值天空处理。")
                sky_mask_soft = fallback_sky_mask_from_depth(depth_for_haze)

    atmospheric_light = estimate_atmospheric_light(
        image,
        sky_mask_soft,
        patch_size=args.dark_channel_patch,
    )
    print(
        "[INFO] A="
        f"{np.round(atmospheric_light, 4).tolist()}, "
        f"depth_m range=({depth_for_haze.min():.3f}, {depth_for_haze.max():.3f})"
    )

    stem = vis_path.stem
    save_gray_float(debug_dir / f"{stem}_depth.png", normalize_for_debug(depth_for_haze))
    save_gray_float(debug_dir / f"{stem}_sky_mask_soft.png", sky_mask_soft)

    debug_t_saved = False
    for visibility in args.visibilities:
        hazy, t_final = apply_atmospheric_scattering(
            image,
            depth_for_haze,
            sky_mask_soft,
            atmospheric_light,
            visibility_m=visibility,
            t_sky=args.t_sky,
        )
        tag = visibility_tag(visibility)
        out_path = out_dir / f"{stem}_{tag}.jpg"
        save_rgb_float(out_path, hazy)
        print(f"[INFO] saved: {out_path}")
        if args.save_fogmap:
            save_fogmap(fogmap_dir / f"{stem}_{tag}_fogmap.png", t_final)

        if not debug_t_saved and abs(float(visibility) - float(args.debug_visibility)) < 1e-6:
            save_gray_float(debug_dir / f"{stem}_{tag}_t_final.png", t_final)
            debug_t_saved = True

    if not debug_t_saved:
        visibility = float(args.visibilities[min(1, len(args.visibilities) - 1)])
        _, t_final = apply_atmospheric_scattering(
            image,
            depth_for_haze,
            sky_mask_soft,
            atmospheric_light,
            visibility_m=visibility,
            t_sky=args.t_sky,
        )
        save_gray_float(debug_dir / f"{stem}_{visibility_tag(visibility)}_t_final.png", t_final)

def main() -> None:
    args = parse_args()
    require_runtime_dependencies()

    vis_dir = Path(args.vis_dir)
    ir_dir = Path(args.ir_dir)
    out_dir = Path(args.out_dir)
    debug_dir = out_dir / "debug"
    fogmap_dir = out_dir / "fogmap"
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    fogmap_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(vis_dir)
    if not images:
        print(f"[WARN] 未找到输入图像: {vis_dir}")
        return

    depth_estimator = DepthEstimator(args.model_name_or_path, args.device)
    total = len(images)
    for index, vis_path in enumerate(images, start=1):
        print(f"\n[{index}/{total}] {vis_path.name}")
        try:
            process_one_image(
                vis_path=vis_path,
                ir_dir=ir_dir,
                out_dir=out_dir,
                debug_dir=debug_dir,
                fogmap_dir=fogmap_dir,
                depth_estimator=depth_estimator,
                args=args,
            )
        except Exception as exc:
            print(f"[ERROR] 处理失败，跳过: {vis_path} ({exc})")


if __name__ == "__main__":
    main()
