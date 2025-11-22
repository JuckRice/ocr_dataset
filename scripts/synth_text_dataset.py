#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SynthText Dataset Generator
==============================================

What this script does
---------------------
- Randomly samples background images from a local directory (e.g., ImageNet subset) and crops them to a target resolution.
- Places 1–8 text instances per image (truncated Poisson with λ≈3 to avoid monotonous density).
- Generates alphanumeric strings only: [A–Z a–z 0–9] with the length buckets {1–4:30%, 5–8:40%, 9–12:25%, >12:5%}.
- Typography: collects fonts from a directory (ttf/otf/ttc/otc), randomizes size, letter-spacing, italic shear,
  stroke (fake "weight"), and optional shadow.
- Colors: samples text colors from background patches and targets one of three contrast bands (high/medium/low). Adds a
  1–2 px stroke for low contrast to maintain visibility.
- Geometry & degradations: rotation, perspective, sine-curve (curved baseline), optional occlusion; motion/defocus blur,
  noise, JPEG compression, exposure/gamma, low-res down-up sampling.
- Compositing: alpha/screen/Poisson (OpenCV seamlessClone).
- Quality gating: enforces effective contrast, visible ratio, and minimum character height (≥8 px by default).
- Annotation: supports L1/L2/L3 granularity with polygons/bboxes/text/reading order/difficulty and per-character polygons.
- Output: final image (.jpg) + JSON per image or JSONL; coordinates are provided in both absolute pixels and normalized 0–1.

Notes
-----
1) Backgrounds from ImageNet or similar are generally research-use only. Check licenses before redistribution.
2) If no backgrounds or fonts are found, the script falls back to gradient backgrounds and PIL's default font to validate the pipeline.

Dependencies
------------
pip install -r requirements.txt
- numpy, pillow, opencv-python, tqdm

Example
-------
python synth_text_dataset_en.py \
  --backgrounds /path/to/imagenet_subset \
  --fonts /path/to/fonts_dir \
  --out_dir ./out \
  --num_images 1000 \
  --res 1024 1024 \
  --jsonl
"""
import argparse
import json
import math
import os
import random
import sys
import multiprocessing
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import cv2
from tqdm import tqdm


# -------------------------- Utilities --------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def truncated_poisson(lam: float = 3.0, lo: int = 1, hi: int = 8) -> int:
    """Sample from a truncated Poisson distribution over [lo, hi]."""
    ks = np.arange(lo, hi + 1)
    pmf = np.exp(-lam) * np.power(lam, ks) / np.array([math.factorial(k) for k in ks], dtype=np.float64)
    pmf /= pmf.sum()
    return int(np.random.choice(ks, p=pmf))


ALPHANUM = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'


def sample_text_length() -> int:
    """Bucketed length distribution: {1–4:30%, 5–8:40%, 9–12:25%, >12:5%} (>12 => 13–24 uniform)."""
    r = random.random()
    if r < 0.30:
        L = random.randint(1, 4)
    elif r < 0.70:
        L = random.randint(5, 8)
    elif r < 0.95:
        L = random.randint(9, 12)
    else:
        L = random.randint(13, 24)
    return L


def sample_string() -> str:
    """Return a random alphanumeric string with the specified length distribution."""
    L = sample_text_length()
    return ''.join(random.choice(ALPHANUM) for _ in range(L))


def list_files_with_ext(root: Path, exts=('.jpg', '.jpeg', '.png', '.bmp', '.webp')) -> List[Path]:
    """List files under `root` that match the given extensions (case-insensitive)."""
    if not root or not root.exists():
        return []
    files = []
    for ext in exts:
        files.extend(root.rglob(f'*{ext}'))
    return files


def list_fonts(font_dir: Optional[Path]) -> List[Path]:
    """List font files under `font_dir` (ttf/otf/ttc/otc). Falls back to PIL default if none found."""
    sys_fonts = []
    if font_dir and font_dir.exists():
        for ext in ('.ttf', '.otf', '.ttc', '.otc'):
            sys_fonts.extend(font_dir.rglob(f'*{ext}'))
    # Fallback: PIL default font (ASCII-friendly, low quality; for pipeline tests only)
    if not sys_fonts:
        try:
            _ = ImageFont.load_default()
        except Exception:
            pass
    return sys_fonts


def ensure_min_side(img: Image.Image, min_short_side: int) -> Image.Image:
    """Upscale an image so that its shorter side >= min_short_side (keeps aspect ratio)."""
    w, h = img.size
    short = min(w, h)
    if short >= min_short_side:
        return img
    scale = min_short_side / short
    nw, nh = int(w * scale), int(h * scale)
    return img.resize((nw, nh), Image.LANCZOS)


def random_crop_to_res(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Random crop (with upscaling if needed) to reach target resolution."""
    w, h = img.size
    if w == target_w and h == target_h:
        return img
    if w < target_w or h < target_h:
        scale = max(target_w / w, target_h / h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        w, h = img.size
    x = random.randint(0, w - target_w)
    y = random.randint(0, h - target_h)
    return img.crop((x, y, x + target_w, y + target_h))


def rgb_to_lum(img_np: np.ndarray) -> np.ndarray:
    """Convert sRGB uint8 (RGB) to linear luminance Y in [0,1]."""
    r = img_np[..., 0]
    g = img_np[..., 1]
    b = img_np[..., 2]
    def to_lin(x):
        x = x / 255.0
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    r_lin = to_lin(r.astype(np.float32))
    g_lin = to_lin(g.astype(np.float32))
    b_lin = to_lin(b.astype(np.float32))
    Y = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
    return Y


def pick_text_color(bg_rgb: np.ndarray, contrast_level: str = 'medium') -> Tuple[Tuple[int, int, int], str]:
    """
    Pick a text color based on local background luminance aimed at three contrast bands.
    contrast_level: 'high' | 'medium' | 'low'
    Target ΔY in linear luminance domain: high≈0.50, medium≈0.30, low≈0.15
    """
    Y = rgb_to_lum(bg_rgb)
    bg_mean = float(np.mean(Y))
    deltas = {'high': 0.50, 'medium': 0.30, 'low': 0.15}
    target_delta = deltas.get(contrast_level, 0.30)
    # Randomly push brighter or darker
    if random.random() < 0.5:
        Yt = min(1.0, bg_mean + target_delta)
    else:
        Yt = max(0.0, bg_mean - target_delta)

    # Create an HSV color and approximate luminance by using V≈Yt
    import colorsys
    h = random.random()
    s = random.uniform(0.2, 0.8)
    v = Yt
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    rgb = (int(r * 255), int(g * 255), int(b * 255))
    return rgb, contrast_level


def ensure_outline_for_low_contrast(fill_rgb: Tuple[int, int, int], contrast_level: str) -> int:
    """Return recommended outline width (px) based on contrast band."""
    if contrast_level == 'low':
        return random.choice([1, 2])
    if contrast_level == 'medium':
        return random.choice([0, 1])
    return 0


def iou_aabb(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Axis-aligned IoU for two boxes (x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / area if area > 0 else 0.0


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def to_bbox_from_poly(poly: List[Tuple[float, float]]) -> Tuple[int, int, int, int]:
    """Bounding AABB from a 4-point polygon."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def normalize_poly(poly: List[Tuple[float, float]], W: int, H: int) -> List[Tuple[float, float]]:
    """Normalize polygon coordinates by image size to [0,1]."""
    return [(clamp01(x / W), clamp01(y / H)) for x, y in poly]


def compute_angle_deg(poly: List[Tuple[float, float]]) -> float:
    """Estimate orientation by the top edge vector angle (poly[0]→poly[1])."""
    (x1, y1), (x2, y2) = poly[0], poly[1]
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return float(ang)


# -------------------------- Text rendering (PIL) --------------------------

@dataclass
class RenderParams:
    font_path: str
    font_size_px: int
    italic_shear: float   # shear angle in degrees (x-direction)
    letter_spacing_px: int
    stroke_px: int
    fill_rgb: Tuple[int, int, int]
    stroke_rgb: Tuple[int, int, int]
    shadow_px: int
    curve_amp_px: float   # sine baseline amplitude in px
    curve_freq: float     # cycles over width (0.5~2)
    curve_phase: float    # phase in [0, 2π)


def load_font(path: Optional[Path], size: int) -> ImageFont.FreeTypeFont:
    """Load a font with given size; fallback to PIL default font if unavailable."""
    if path and Path(path).exists():
        try:
            # return ImageFont.truetype(str(path), size=size, layout_engine=ImageFont.LAYOUT_BASIC)
            return ImageFont.truetype(str(path), size=size)
        except Exception as e:
            pass
            # print(f"[Debug Font Load Error] Failed to load font: {path}. Error: {e}")
    return ImageFont.load_default()


def render_text_mask(text: str, params: RenderParams) -> Tuple[Image.Image, List[List[Tuple[float, float]]]]:
    """
    Render text onto an RGBA canvas (alpha used), returning the image and per-character polygons
    in the local canvas coordinate system.

    We draw per character to implement letter-spacing and an outline (stroke) to mimic boldness.
    """
    font = load_font(Path(params.font_path) if params.font_path else None, params.font_size_px)
    # Generous canvas size estimate
    est_w = int((params.font_size_px * 0.65 + params.letter_spacing_px) * max(1, len(text)) + 20)
    est_h = int(params.font_size_px * 2.5)
    canvas = Image.new('RGBA', (est_w, est_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    x, y = 10, int(params.font_size_px * 0.7)
    char_polys = []
    for ch in text:
        try:
            # --- 尝试渲染 ---
            bbox = font.getbbox(ch)  # (x0,y0,x1,y1) in font metrics
            ch_w = bbox[2] - bbox[0]
            ch_h = bbox[3] - bbox[1]
            if params.stroke_px > 0:
                draw.text((x, y), ch, font=font, fill=params.stroke_rgb + (255,), stroke_width=params.stroke_px, stroke_fill=params.stroke_rgb + (255,))
            draw.text((x, y), ch, font=font, fill=params.fill_rgb + (255,), stroke_width=0)
            
            # --- 仅在成功时才添加多边形和移动x ---
            poly = [(x, y + bbox[1]), (x + ch_w, y + bbox[1]), (x + ch_w, y + bbox[3]), (x, y + bbox[3])]
            char_polys.append(poly)
            x += ch_w + params.letter_spacing_px
            
        except OSError as e:
            print(f" [Warn] font {Path(params.font_path).name} error when processing '{ch}' glyph, skipping character.")
            pass


    # Optional drop shadow (simple offset + Gaussian)
    if params.shadow_px > 0:
        try:
            shadow = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
            sdraw = ImageDraw.Draw(shadow)
            sx, sy = 10 + params.shadow_px, int(params.font_size_px * 0.7) + params.shadow_px
            sdraw.text((sx, sy), text, font=font, fill=(0, 0, 0, 180))
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=params.shadow_px * 0.5))
            canvas = Image.alpha_composite(shadow, canvas)
        except OSError as e:
            print(f"[Warn] {Path(params.font_path).name}  {e} when processing shadow, skipping shadow.")
            pass # 'canvas' remains unchanged

    # Trim transparent margins
    bbox = canvas.getbbox()
    if bbox:
        canvas = canvas.crop(bbox)
        dx, dy = bbox[0], bbox[1]
        char_polys = [[(px - dx, py - dy) for (px, py) in poly] for poly in char_polys]

    # Italic shear via affine transform (x-direction)
    if abs(params.italic_shear) > 1e-3:
        shear = math.tan(math.radians(params.italic_shear))
        w, h = canvas.size
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        new_w = int(w + abs(shear) * h)
        canvas = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGBA2BGRA)
        canvas = cv2.warpAffine(canvas, M, (new_w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA))
        char_polys = [[(px + shear * py, py) for (px, py) in poly] for poly in char_polys]

    # Curved baseline (sine)
    if params.curve_amp_px != 0.0:
        canvas, char_polys = sine_curve_warp(canvas, char_polys, params.curve_amp_px, params.curve_freq, params.curve_phase)

    return canvas, char_polys


def sine_curve_warp(img_rgba: Image.Image, char_polys: List[List[Tuple[float, float]]],
                    amp_px: float, freq: float, phase: float):
    """Apply vertical sine-wave displacement: y' = y + A*sin(2π*freq*(x/W) + phase)."""
    w, h = img_rgba.size
    src = np.array(img_rgba)
    dst = np.zeros_like(src)
    for x in range(w):
        shift = int(round(amp_px * math.sin(2 * math.pi * freq * (x / max(1, w)) + phase)))
        if shift >= 0:
            # --- SHIFT DOWN ---
            # Calculate how many pixels to copy
            copy_len = h - shift
            if copy_len > 0:
                # Only copy if the length is positive
                # Copy top part of src to bottom part of dst
                dst[shift:h, x] = src[0:copy_len, x]
        else:
            # --- SHIFT UP --- (shift is negative)
            # Calculate how many pixels to copy
            copy_len = h + shift
            if copy_len > 0:
                # Only copy if the length is positive
                # Copy bottom part of src to top part of dst
                dst[0:copy_len, x] = src[-shift:h, x]
    def map_point(px, py):
        dy = amp_px * math.sin(2 * math.pi * freq * (px / max(1, w)) + phase)
        return (px, py + dy)
    new_char_polys = [[map_point(px, py) for (px, py) in poly] for poly in char_polys]
    out = Image.fromarray(dst)
    return out, new_char_polys


# -------------------------- Geometry & compositing --------------------------

def random_geometric_params(max_rotate=30.0, perspective_strength=0.25) -> Dict[str, Any]:
    """Sample lightweight rotation and perspective jitter parameters."""
    rot = random.uniform(-max_rotate, max_rotate)
    ps = perspective_strength
    return {
        'rotate_deg': rot,
        'perspective': [random.uniform(-ps, ps) for _ in range(8)],
    }


def apply_perspective_and_rotate(fg_rgba: Image.Image, geo: Dict[str, Any]) -> Tuple[Image.Image, np.ndarray]:
    """
    Apply rotation + perspective to a foreground RGBA text patch.
    Returns the transformed image and the 3x3 homography matrix H
    (mapping local canvas coordinates -> transformed canvas coordinates).
    """
    w, h = fg_rgba.size
    rot = geo.get('rotate_deg', 0.0)
    img_rot = fg_rgba.rotate(rot, expand=True, resample=Image.BICUBIC)
    w2, h2 = img_rot.size

    # Perspective: source corners -> jittered destination corners
    src = np.float32([[0, 0], [w2 - 1, 0], [w2 - 1, h2 - 1], [0, h2 - 1]])
    dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4 = geo.get('perspective', [0]*8)
    dst = np.float32([
        [0 + dx1 * w2, 0 + dy1 * h2],
        [w2 - 1 + dx2 * w2, 0 + dy2 * h2],
        [w2 - 1 + dx3 * w2, h2 - 1 + dy3 * h2],
        [0 + dx4 * w2, h2 - 1 + dy4 * h2],
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(cv2.cvtColor(np.array(img_rot), cv2.COLOR_RGBA2BGRA), M, (w2, h2),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    out_img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGRA2RGBA))
    return out_img, M


def transform_points(H: np.ndarray, pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Apply homography H to a list of 2D points (x,y)."""
    P = np.array([[x, y, 1.0] for (x, y) in pts], dtype=np.float32).T  # 3xN
    Q = H @ P  # 3xN
    qx = Q[0, :] / (Q[2, :] + 1e-8)
    qy = Q[1, :] / (Q[2, :] + 1e-8)
    return [(float(x), float(y)) for x, y in zip(qx, qy)]


def poisson_blend(bg_bgr: np.ndarray, fg_bgr: np.ndarray, mask: np.ndarray, x: int, y: int) -> np.ndarray:
    """Poisson seamlessClone centered at (x+w/2, y+h/2); falls back to alpha blend if it fails."""
    H, W = bg_bgr.shape[:2]
    h, w = fg_bgr.shape[:2]
    center = (int(x + w / 2), int(y + h / 2))
    try:
        blended = cv2.seamlessClone(fg_bgr, bg_bgr, mask, center, cv2.MIXED_CLONE)
        return blended
    except Exception:
        return bg_bgr


def screen_blend(bg: np.ndarray, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Screen blend mode: 1 - (1 - A)*(1 - B), mixed by alpha."""
    bg_f = bg.astype(np.float32) / 255.0
    fg_f = fg.astype(np.float32) / 255.0
    a = alpha.astype(np.float32) / 255.0
    comp = 1.0 - (1.0 - bg_f) * (1.0 - fg_f)
    out = bg_f * (1 - a[..., None]) + comp * a[..., None]
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def alpha_blend(bg: np.ndarray, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Standard alpha composite."""
    bg_f = bg.astype(np.float32)
    fg_f = fg.astype(np.float32)
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    out = bg_f * (1 - a) + fg_f * a
    return np.clip(out, 0, 255).astype(np.uint8)


# -------------------------- Degradations --------------------------

def motion_blur(img: np.ndarray, ksize: int = 7, angle: float = 0.0) -> np.ndarray:
    """Approximate motion blur with a rotated 1D kernel."""
    k = np.zeros((ksize, ksize), dtype=np.float32)
    k[ksize // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, M, (ksize, ksize))
    k /= (k.sum() + 1e-8)
    return cv2.filter2D(img, -1, k)


def add_noise(img: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Gaussian noise with stddev `sigma` in pixel space."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg_compress(img: np.ndarray, quality: int = 75) -> np.ndarray:
    """Round-trip JPEG encoding at `quality` level."""
    enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])[1]
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec


def adjust_exposure_gamma(img: np.ndarray, exposure: float = 0.0, gamma: float = 1.0) -> np.ndarray:
    """Exposure offset [-0.3,0.3] + gamma [0.7,1.5]."""
    img_f = img.astype(np.float32) / 255.0
    img_f = np.clip(img_f + exposure, 0, 1)
    img_f = np.power(img_f, 1.0 / gamma)
    return (img_f * 255.0).astype(np.uint8)


def lowres_resample(img: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Downscale-then-upscale to simulate low-resolution sampling artifacts."""
    H, W = img.shape[:2]
    newW, newH = max(1, int(W * scale)), max(1, int(H * scale))
    small = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (W, H), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST]))
    return back


# -------------------------- Quality metrics --------------------------

def effective_contrast(final_bgr: np.ndarray, mask: np.ndarray) -> float:
    """
    Difference in linear luminance between text region and a ring-shaped neighborhood (dilated minus core).
    Returns a scalar in [0,1].
    """
    if mask.max() == 0:
        return 0.0
    Y = rgb_to_lum(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))
    m = (mask > 128).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    ring = cv2.dilate(m, kernel, iterations=2) - m
    if ring.sum() < 10:
        return 0.0
    text_mean = float(Y[m.astype(bool)].mean()) if m.sum() > 0 else 0.0
    ring_mean = float(Y[ring.astype(bool)].mean())
    return abs(text_mean - ring_mean)


def visible_ratio(mask_after: np.ndarray) -> float:
    """Fraction of visible text mask pixels over the entire image (strict)."""
    m = (mask_after > 128).astype(np.float32)
    return float(m.mean())


def min_char_height_px(char_polys_img: List[List[Tuple[float, float]]]) -> float:
    """Minimum character height across all per-character polys of an instance."""
    if not char_polys_img:
        return 0.0
    hs = []
    for poly in char_polys_img:
        ys = [p[1] for p in poly]
        hs.append(max(ys) - min(ys))
    return float(min(hs)) if hs else 0.0


# -------------------------- Multiprocessing --------------------------

_GLOBAL_GENERATOR_CONFIG = None  # 用于存储配置的全局变量

def _init_worker(cfg: 'GenConfig'):
    """
    初始化 worker 进程。
    将 GenConfig 存储在子进程的全局变量中。
    """
    global _GLOBAL_GENERATOR_CONFIG
    _GLOBAL_GENERATOR_CONFIG = cfg

def _generate_task(idx: int) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    单个 worker 进程执行的任务 (生成一张图)。
    """
    global _GLOBAL_GENERATOR_CONFIG
    if _GLOBAL_GENERATOR_CONFIG is None:
        print("错误：Worker 进程未正确初始化。")
        return None
    
    # 1. 在子进程中重新创建生成器实例
    # 这是必须的，因为 'self' 实例不能被序列化传递
    gen = SynthTextDatasetGenerator(_GLOBAL_GENERATOR_CONFIG)
    
    # 2. 为此特定任务设置唯一的随机种子
    # (这是为了保证可复现性 和 任务独立性)
    set_seed(_GLOBAL_GENERATOR_CONFIG.seed + idx)
    
    # 3. 执行单个图像的生成
    return gen.generate_one(idx, level=2)


# -------------------------- Main generator --------------------------

@dataclass
class GenConfig:
    backgrounds_dir: str
    fonts_dir: Optional[str]
    out_dir: str
    width: int = 640
    height: int = 640
    num_images: int = 100
    seed: int = 123
    per_image_retry: int = 8
    per_instance_retry: int = 8
    min_char_height_px: int = 8
    save_jsonl: bool = False
    split: str = "train"
    ood_font_frac: float = 0.1  # placeholder: reserve some fonts for OOD if you implement a split strategy


class SynthTextDatasetGenerator:
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg

        self.bg_files = list_files_with_ext(Path(cfg.backgrounds_dir)) if cfg.backgrounds_dir else []
        self.font_files = list_fonts(Path(cfg.fonts_dir) if cfg.fonts_dir else None)

        Path(cfg.out_dir, 'images').mkdir(parents=True, exist_ok=True)
        Path(cfg.out_dir, 'labels').mkdir(parents=True, exist_ok=True)

        if not self.bg_files:
            print('[warn] No backgrounds found; using gradient backgrounds for pipeline sanity check.')
        if not self.font_files:
            print('[warn] No fonts found; falling back to PIL default font (for testing only).')

    def _load_random_bg(self) -> Image.Image:
        """Load a random background; fallback to gradient background if none available."""
        if self.bg_files:
            img = Image.open(random.choice(self.bg_files)).convert('RGB')
        else:
            W, H = self.cfg.width, self.cfg.height
            grad = np.linspace(0, 255, W, dtype=np.uint8)[None, :].repeat(H, axis=0)
            col = np.stack([grad, np.flipud(grad), np.roll(grad, random.randint(0, W-1), axis=1)], axis=-1)
            img = Image.fromarray(col, mode='RGB')
        img = ensure_min_side(img, min(self.cfg.width, self.cfg.height))
        img = random_crop_to_res(img, self.cfg.width, self.cfg.height)
        return img

    def _sample_render_params(self, bg_patch_rgb: np.ndarray, font_path: Optional[str]) -> RenderParams:
        """Sample text rendering parameters conditioned on a local background patch."""
        contrast_band = random.choices(['high', 'medium', 'low'], weights=[0.35, 0.45, 0.20])[0]
        fill_rgb, _ = pick_text_color(bg_patch_rgb, contrast_band)
        stroke_px = ensure_outline_for_low_contrast(fill_rgb, contrast_band)
        stroke_rgb = (0, 0, 0) if np.mean(fill_rgb) > 127 else (255, 255, 255)
        font_size = random.randint(int(self.cfg.height * 0.02), int(self.cfg.height * 0.12))
        italic_shear = random.uniform(-12, 12)
        letter_spacing = random.randint(0, max(1, font_size // 12))
        shadow_px = random.choice([0, 0, 1, 2])
        curve_amp = random.choice([0.0, 0.0, random.uniform(2.0, font_size * 0.2)])
        curve_freq = random.uniform(0.5, 2.0)
        curve_phase = random.uniform(0, 2 * math.pi)
        return RenderParams(
            font_path=font_path or '',
            font_size_px=font_size,
            italic_shear=italic_shear,
            letter_spacing_px=letter_spacing,
            stroke_px=stroke_px,
            fill_rgb=fill_rgb,
            stroke_rgb=stroke_rgb,
            shadow_px=shadow_px,
            curve_amp_px=curve_amp,
            curve_freq=curve_freq,
            curve_phase=curve_phase
        )

    def _place_non_overlapping(self, W: int, H: int, w: int, h: int, exist_bboxes: List[Tuple[int, int, int, int]], iou_th=0.2, max_try=50):
        """Find a placement (x,y) whose AABB IoU against existing instances is <= iou_th."""
        for _ in range(max_try):
            x = random.randint(0, max(0, W - w))
            y = random.randint(0, max(0, H - h))
            cand = (x, y, x + w, y + h)
            ok = all(iou_aabb(cand, b) <= iou_th for b in exist_bboxes)
            if ok:
                return x, y
        return None

    def _difficulty_tags(self, blur_sigma: float, persp_strength: float, contrast_band: str) -> Dict[str, str]:
        """Map numeric parameters to coarse difficulty tags."""
        blur_tag = 'none' if blur_sigma < 0.5 else ('low' if blur_sigma < 1.5 else ('medium' if blur_sigma < 2.5 else 'high'))
        persp_tag = 'none' if persp_strength < 0.05 else ('moderate' if persp_strength < 0.15 else 'strong')
        return {
            'blur': blur_tag,
            'perspective': persp_tag,
            'contrast': contrast_band
        }

    def _build_instance(self, bg_img: Image.Image, text: str, exist_bboxes: List[Tuple[int, int, int, int]]) -> Optional[Dict[str, Any]]:
        """Render a single text instance, composite onto the background, and compute metadata + quality metrics."""
        W, H = bg_img.size
        # Condition color on a small random background patch
        x0, y0 = random.randint(0, W - 1), random.randint(0, H - 1)
        patch = np.array(bg_img)[max(0, y0 - 32):min(H, y0 + 32), max(0, x0 - 64):min(W, x0 + 64)]
        if patch.size == 0:
            patch = np.array(bg_img)
        params = self._sample_render_params(patch, random.choice(self.font_files).as_posix() if self.font_files else None)

        # Render local RGBA text patch
        text_rgba, char_polys_local = render_text_mask(text, params)

        # Apply geometry (rotation + perspective)
        geo = random_geometric_params(max_rotate=30.0, perspective_strength=0.20)
        text_rgba2, H_geo = apply_perspective_and_rotate(text_rgba, geo)

        # Choose placement avoiding heavy overlap
        w2, h2 = text_rgba2.size
        pos = self._place_non_overlapping(W, H, w2, h2, exist_bboxes, iou_th=0.2)
        if pos is None:
            return None
        x, y = pos

        # Instance polygon (corners) after geometry + translation
        src_corners = [(0, 0), (w2 - 1, 0), (w2 - 1, h2 - 1), (0, h2 - 1)]
        poly_after = transform_points(H_geo, src_corners)
        poly_after = [(px + x, py + y) for (px, py) in poly_after]
        bbox = to_bbox_from_poly(poly_after)

        # Per-character polygons after geometry + translation
        char_polys_after = []
        for poly in char_polys_local:
            p2 = transform_points(H_geo, poly)
            p2 = [(px + x, py + y) for (px, py) in p2]
            char_polys_after.append(p2)

        # Choose a blending mode
        mode = random.choices(['alpha', 'screen', 'poisson'], weights=[0.5, 0.3, 0.2])[0]

        # Composite
        bg_bgr = cv2.cvtColor(np.array(bg_img), cv2.COLOR_RGB2BGR)
        fg_bgra = cv2.cvtColor(np.array(text_rgba2), cv2.COLOR_RGBA2BGRA)
        fg_bgr, fg_a = fg_bgra[..., :3], fg_bgra[..., 3]
        Hc, Wc = bg_bgr.shape[:2]

        x2, y2 = min(Wc, x + w2), min(Hc, y + h2)
        roi_bg = bg_bgr[y:y2, x:x2]
        roi_fg = fg_bgr[0:(y2 - y), 0:(x2 - x)]
        roi_a = fg_a[0:(y2 - y), 0:(x2 - x)]

        if roi_bg.size == 0 or roi_fg.size == 0:
            return None

        merged = bg_bgr.copy()
        if mode == 'alpha':
            merged[y:y2, x:x2] = alpha_blend(roi_bg, roi_fg, roi_a)
        elif mode == 'screen':
            merged[y:y2, x:x2] = screen_blend(roi_bg, roi_fg, roi_a)
        else:
            mask = np.zeros((Hc, Wc), dtype=np.uint8)
            mask[y:y2, x:x2] = roi_a
            fg_full = np.zeros_like(bg_bgr)
            fg_full[y:y2, x:x2] = roi_fg
            merged = poisson_blend(bg_bgr, fg_full, mask, x, y)

        # Optional occlusion (copy a background block over part of the text)
        if random.random() < 0.25:
            occ_w = random.randint(int(0.1 * w2), int(0.4 * w2))
            occ_h = random.randint(int(0.1 * h2), int(0.4 * h2))
            ox = random.randint(x, max(x, x2 - occ_w))
            oy = random.randint(y, max(y, y2 - occ_h))
            merged[oy:oy + occ_h, ox:ox + occ_w] = bg_bgr[oy:oy + occ_h, ox:ox + occ_w]

        # Degradations: blur/noise/exposure/gamma/low-res/JPEG
        blur_sigma = 0.0
        if random.random() < 0.7:
            if random.random() < 0.5:
                k = random.choice([3, 5, 7, 9])
                angle = random.uniform(0, 180)
                merged = motion_blur(merged, ksize=k, angle=angle)
                blur_sigma = 1.2
            else:
                sigma = random.uniform(0.5, 2.5)
                merged = cv2.GaussianBlur(merged, (0, 0), sigmaX=sigma, sigmaY=sigma)
                blur_sigma = sigma

        if random.random() < 0.5:
            merged = add_noise(merged, sigma=random.uniform(2, 12))

        if random.random() < 0.6:
            merged = adjust_exposure_gamma(merged, exposure=random.uniform(-0.2, 0.2), gamma=random.uniform(0.8, 1.4))

        if random.random() < 0.6:
            merged = lowres_resample(merged, scale=random.uniform(0.4, 0.9))

        if random.random() < 0.7:
            merged = jpeg_compress(merged, quality=random.randint(40, 90))

        # Quality metrics on the final image
        final_mask = np.zeros((Hc, Wc), dtype=np.uint8)
        final_mask[y:y2, x:x2] = roi_a
        eff_c = effective_contrast(merged, final_mask)   # 0..1
        vis_r = visible_ratio(final_mask)                 # 0..1 (strict; over entire image)
        min_ch = min_char_height_px(char_polys_after)

        return {
            'merged_bgr': merged,
            'poly': poly_after,
            'bbox': bbox,
            'char_polys': char_polys_after,
            'final_mask': final_mask,
            'contrast_val': eff_c,
            'visible_ratio': vis_r,
            'min_char_h': min_ch,
            'render_params': params,
            'geo': geo,
            'blend_mode': mode,
            'position_xy': (x, y),
            'size_wh': (w2, h2),
            'contrast_band': 'high' if eff_c >= 0.45 else ('medium' if eff_c >= 0.25 else 'low'),
            'blur_sigma': blur_sigma
        }

    def _reading_order(self, instances: List[Dict[str, Any]]) -> List[int]:
        """Top-to-bottom, left-to-right order based on bbox top-left (with vertical tolerance)."""
        idxs = list(range(len(instances)))
        def key(i):
            x1, y1, x2, y2 = instances[i]['bbox']
            return (int(y1 // 16), x1)  # 16 px row tolerance
        idxs.sort(key=key)
        return idxs

    def _to_annotation(self, image_id: str, W: int, H: int, instances: List[Dict[str, Any]], level: int) -> Dict[str, Any]:
        """Build the JSON annotation for an image, including L1/L2/L3 fields as requested."""
        ann = {
            'version': 'synthtext-fast-v1',
            'image_id': image_id,
            'image_size': {'width': W, 'height': H},
            'split': self.cfg.split,
            'instances': []
        }
        order = self._reading_order(instances)
        order_map = {idx: rank for rank, idx in enumerate(order)}
        for i, ins in enumerate(instances):
            poly = [(float(x), float(y)) for (x, y) in ins['poly']]
            bbox = [int(v) for v in ins['bbox']]
            poly_norm = normalize_poly(poly, W, H)
            bx1, by1, bx2, by2 = bbox
            bbox_norm = [clamp01(bx1 / W), clamp01(by1 / H), clamp01(bx2 / W), clamp01(by2 / H)]
            params: RenderParams = ins['render_params']
            geo = ins['geo']
            item = {
                'text': ins['text'],
                'poly': poly,
                'poly_norm': poly_norm,
                'bbox': bbox,
                'bbox_norm': bbox_norm,
                'orientation_deg': compute_angle_deg(poly),
            }
            if level >= 2:
                item['reading_order'] = order_map[i]
                item['difficulty'] = self._difficulty_tags(ins['blur_sigma'], max(abs(g) for g in geo['perspective']), ins['contrast_band'])
            if level >= 3:
                item['chars'] = [
                    {'char': ch, 'poly': [(float(x), float(y)) for (x, y) in cp]} for ch, cp in zip(ins['text'], ins['char_polys'])
                ]
                item['render_params'] = {
                    'font_family': Path(params.font_path).name if params.font_path else 'PIL-default',
                    'font_size_px': params.font_size_px,
                    'fill_rgb': params.fill_rgb,
                    'stroke_px': params.stroke_px,
                    'stroke_rgb': params.stroke_rgb,
                    'italic_shear_deg': params.italic_shear,
                    'letter_spacing_px': params.letter_spacing_px,
                    'shadow_px': params.shadow_px,
                }
                item['geom_params'] = {
                    'curve_amp_px': params.curve_amp_px,
                    'curve_freq': params.curve_freq,
                    'curve_phase': params.curve_phase,
                    'rotate_deg': geo['rotate_deg'],
                    'perspective': geo['perspective'],
                    'blend_mode': ins['blend_mode'],
                    'position_xy': ins['position_xy'],
                    'size_wh': ins['size_wh']
                }
                item['metrics'] = {
                    'effective_contrast_lin': ins['contrast_val'],
                    'visible_ratio': ins['visible_ratio'],
                    'min_char_height_px': ins['min_char_h']
                }
            ann['instances'].append(item)
        return ann

    def generate_one(self, idx: int, level: int = 2) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Generate a single composite image plus its annotation; retries if quality gates fail."""
        for _ in range(self.cfg.per_image_retry):
            bg = self._load_random_bg()
            W, H = bg.size
            instances = []
            exist_bboxes = []
            num = truncated_poisson(lam=3.0, lo=1, hi=8)
            if random.random() < 0.1:
                num = random.randint(1, 8)

            success = True
            merged_bgr = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
            for _k in range(num):
                ok = False
                for _ in range(self.cfg.per_instance_retry):
                    text = sample_string()
                    ins = self._build_instance(Image.fromarray(cv2.cvtColor(merged_bgr, cv2.COLOR_BGR2RGB)), text, exist_bboxes)
                    if ins is None:
                        # print(f"[Debug] Image {idx}: _build_instance returned None (placement failed?).")
                        continue
                    if ins['min_char_h'] < self.cfg.min_char_height_px:
                        # print(f"[Debug] Image {idx}: Failed quality gate: min_char_h={ins['min_char_h']:.1f}px (Need >= {self.cfg.min_char_height_px})")
                        continue
                    if ins['contrast_val'] < 0.08:
                        # print(f"[Debug] Image {idx}: Failed quality gate: contrast_val={ins['contrast_val']:.3f} (Need >= 0.08)")
                        continue
                    if ins['visible_ratio'] < 0.002:
                        # print(f"[Debug] Image {idx}: Failed quality gate: visible_ratio={ins['visible_ratio']:.4f} (Need >= 0.002)")
                        continue
                    ins['text'] = text
                    merged_bgr = ins['merged_bgr']
                    instances.append(ins)
                    exist_bboxes.append(ins['bbox'])
                    ok = True
                    break
                if not ok:
                    success = False
                    break
            if success and len(instances) > 0:
                image_id = f"{self.cfg.split}_{idx:07d}"
                ann = self._to_annotation(image_id, W, H, instances, level=level)
                return merged_bgr, ann
        return None

    def run(self, level: int = 2):
        """Main loop: generate images and write JSON/JSONL annotations."""
        out_images = Path(self.cfg.out_dir, 'images')
        out_labels = Path(self.cfg.out_dir, 'labels')
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        """ Resume logic: check existing files to determine start index. """
        start_idx = 0
        print(f"Checking {out_images} for existing files '{self.cfg.split}_'*.jpg' to resume from...")
        
        # scan existing files
        existing_files = out_images.glob(f"{self.cfg.split}_*.jpg")
        
        max_idx = -1
        for f in existing_files:
            try:
                # 'train_0000123.jpg'
                # extracting '0000123'
                idx_str = f.stem.split('_')[-1]
                idx_int = int(idx_str)
                if idx_int > max_idx:
                    max_idx = idx_int
            except Exception:
                # malformed filename, skip
                continue
        
        if max_idx >= 0:
            # start from next index
            start_idx = max_idx + 1
            print(f"Existing files found, resume index from {start_idx}.")

        if start_idx >= self.cfg.num_images:
             print(f"Generation finished，{start_idx}/{self.cfg.num_images} data generated! Congratulations!")
             return
        
        jsonl_fp = None
        if self.cfg.save_jsonl:
            jsonl_path = Path(self.cfg.out_dir, f'annotations_{self.cfg.split}.jsonl')
            
            # if resuming, open in append mode(a), else write mode(w)
            file_mode = 'a' if start_idx > 0 else 'w'
            jsonl_fp = open(jsonl_path, file_mode, encoding='utf-8')
            
            if file_mode == 'a':
                print(f"[append mode] appending new labels into: {jsonl_path}")
        
        
        # Adjust number of tasks to run based on resume index
        num_tasks_to_run = self.cfg.num_images - start_idx
        pbar = tqdm(range(start_idx, self.cfg.num_images), 
                    total=num_tasks_to_run,
                    desc=f'generating (split: {self.cfg.split})')

        for i in pbar:
            set_seed(self.cfg.seed + i) 
            
            res = self.generate_one(i, level=2) 
            if res is None:
                continue
            
            img_bgr, ann = res
            image_id = ann['image_id'] 
            img_fp = out_images / f'{image_id}.jpg' 
            cv2.imwrite(str(img_fp), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95]) 

            if jsonl_fp:
                jsonl_fp.write(json.dumps(ann, ensure_ascii=False) + '\n') 
            else:
                with open(out_labels / f'{image_id}.json', 'w', encoding='utf-8') as f: #
                    json.dump(ann, f, ensure_ascii=False, indent=2) #

        if jsonl_fp:
            jsonl_fp.close()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backgrounds', type=str, default='', help='Background directory (e.g., ImageNet subset). If empty, use gradient backgrounds for testing.')
    ap.add_argument('--fonts', type=str, default='', help='Fonts directory with ttf/otf/ttc/otc. If empty, use PIL default font (testing only).')
    ap.add_argument('--out_dir', type=str, default='./out', help='Output directory.')
    ap.add_argument('--num_images', type=int, default=100, help='Number of images to generate.')
    ap.add_argument('--res', type=int, nargs=2, default=[640, 640], metavar=('W', 'H'), help='Output resolution WxH.')
    ap.add_argument('--seed', type=int, default=123, help='Random seed.')
    ap.add_argument('--jsonl', action='store_true', help='Write jsonl instead of per-image JSON files.')
    ap.add_argument('--min_char_px', type=int, default=8, help='Minimum character height in pixels for quality gating.')
    ap.add_argument('--split', type=str, default='train', help='Dataset split (e.g., train, val, test).')
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = GenConfig(
        backgrounds_dir=args.backgrounds,
        fonts_dir=args.fonts if args.fonts else None,
        out_dir=args.out_dir,
        width=int(args.res[0]),
        height=int(args.res[1]),
        num_images=int(args.num_images),
        seed=int(args.seed),
        save_jsonl=bool(args.jsonl),
        min_char_height_px=int(args.min_char_px),
        split=args.split
    )
    gen = SynthTextDatasetGenerator(cfg)
    gen.run(level=2)


if __name__ == '__main__':
    main()
