"""Print-and-scan simulator.

Models what happens between a digital file and a scanned (or flat,
straight-on photographed) print:

1. print at high resolution with halftoning (Floyd–Steinberg dots per ink channel)
2. ink spread (blur at print resolution)
3. paper placed slightly crooked on the scanner (rotation, white border)
4. scanner resolution differs from the file (rescale) and optics blur
5. exposure / contrast / gamma shift, sensor noise, JPEG on save
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter


def _halftone(img: Image.Image) -> Image.Image:
    """CMY halftone via error-diffusion dithering, one dot layer per ink."""
    rgb = img.convert("RGB")
    chans = [ch.convert("1").convert("L") for ch in rgb.split()]   # dithered dots
    return Image.merge("RGB", chans)


def print_scan(
    img: Image.Image,
    scale: float = 1.0,
    angle: float = 0.0,
    dpi_factor: int = 3,
    ink_blur: float = 1.2,
    scan_blur: float = 0.6,
    exposure: float = 1.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    noise: float = 2.0,
    jpeg_quality: int | None = 90,
    halftone: bool = True,
    seed: int = 0,
) -> Image.Image:
    rng = np.random.default_rng(seed)
    w, h = img.size

    # 1-2. print: upsample to printer resolution, halftone, ink spread
    p = img.convert("RGB").resize((w * dpi_factor, h * dpi_factor), Image.BICUBIC)
    if halftone:
        p = _halftone(p)
    p = p.filter(ImageFilter.GaussianBlur(ink_blur * dpi_factor / 2))

    # 3. crooked placement on the scanner bed
    if angle:
        p = p.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))

    # 4. scanner resolution + optics
    s = scale / dpi_factor
    p = p.resize((max(1, round(p.width * s)), max(1, round(p.height * s))), Image.LANCZOS)
    if scan_blur:
        p = p.filter(ImageFilter.GaussianBlur(scan_blur))

    # 5. exposure, contrast, gamma, noise
    a = np.asarray(p, np.float64) / 255.0
    a = np.clip((a - 0.5) * contrast + 0.5, 0, 1) * exposure
    a = np.clip(a, 0, 1) ** gamma * 255.0
    a = a + rng.normal(0, noise, a.shape)
    out = Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8))

    if jpeg_quality:
        buf = io.BytesIO()
        out.save(buf, "JPEG", quality=jpeg_quality)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
    return out
