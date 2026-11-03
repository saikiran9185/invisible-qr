"""Phone-camera simulator: a printed poster photographed on a wall.

Models what happens between a print file and a phone photo of the print:

1. print with halftone dots and ink spread
2. the poster sits on a wall; the phone looks at it from an angle
   (true 3-D perspective: yaw, pitch, roll, distance)
3. lens blur, uneven lighting + vignette, white-balance shift
4. sensor noise, phone sharpening, downscale (e.g. WhatsApp ~1600 px)
5. JPEG compression
"""
from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _need_cv2():
    if cv2 is None:
        raise ImportError("phone-photo features need OpenCV: pip install opencv-python-headless")


def _halftone(rgb: np.ndarray) -> np.ndarray:
    chans = [np.asarray(Image.fromarray(rgb[..., i]).convert("1").convert("L")) for i in range(3)]
    return np.stack(chans, -1)


def wall(width: int, height: int, seed: int = 0) -> np.ndarray:
    """Plain painted wall: off-white with soft blotches and fine grain."""
    rng = np.random.default_rng(seed)
    base = np.array([236, 226, 205], np.float32)
    low = cv2.resize(rng.normal(0, 1, (12, 9)).astype(np.float32), (width, height),
                     interpolation=cv2.INTER_CUBIC)
    grain = cv2.GaussianBlur(rng.normal(0, 1, (height, width)).astype(np.float32), (0, 0), 1.5)
    img = base[None, None, :] + 6 * low[..., None] + 3 * grain[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def poster_corners(frame_w: int, frame_h: int, poster_w: float, poster_h: float, *,
                   fill: float = 0.5, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
                   hfov: float = 70.0, offset=(0.0, 0.0)) -> np.ndarray:
    """Project the poster's 4 corners into the photo with a pinhole camera.
    fill = fraction of the frame width the poster spans (roughly); angles in degrees."""
    f = frame_w / (2 * math.tan(math.radians(hfov) / 2))
    W, H = 1.0, poster_h / poster_w
    z = f * W / (fill * frame_w)                    # distance so the poster spans `fill`
    P = np.array([[-W / 2, -H / 2, 0], [W / 2, -H / 2, 0], [W / 2, H / 2, 0], [-W / 2, H / 2, 0]])
    a, b, c = (math.radians(t) for t in (yaw, pitch, roll))
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    Q = P @ (Rz @ Rx @ Ry).T + np.array([offset[0] * W, offset[1] * W, z])
    return np.stack([f * Q[:, 0] / Q[:, 2] + frame_w / 2, f * Q[:, 1] / Q[:, 2] + frame_h / 2], 1)


def photograph(img: Image.Image, *, corners=None, scene: Image.Image | None = None,
               frame=(4000, 3000), fill: float = 0.5, yaw: float = 15, pitch: float = -8,
               roll: float = 3, offset=(0.0, 0.0), out_long: int | None = 1600,
               halftone: bool = True, blur: float = 1.0, light: float = 0.25,
               warmth: float = 0.08, noise: float = 3.0, sharpen: float = 0.6,
               jpeg_quality: int | None = 75, seed: int = 0) -> Image.Image:
    """Simulate a phone photo of a printed poster.

    corners: 4x2 poster corners in *frame* pixels (TL, TR, BR, BL); if omitted
    they come from a 3-D camera pose (fill, yaw, pitch, roll, offset).
    scene: background photo (resized to `frame`); default a plain wall.
    out_long: final long side in px (1600 ≈ WhatsApp); None keeps `frame`.
    """
    _need_cv2()
    rng = np.random.default_rng(seed)
    fw, fh = frame
    rgb = np.asarray(img.convert("RGB"))
    ph, pw = rgb.shape[:2]
    if corners is None:
        corners = poster_corners(fw, fh, pw, ph, fill=fill, yaw=yaw, pitch=pitch,
                                 roll=roll, offset=offset)
    corners = np.asarray(corners, np.float32)

    # 1. print: bring the file to ~1.5x its size in the photo, halftone, ink spread
    span = max(np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[3]))
    k = min(1.0, 3.0 * span / pw)                   # halftone at ~3x photo resolution
    p = cv2.resize(rgb, (max(8, round(pw * k)), max(8, round(ph * k))), interpolation=cv2.INTER_AREA)
    if halftone:
        p = cv2.GaussianBlur(_halftone(p), (0, 0), 0.9)
    p = cv2.resize(p, (max(8, round(p.shape[1] / 2)), max(8, round(p.shape[0] / 2))),
                   interpolation=cv2.INTER_AREA)   # area-average the dots, as the lens would

    # 2. perspective onto the wall
    src = np.array([[0, 0], [p.shape[1], 0], [p.shape[1], p.shape[0]], [0, p.shape[0]]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, corners)
    bg = (np.asarray(scene.convert("RGB").resize((fw, fh), Image.LANCZOS)) if scene is not None
          else wall(fw, fh, seed))
    warped = cv2.warpPerspective(p, Hm, (fw, fh), flags=cv2.INTER_LINEAR)
    m = cv2.warpPerspective(np.ones(p.shape[:2], np.float32), Hm, (fw, fh), flags=cv2.INTER_LINEAR)
    out = bg.astype(np.float32) * (1 - m[..., None]) + warped.astype(np.float32) * m[..., None]

    # 3. optics + light
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    yy, xx = np.mgrid[0:fh, 0:fw].astype(np.float32)
    gx, gy = rng.uniform(-1, 1, 2)
    grad = 1 + light * (gx * (xx / fw - 0.5) + gy * (yy / fh - 0.5))
    vign = 1 - 0.35 * light * (((xx - fw / 2) / fw) ** 2 + ((yy - fh / 2) / fh) ** 2) * 4
    out *= (grad * vign)[..., None]
    out *= np.array([1 + warmth, 1.0, 1 - warmth], np.float32)

    # 4. sensor noise, phone sharpening, downscale
    out += rng.normal(0, noise, out.shape).astype(np.float32)
    if sharpen:
        out = out + sharpen * (out - cv2.GaussianBlur(out, (0, 0), 2.0))
    out = np.clip(out, 0, 255).astype(np.uint8)
    if out_long and max(fw, fh) > out_long:
        s = out_long / max(fw, fh)
        out = cv2.resize(out, (round(fw * s), round(fh * s)), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(out)
    if jpeg_quality:
        buf = io.BytesIO()
        pil.save(buf, "JPEG", quality=jpeg_quality)
        buf.seek(0)
        pil = Image.open(buf).convert("RGB")
    return pil
