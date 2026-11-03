"""Print-size maths: how big the hidden pattern is on paper, and how far
away a phone (or how coarse a scanner) can still read it."""
from __future__ import annotations

import math
import re

# Smallest data cell the photo decoder reliably reads, in photo pixels
# (measured with the phone-photo simulator: perspective, halftone, light, JPEG ~75).
MIN_CELL_PX_PHOTO = 10.0
MIN_CELL_PX_SCAN = 7.0

# Typical phone main camera: 12 MP, ~26 mm-equivalent lens.
PHONE_LONG_PX = 4032
PHONE_HFOV_DEG = 69.4
WHATSAPP_LONG_PX = 1600          # WhatsApp / most chat apps downscale photos to this

PAPER = {  # mm, portrait
    "A0": (841, 1189), "A1": (594, 841), "A2": (420, 594), "A3": (297, 420),
    "A4": (210, 297), "A5": (148, 210), "A6": (105, 148),
    "LETTER": (216, 279), "TABLOID": (279, 432),
}


def cell_px_for(size: tuple[int, int], cells: int) -> int:
    return max(12, int(round(min(size) / cells)))


def parse_size(s: str, image_size: tuple[int, int]) -> tuple[float, float]:
    """'A3' / 'a3' / '420x594mm' / '42x59.4cm' / '16.5x23.4in' -> (w_mm, h_mm),
    oriented like the image."""
    t = s.strip().upper().replace(" ", "")
    if t in PAPER:
        w, h = PAPER[t]
    else:
        m = re.fullmatch(r"([\d.]+)[X×]([\d.]+)(MM|CM|IN)?", t)
        if not m:
            raise ValueError(f"unknown print size {s!r} (use A0–A6, LETTER, or e.g. 420x594mm)")
        k = {"MM": 1, "CM": 10, "IN": 25.4, None: 1}[m.group(3)]
        w, h = float(m.group(1)) * k, float(m.group(2)) * k
    iw, ih = image_size
    if (iw > ih) != (w > h):
        w, h = h, w
    return w, h


def reading_guide(image_size: tuple[int, int], cell_px: int, print_size: str) -> dict:
    w_mm, h_mm = parse_size(print_size, image_size)
    iw, ih = image_size
    dpi = iw / (w_mm / 25.4)
    cell_mm = cell_px * w_mm / iw
    cells_w = iw / cell_px
    need_px = MIN_CELL_PX_PHOTO * cells_w                    # poster width in photo px
    focal = PHONE_LONG_PX / (2 * math.tan(math.radians(PHONE_HFOV_DEG) / 2))
    d_full = focal * (w_mm / 1000) / need_px
    d_chat = d_full * WHATSAPP_LONG_PX / PHONE_LONG_PX
    return {
        "size_mm": [round(w_mm), round(h_mm)],
        "file_dpi": round(dpi),
        "cell_mm": round(cell_mm, 2),
        "pattern_repeat_mm": round(16 * cell_mm, 1),
        "photo_min_poster_width_px": int(math.ceil(need_px)),
        "phone_max_distance_m": round(d_full, 2),
        "phone_max_distance_m_if_sent_on_whatsapp": round(d_chat, 2),
        "scanner_min_dpi": int(math.ceil(MIN_CELL_PX_SCAN / (cell_mm / 25.4))),
        "note": "Distances assume a 12 MP phone, 1x lens, whole poster in frame. "
                "Fewer --cells = coarser pattern = farther reading distance.",
    }
