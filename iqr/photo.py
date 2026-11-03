"""Decode a print-profile code from a phone photo of a printed poster.

photo -> find poster outlines -> polish corners with the hidden pilot
-> flatten -> print-profile decoder (affine pilot lock + weighted voting).
Falls back to the whole frame when no outline is found (poster fills the
photo, or it was cropped).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .codec import DecodeResult
from .locate import find_quads, rectify, refine_quad
from .print_codec import decode_print


def decode_photo(img: Image.Image, max_quads: int = 3, max_side: int = 4800,
                 poster_px: int = 1800, time_budget: float = 45.0) -> DecodeResult:
    import time
    deadline = time.monotonic() + time_budget
    img = img.convert("RGB")
    if max(img.size) > max_side:                # only shrink beyond 20+ MP originals
        s = max_side / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    w, h = img.size
    quads = find_quads(img, max_quads=max_quads)
    if not quads:                                # poster fills / was cropped to the frame
        m = 0.01 * max(w, h)
        quads = [np.array([[m, m], [w - m, m], [w - m, h - m], [m, h - m]])]
    best = DecodeResult(False, None, False, 0.0, 0.0)
    for q in quads:
        if time.monotonic() > deadline:
            break
        for pick in (0, 1):                      # two best pilot guesses, tried lazily
            if time.monotonic() > deadline:
                break
            cand = refine_quad(img, q, pick=pick)
            if cand is None:
                cand = q if pick == 0 else None
            if cand is None:
                continue
            # flatten at most poster_px on the long side: with 48 cells that is
            # still ~26 px per cell, and keeps big posters in 12 MP photos fast
            span = max(np.ptp(cand[:, 0]), np.ptp(cand[:, 1]))
            r = decode_print(rectify(img, cand, upscale=min(1.0, poster_px / span)), deadline)
            if r.found and r.crc_ok:
                return r
            if r.sync_score > best.sync_score:
                best = r
    return best
