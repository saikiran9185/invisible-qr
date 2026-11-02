"""Classical baseline codec for InvisibleQR (Phase 1).

Distributed invisible code: the image is split into 8x8 pixel blocks, and the
blocks are grouped into repeating 12x12-block tiles. Every block in a tile
carries one bit of a 144-slot frame, embedded by quantisation index
modulation (QIM) on two mid-frequency DCT coefficients of the luminance
channel. The tile repeats across the whole image, so any region of
about 96x96 pixels holds a full copy, and more area means more votes.

    frame (144 slots) = SYNC(16) + PAYLOAD(92) + SYNC(16) + PAYLOAD[:20]

The decoder searches all 64 pixel offsets and all 144 tile phases, uses the
sync pattern to lock on, then majority-votes every copy of each bit. This
makes it survive cropping and JPEG compression. It does not survive
rescaling, rotation or perspective — that is what the neural codec in
Phase 2 is for (see docs/SPEC.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

from .payload import PAYLOAD_BITS, Payload

BLOCK = 8
TILE = 12                      # tile = TILE x TILE blocks
SLOTS = TILE * TILE            # 144
SYNC = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0], dtype=np.int8)
COEFFS = [(2, 3), (3, 2)]      # mid-frequency DCT positions used per block
DEFAULT_DELTA = 18.0           # QIM step: higher = more robust, less invisible


def _frame(bits: list[int]) -> np.ndarray:
    f = np.concatenate([SYNC, np.array(bits, np.int8), SYNC, np.array(bits[:20], np.int8)])
    assert f.size == SLOTS
    return f


# Which payload bit / sync bit each slot holds (used to vote on decode)
_SLOT_KIND = np.array(["s"] * 16 + ["p"] * 92 + ["s"] * 16 + ["p"] * 20)
_SLOT_INDEX = np.array(list(range(16)) + list(range(92)) + list(range(16)) + list(range(20)))


def _to_ycbcr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB").convert("YCbCr"), dtype=np.float64)


def _from_ycbcr(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "YCbCr").convert("RGB")


def encode(img: Image.Image, payload: Payload, delta: float = DEFAULT_DELTA) -> Image.Image:
    """Embed the payload invisibly and return a new RGB image."""
    ycc = _to_ycbcr(img)
    y = ycc[..., 0]
    frame = _frame(payload.to_bits())
    hb, wb = y.shape[0] // BLOCK, y.shape[1] // BLOCK
    if hb < TILE or wb < TILE:
        raise ValueError(f"image too small: need at least {TILE*BLOCK}x{TILE*BLOCK} px")

    for by in range(hb):
        for bx in range(wb):
            bit = frame[(by % TILE) * TILE + (bx % TILE)]
            sl = np.s_[by * BLOCK:(by + 1) * BLOCK, bx * BLOCK:(bx + 1) * BLOCK]
            d = dctn(y[sl], norm="ortho")
            for (u, v) in COEFFS:
                off = 0.5 * delta * bit
                d[u, v] = np.round((d[u, v] - off) / delta) * delta + off
            y[sl] = idctn(d, norm="ortho")
    ycc[..., 0] = y
    return _from_ycbcr(ycc)


def _soft_grid(y: np.ndarray, oy: int, ox: int, delta: float) -> np.ndarray:
    """Soft bit per block: +1 ~ bit 0, -1 ~ bit 1."""
    y = y[oy:, ox:]
    hb, wb = y.shape[0] // BLOCK, y.shape[1] // BLOCK
    blocks = y[:hb * BLOCK, :wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK).transpose(0, 2, 1, 3)
    d = dctn(blocks, axes=(2, 3), norm="ortho")
    s = np.zeros((hb, wb))
    for (u, v) in COEFFS:
        s += np.cos(2 * np.pi * d[:, :, u, v] / delta)
    return s / len(COEFFS)


@dataclass
class DecodeResult:
    found: bool
    payload: Payload | None
    crc_ok: bool
    confidence: float      # mean |vote| on payload bits, 0..1
    sync_score: float      # sync correlation, 0..1

    def to_dict(self) -> dict:
        if not self.payload:
            return {"found": False}
        d = self.payload.to_dict(self.crc_ok)
        d["confidence"] = round(self.confidence, 3)
        return d


def decode(img: Image.Image, delta: float = DEFAULT_DELTA) -> DecodeResult:
    y = _to_ycbcr(img)[..., 0]
    sync_sign = 1.0 - 2.0 * np.concatenate([SYNC, SYNC])      # bit0 -> +1
    sync_slots = np.where(_SLOT_KIND == "s")[0]
    best = None
    for oy in range(BLOCK):
        for ox in range(BLOCK):
            s = _soft_grid(y, oy, ox, delta)
            hb, wb = s.shape
            if hb < TILE or wb < TILE:
                continue
            # fold the grid onto one tile: acc[r, c] = sum of blocks at (r, c) mod TILE
            acc = np.zeros((TILE, TILE))
            ph, pw = (hb // TILE) * TILE, (wb // TILE) * TILE
            acc += s[:ph, :pw].reshape(hb // TILE, TILE, wb // TILE, TILE).sum(axis=(0, 2))
            acc /= (hb // TILE) * (wb // TILE)
            for py in range(TILE):
                for px in range(TILE):
                    slots = np.roll(acc, (py, px), axis=(0, 1)).ravel()
                    score = float(np.dot(slots[sync_slots], sync_sign)) / sync_slots.size
                    if best is None or score > best[0]:
                        best = (score, slots)
    if best is None or best[0] < 0.2:
        return DecodeResult(False, None, False, 0.0, best[0] if best else 0.0)

    score, slots = best
    votes = np.zeros(PAYLOAD_BITS)
    for i in np.where(_SLOT_KIND == "p")[0]:
        votes[_SLOT_INDEX[i]] += slots[i]
    bits = (votes < 0).astype(int).tolist()
    payload, ok = Payload.from_bits(bits)
    conf = float(np.mean(np.abs(slots[_SLOT_KIND == "p"])))
    return DecodeResult(True, payload, ok, min(conf, 1.0), score)


def psnr(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB"), np.float64)
    z = np.asarray(b.convert("RGB"), np.float64)
    mse = np.mean((x - z) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
