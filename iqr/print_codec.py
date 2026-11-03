"""Print profile (C2) — survives printing, scanning, rescaling and rotation.

Differences from the digital profile (C1):

* Signal is an additive pattern, not coefficient quantisation, so it
  survives halftoning and blur.
* Every cell (16 canonical px) carries one bit as +/- a smooth carrier
  sin(2πu/8)·sin(2πv/8). The carrier is zero on cell borders, so flipping
  bits never creates visible seams.
* A faint, unmodulated diagonal *pilot* pattern (period 13) adds four sharp
  peaks to the image's frequency spectrum. Wherever they land after
  printing and scanning tells the decoder the new scale and rotation.
* Strength follows local texture (perceptual masking): busy regions hide
  more signal, flat regions (sky, plain packaging) get less.
* Everything is defined in "canonical" units, so the encoder can pick a
  cell size that suits the print size and the decoder needs no settings.

Decoding: find pilot peaks → estimate scale + rotation → resample to the
canonical grid → search cell offset, 90° orientation and tile phase using
the sync pattern → sum every tile copy → soft Viterbi decoding → CRC.

The frame is a 16x16-cell tile: 32 sync bits + the 92-bit payload under a
rate-1/2 convolutional code (196 bits) + 28 repeated bits, scattered by a
fixed permutation.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter, uniform_filter

from .codec import SYNC, DecodeResult
from .ecc import conv_encode, viterbi_decode
from .payload import PAYLOAD_BITS, Payload

CELL = 16                 # canonical px per cell
CARRIER = 8.0             # canonical carrier period (2 periods per cell)
PILOT = 13.0              # canonical pilot period (diagonal)
DATA_AMP = 3.0             # base data amplitude (grey levels)
PILOT_AMP = 1.5            # base pilot amplitude (geometry anchor for photos)
CELLS_SHORT = 48           # default data cells across the artwork's shorter side
SYNC_THRESHOLD = 0.5
WEIGHTED = True            # weight cells by local noise when decoding
DUP_THRESHOLD = 0.3

# Frame: one tile = PT x PT cells = 256 slots.
#   logical layout = SYNC x2 (32) + conv-coded payload (196) + coded[:28] (28)
# then a fixed pseudo-random permutation scatters the slots across the tile,
# so a flat or damaged region never knocks out neighbouring code bits.
PT = 16
PSLOTS = PT * PT
CODED = 2 * (PAYLOAD_BITS + 6)            # 196
_PERM = np.random.default_rng(0x1A2B).permutation(PSLOTS)   # logical -> physical
_SYNC_BITS = np.concatenate([SYNC, SYNC])


def _frame(bits) -> np.ndarray:
    coded = np.array(conv_encode(bits), np.int8)
    logical = np.concatenate([_SYNC_BITS, coded, coded[:PSLOTS - 32 - CODED]])
    phys = np.zeros(PSLOTS, np.int8)
    phys[_PERM] = logical
    return phys


# ---------------------------------------------------------------- helpers
def _luma(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGB"), np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def _highpass(y: np.ndarray, sigma: float) -> np.ndarray:
    return y - gaussian_filter(y, sigma)


def _mask(y: np.ndarray, win: int, floor: float = 0.5) -> np.ndarray:
    """Perceptual mask: `floor` on flat areas, up to 1.4 on textured areas."""
    m = uniform_filter(y, win)
    sd = np.sqrt(np.maximum(uniform_filter(y * y, win) - m * m, 0))
    lum = np.clip(np.minimum(m, 255 - m) / 40.0, 0.35, 1.0)   # protect near-black/white
    return np.clip(floor + sd / 16.0, floor, 1.4) * lum


# ---------------------------------------------------------------- encode
def encode_print(img: Image.Image, payload: Payload, cell_px: int | None = None,
                 strength: float = 1.0, cells: int = CELLS_SHORT) -> Image.Image:
    """Embed a print-robust code.

    The pattern scales with the artwork: by default the shorter side holds
    `cells` data cells (48 = 3 tiles), whatever the file's pixel size or the
    print's physical size. So a phone photo that frames the whole poster
    always sees the same pattern, whether it is an A5 flyer or an A0 banner.
    Fewer cells = coarser pattern = readable from lower-res / farther photos,
    but more visible up close. cell_px overrides this with a fixed size."""
    rgb = np.asarray(img.convert("RGB"), np.float64)
    h, w = rgb.shape[:2]
    if cell_px is None:
        cell_px = max(12, int(round(min(h, w) / cells)))
    if min(h, w) < PT * cell_px:
        raise ValueError(f"image too small for cell_px={cell_px}: need {PT*cell_px}px")
    k = CELL / cell_px                                   # image px -> canonical px
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    u, v = (u + 0.5) * k, (v + 0.5) * k

    frame = _frame(payload.to_bits())
    ci, cj = (v // CELL).astype(int) % PT, (u // CELL).astype(int) % PT
    sign = 1.0 - 2.0 * frame[ci * PT + cj]            # bit0 -> +1

    data = DATA_AMP * sign * np.sin(2 * np.pi * u / CARRIER) * np.sin(2 * np.pi * v / CARRIER)
    pilot = PILOT_AMP * (np.cos(2 * np.pi * (u + v) / PILOT) + np.cos(2 * np.pi * (u - v) / PILOT))

    y = _luma(img)
    win = max(5, cell_px)
    # the pilot is a regular lattice — the kind of pattern eyes spot first on
    # plain areas — so it backs off much more there than the data does; it
    # only needs total energy across the poster, which busy areas supply
    delta = strength * (_mask(y, win) * data + _mask(y, win, floor=0.2) * pilot)
    out = rgb + delta[..., None]                         # luminance-only change
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- decode
_F = np.array([[1.0, 1.0], [1.0, -1.0]]) / PILOT     # columns: canonical pilot freqs


def _peaks(y: np.ndarray, k: int = 8):
    """Strongest pilot-like spectral peaks: list of (strength, gx, gy) in
    cycles/pixel (half plane gy > 0), refined to sub-bin precision."""
    hp = _highpass(y, 4.0)
    h, w = hp.shape
    n = 1 << int(np.ceil(np.log2(max(h, w))))
    win = np.outer(np.hanning(h), np.hanning(w))
    F = np.abs(np.fft.fftshift(np.fft.fft2(hp * win, s=(n, n))))
    wh = F / (gaussian_filter(F, 3.0) + 1e-9)             # whiten: keep sharp spikes
    f = (np.arange(n) - n // 2) / n
    fx, fy = np.meshgrid(f, f)
    r = np.hypot(fx, fy)
    r0 = np.sqrt(2) / PILOT
    ann = (r > r0 / 4.0) & (r < r0 / 0.4)                 # scale 0.4x .. 4x
    ann &= (np.abs(fx) > 2 / n) & (np.abs(fy) > 2 / n)    # skip axis artefacts
    ann &= fy > 0                                         # half plane (symmetry)
    score = np.where(ann, wh, 0)
    peaks = (score == maximum_filter(score, size=9)) & (score > 0)
    py, px = np.nonzero(peaks)
    idx = np.argsort(score[py, px])[::-1][: k]

    def sub(a, b, c):
        d = a - 2 * b + c
        return 0.0 if d == 0 else 0.5 * (a - c) / d
    out = []
    for i in idx:
        yy, xx = py[i], px[i]
        dy = sub(F[yy - 1, xx], F[yy, xx], F[yy + 1, xx]) if 0 < yy < n - 1 else 0
        dx = sub(F[yy, xx - 1], F[yy, xx], F[yy, xx + 1]) if 0 < xx < n - 1 else 0
        out.append((float(score[yy, xx]), fx[yy, xx] + dx / n, fy[yy, xx] + dy / n))
    return out, hp * win


def _refine_peak(hpw: np.ndarray, gx: float, gy: float) -> tuple[float, float]:
    """Polish a peak frequency by maximising the windowed DFT magnitude
    around it (zoomed search, ~1/20 bin)."""
    h, w = hpw.shape
    sy, sx = max(1, h // 512), max(1, w // 512)          # subsample for speed
    sub = hpw[::sy, ::sx]
    yy = (np.arange(0, h, sy)[: sub.shape[0]])[:, None].astype(np.float64)
    xx = (np.arange(0, w, sx)[: sub.shape[1]])[None, :].astype(np.float64)
    n = 1 << int(np.ceil(np.log2(max(h, w))))
    step = 0.5 / n
    best = (gx, gy)

    def mag(a, b):
        ex = np.exp(-2j * np.pi * a * xx)
        ey = np.exp(-2j * np.pi * b * yy)
        return abs(np.sum((sub * ex) * ey))
    cur = mag(gx, gy)
    for _ in range(4):
        cand = [(best[0] + dx * step, best[1] + dy * step)
                for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        improved = False
        for a, b in cand:
            m = mag(a, b)
            if m > cur:
                cur, best, improved = m, (a, b), True
        if not improved:
            step /= 3
    return best


def _pilot_candidates(y: np.ndarray, k: int = 6) -> list[np.ndarray]:
    """Return up to k 2x2 matrices A (canonical px -> image px) from the pilot.

    The pilot has two frequency vectors f1, f2. After any affine distortion
    (scale, rotation, stretch, shear) they appear at g = A^-T f, so a pair of
    peaks pins down A completely. Single peaks give a similarity fallback."""
    peaks, hpw = _peaks(y, k=10)
    r0 = np.sqrt(2) / PILOT
    out: list[np.ndarray] = []

    def add(A):
        if all(np.abs(A - B).max() > 0.004 * np.abs(B).max() for B in out):
            out.append(A)

    pairs = []
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            si, gxi, gyi = peaks[i]
            sj, gxj, gyj = peaks[j]
            ri, rj = np.hypot(gxi, gyi), np.hypot(gxj, gyj)
            if not 0.75 < ri / rj < 1.33:
                continue
            cosang = abs(gxi * gxj + gyi * gyj) / (ri * rj)
            if cosang > 0.5:                              # need roughly 90° apart
                continue
            pairs.append((min(si, sj), i, j))
    pairs.sort(reverse=True)
    refined = {}

    def ref(i):
        if i not in refined:
            refined[i] = _refine_peak(hpw, peaks[i][1], peaks[i][2])
        return refined[i]
    for _, i, j in pairs[:3]:
        g1, g2 = np.array(ref(i)), np.array(ref(j))
        if g1[0] * g2[1] - g1[1] * g2[0] > 0:            # need det G < 0 (no mirror)
            g2 = -g2
        G = np.stack([g1, g2], 1)
        AmT = G @ np.linalg.inv(_F)                       # A^-T
        add(np.linalg.inv(AmT).T)
    for s, gx, gy in peaks[:k]:                           # similarity fallback
        gx, gy = ref(peaks.index((s, gx, gy)))
        scale = r0 / np.hypot(gx, gy)
        ang = np.arctan2(gy, gx) - np.pi / 4
        c, sn = np.cos(ang), np.sin(ang)
        add(scale * np.array([[c, -sn], [sn, c]]))
        if len(out) >= k:
            break
    return out[:k]


def _resample(y: np.ndarray, A: np.ndarray):
    """Map image to the canonical grid (undo the affine map A)."""
    h, w = y.shape
    cy, cx = (h - 1) / 2, (w - 1) / 2
    Ai = np.linalg.inv(A)
    corners = Ai @ np.array([[-cx, cx, cx, -cx], [-cy, -cy, cy, cy]])
    cw = int(np.ptp(corners[0])) + 1
    ch = int(np.ptp(corners[1])) + 1
    vv, uu = np.mgrid[0:ch, 0:cw].astype(np.float64)
    vv -= (ch - 1) / 2
    uu -= (cw - 1) / 2
    x = A[0, 0] * uu + A[0, 1] * vv + cx
    yv = A[1, 0] * uu + A[1, 1] * vv + cy
    out = map_coordinates(y, [yv, x], order=1, mode="constant", cval=np.nan)
    valid = ~np.isnan(out)
    out[~valid] = 0.0
    return out, valid


_SYNC_T = np.zeros(PSLOTS)
_SYNC_T[_PERM[:32]] = 1.0 - 2.0 * _SYNC_BITS
_SYNC_T = _SYNC_T.reshape(PT, PT)
_SYNC_TF = np.conj(np.fft.fft2(_SYNC_T))
_NSYNC = 32


def _carrier(shape):
    u = np.sin(2 * np.pi * (np.arange(shape[1]) + 0.5) / CARRIER)
    v = np.sin(2 * np.pi * (np.arange(shape[0]) + 0.5) / CARRIER)
    return np.outer(v, u)


def _fold(S: np.ndarray) -> np.ndarray:
    hc, wc = S.shape
    ph, pw = -hc % PT, -wc % PT
    P = np.pad(S, ((0, ph), (0, pw)))
    return P.reshape(P.shape[0] // PT, PT, P.shape[1] // PT, PT).sum(axis=(0, 2))


def _box_grid(I: np.ndarray, oy: int, ox: int) -> np.ndarray:
    """Sums over CELLxCELL boxes starting at (oy, ox), from integral image I."""
    h, w = I.shape[0] - 1, I.shape[1] - 1
    hc, wc = (h - oy) // CELL, (w - ox) // CELL
    ys, xs = oy + CELL * np.arange(hc + 1), ox + CELL * np.arange(wc + 1)
    J = I[np.ix_(ys, xs)]
    return J[1:, 1:] - J[:-1, 1:] - J[1:, :-1] + J[:-1, :-1]


def _search_grid(c: np.ndarray, valid: np.ndarray):
    """Search cell offsets, 90° orientations and tile phases on a canonical
    image. The carrier at any offset is a mix of four quadrature products
    (sin/cos in each axis), so each orientation needs only four full-image
    multiplies plus integral-image box sums.
    Returns (score, slots[256]) for the best lock, slots normalised."""
    hp = _highpass(c, 3.0) * valid
    best = (-1.0, None)
    for rot in range(4):
        r = np.rot90(hp, rot)
        h, w = r.shape
        if h < PT * CELL + CELL or w < PT * CELL + CELL:
            continue
        ax = lambda n, f: f(2 * np.pi * (np.arange(n) + 0.5) / CARRIER)
        sv, cv, su, cu = ax(h, np.sin), ax(h, np.cos), ax(w, np.sin), ax(w, np.cos)
        I = {}
        for ky, fy in (("s", sv), ("c", cv)):
            for kx, fx in (("s", su), ("c", cu)):
                q = r * fy[:, None] * fx[None, :]
                I[ky + kx] = np.pad(q.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        for oy in range(0, CELL, 2):
            ty = 2 * np.pi * oy / CARRIER
            wys, wyc = np.cos(ty), -np.sin(ty)          # sin(A-t) = sinA cos t - cosA sin t
            for ox in range(0, CELL, 2):
                tx = 2 * np.pi * ox / CARRIER
                wxs, wxc = np.cos(tx), -np.sin(tx)
                B = [_box_grid(I[q], oy, ox) for q in ("ss", "sc", "cs", "cc")]
                S = wys * wxs * B[0] + wys * wxc * B[1] + wyc * wxs * B[2] + wyc * wxc * B[3]
                if WEIGHTED:
                    # energy near the carrier frequency in each cell (any phase):
                    # busy image areas are noisy, so their vote counts less
                    E = B[0] ** 2 + B[1] ** 2 + B[2] ** 2 + B[3] ** 2
                    S = S / np.sqrt(E + np.median(E) * 0.25 + 1e-12)
                acc = _fold(S)
                acc /= np.sqrt(np.mean(acc ** 2)) + 1e-12
                corr = np.real(np.fft.ifft2(np.fft.fft2(acc) * _SYNC_TF)) / _NSYNC
                k = int(np.argmax(np.abs(corr)))
                sc = corr.flat[k]
                if abs(sc) > best[0]:
                    py, px = divmod(k, PT)
                    slots = np.roll(acc, (-py, -px), axis=(0, 1)).ravel() * np.sign(sc)
                    best = (abs(sc), slots)
    return best


def _dup_agreement(slots) -> float:
    """The first 28 coded bits are stored twice; real codes agree, noise doesn't."""
    lg = slots[_PERM]
    a, b = lg[32:60], lg[32 + CODED:]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


_last_dup = [0.0]


def _finish(score: float, slots) -> DecodeResult:
    _last_dup[0] = _dup_agreement(slots) if slots is not None else 0.0
    if slots is None or score < SYNC_THRESHOLD or _dup_agreement(slots) < DUP_THRESHOLD:
        return DecodeResult(False, None, False, 0.0, score)
    lg = slots[_PERM]
    soft = lg[32:32 + CODED].copy()
    soft[:PSLOTS - 32 - CODED] += lg[32 + CODED:]           # combine repeated bits
    bits = viterbi_decode(soft, PAYLOAD_BITS)
    payload, ok = Payload.from_bits(bits)
    # confidence: agreement between soft values and the re-encoded codeword
    ref = 1.0 - 2.0 * np.array(conv_encode(bits))
    conf = float(np.mean(np.sign(soft) == ref))
    return DecodeResult(True, payload, ok, conf, score)


def decode_print(img: Image.Image, deadline: float | None = None) -> DecodeResult:
    y = _luma(img)
    best = DecodeResult(False, None, False, 0.0, 0.0)
    import time
    for A in _pilot_candidates(y, k=4):
        if deadline and time.monotonic() > deadline:
            break
        c, valid = _resample(y, A)
        res = _finish(*_search_grid(c, valid))
        if res.found and res.crc_ok:
            return res
        if res.sync_score > best.sync_score:
            best = res
        if res.sync_score < 0.3 or _last_dup[0] < 0.2:
            continue                      # not even close: wrong peak
        # promising: refine the map a little around the estimate
        for ds, da in ((0.996, 0), (1.004, 0), (1, -0.003), (1, 0.003),
                       (0.992, 0), (1.008, 0)):
            if deadline and time.monotonic() > deadline:
                break
            c_, s_ = np.cos(da), np.sin(da)
            c, valid = _resample(y, ds * np.array([[c_, -s_], [s_, c_]]) @ A)
            res = _finish(*_search_grid(c, valid))
            if res.found and res.crc_ok:
                return res
    return best
