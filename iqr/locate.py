"""Find a printed poster/page in a phone photo and flatten it.

Looks for large four-sided outlines (the paper edge, or the edge of the
artwork inside it), orders the corners clockwise and warps each candidate
to a straight-on rectangle. Any rectangle that lies on the poster works:
the decoder only needs the perspective removed, not the exact paper edge.
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft
from PIL import Image

from .camera import _need_cv2, cv2


def _order(q: np.ndarray) -> np.ndarray:
    """Clockwise from top-left (in image coordinates, y down)."""
    c = q.mean(0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]                       # clockwise on screen (y down)
    start = np.argmin(q.sum(1))                  # top-left = smallest x+y
    return np.roll(q, -start, axis=0)


def _edges(rgb: np.ndarray) -> np.ndarray:
    """Union of edges from luminance and the two colour channels (catches
    white paper on a cream wall, where brightness barely changes)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    out = np.zeros(rgb.shape[:2], np.uint8)
    for i in range(3):
        ch = cv2.GaussianBlur(lab[..., i], (5, 5), 0)
        med = float(np.median(ch))
        sd = float(ch.std()) + 1e-6
        lo = max(4.0, 0.25 * sd) if i else max(10.0, 0.66 * med * 0.3)
        out |= cv2.Canny(ch, lo, 2.5 * lo)
    return cv2.dilate(out, np.ones((3, 3), np.uint8))


def find_quads(img: Image.Image, max_quads: int = 4, work_px: int = 1000) -> list[np.ndarray]:
    """Return up to max_quads 4x2 corner arrays (full-res pixels), largest first."""
    _need_cv2()
    rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    s = min(1.0, work_px / max(h, w))
    small = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    sh, sw = small.shape[:2]
    edges = _edges(small)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads: list[tuple[float, np.ndarray]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.03 * sh * sw:
            continue
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        for eps in (0.01, 0.02, 0.03, 0.05):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                break
        else:
            continue
        q = ap.reshape(4, 2).astype(np.float64)
        a = cv2.contourArea(q.astype(np.float32))
        if a < 0.03 * sh * sw or a > 0.97 * sh * sw:
            continue
        # reject very skewed shapes (interior angles far from 90°)
        qo = _order(q)
        ok = True
        for i in range(4):
            v1, v2 = qo[i - 1] - qo[i], qo[(i + 1) % 4] - qo[i]
            cosang = abs(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            if cosang > 0.5:                     # angle outside 60°..120°
                ok = False
        if ok:
            quads.append((a, qo / s))
    quads.sort(key=lambda t: -t[0])
    out: list[np.ndarray] = []
    for _, q in quads:                           # drop near-duplicates
        if all(np.abs(q - o).max() > 0.02 * max(h, w) for o in out):
            out.append(q)
        if len(out) >= max_quads:
            break
    if out:
        out = [_refine(rgb, q) for q in out]
    return out


def _refine(rgb: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Sub-pixel corner refinement on the full-resolution image."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    pts = q.astype(np.float32).reshape(-1, 1, 2).copy()
    try:
        cv2.cornerSubPix(g, pts, (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        r = pts.reshape(4, 2).astype(np.float64)
        if np.abs(r - q).max() < 6:
            return r
    except cv2.error:
        pass
    return q


def _size(q: np.ndarray, upscale: float = 1.0) -> tuple[int, int]:
    wpx = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])) * upscale
    hpx = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])) * upscale
    return int(round(wpx)), int(round(hpx))


def _warp(arr: np.ndarray, q: np.ndarray, W: int, H: int, interp=None) -> np.ndarray:
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(np.asarray(q, np.float32), dst)
    return cv2.warpPerspective(arr, M, (W, H), flags=interp or cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def rectify(img: Image.Image, quad: np.ndarray, upscale: float = 1.0) -> Image.Image:
    """Warp the quad (TL, TR, BR, BL) to a straight-on rectangle."""
    _need_cv2()
    q = np.asarray(quad, np.float32)
    W, H = _size(q, upscale)
    return Image.fromarray(_warp(np.asarray(img.convert("RGB")), q, W, H))


def _pilot_sharpness(g: np.ndarray, q: np.ndarray, W: int, H: int, targets, n: int) -> float:
    """How sharp the hidden pilot peaks are after rectifying with quad q:
    a correct perspective makes the pilot one clean frequency (tall peaks);
    a wrong one smears it. A flat-top window keeps the poster's corners in
    play — that is exactly where corner errors show up."""
    r = _warp(g, q, W, H, cv2.INTER_LINEAR)
    hp = (r - cv2.GaussianBlur(r, (0, 0), 4.0)) * _window(H, W)
    F = np.abs(sfft.rfft2(hp, s=(n, n), workers=-1))
    tot = 0.0
    for (ky, kx) in targets:
        tot += float(F[max(0, ky - 4):ky + 5, max(0, kx - 4):kx + 5].max())
    return tot


_WIN: dict = {}


def _window(h: int, w: int) -> np.ndarray:
    if (h, w) not in _WIN:
        from scipy.signal.windows import tukey
        _WIN.clear()
        _WIN[(h, w)] = np.outer(tukey(h, 0.15), tukey(w, 0.15)).astype(np.float32)
    return _WIN[(h, w)]


def refine_quad(img: Image.Image, quad: np.ndarray, max_px: int = 1100, pick: int = 0):
    """Polish the 4 corners so the hidden pilot pattern is as sharp as possible.

    Paper edges in photos are unreliable (tape, frames, shadows, curl). The
    pilot is a perfect lattice printed across the whole poster, so the
    corners that make its spectral peaks tallest are the true perspective —
    a few pixels of corner error otherwise smears the grid across the poster.
    """
    from .print_codec import _F, _luma, _pilot_candidates   # local import (no cycle)
    _need_cv2()
    q = np.asarray(quad, np.float64).copy()
    W, H = _size(q)
    s = min(1.0, max_px / max(W, H))         # work on a moderate-size warp
    W, H = max(32, int(W * s)), max(32, int(H * s))
    g = _luma(img).astype(np.float32)
    n = 2 * (1 << int(np.ceil(np.log2(max(W, H)))))     # 2x zero-pad: finer peaks
    first = _warp(g, q, W, H, cv2.INTER_LINEAR)
    # the pilot's signature: two equal-strength peaks at right angles
    A = _pilot_candidates(first.astype(np.float64), k=pick + 1)
    if len(A) <= pick:
        return None
    G = np.linalg.inv(A[pick]).T @ _F
    targets = []
    for gx, gy in G.T:                       # rfft2 keeps kx >= 0: mirror if needed
        if gx < 0:
            gx, gy = -gx, -gy
        targets.append((int(round(gy * n)) % n, int(round(gx * n))))
    cur = _pilot_sharpness(g, q, W, H, targets, n)
    span = max(np.ptp(q[:, 0]), np.ptp(q[:, 1]))
    for step in (0.016 * span, 0.008 * span, 0.004 * span, 0.002 * span):
        for _ in range(6):
            improved = False
            for i in range(4):
                for ax in (0, 1):
                    for sgn in (1, -1):
                        t = q.copy()
                        t[i, ax] += sgn * step
                        v = _pilot_sharpness(g, t, W, H, targets, n)
                        if v > cur * 1.001:
                            cur, q, improved = v, t, True
                            break
            if not improved:
                break
    return q
