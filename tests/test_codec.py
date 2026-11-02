import io

import numpy as np
import pytest
from PIL import Image

from iqr import Payload, decode, encode, psnr

ID = 0x8F32A91C72B4D610


def natural_image(w=512, h=384, seed=0):
    """Synthetic photo-like test image (smooth gradients + texture + noise)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w] / max(w, h)
    r = 128 + 90 * np.sin(6 * xx + 2 * yy)
    g = 120 + 70 * np.cos(4 * yy - 3 * xx)
    b = 110 + 60 * np.sin(9 * xx * yy + 1)
    img = np.stack([r, g, b], -1) + rng.normal(0, 6, (h, w, 3))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


@pytest.fixture(scope="module")
def encoded():
    src = natural_image()
    return src, encode(src, Payload(identifier=ID))


def check(img):
    r = decode(img)
    assert r.found and r.crc_ok, r
    assert r.payload.identifier == ID


def test_payload_roundtrip():
    p = Payload(identifier=ID, data_type=3)
    q, ok = Payload.from_bits(p.to_bits())
    assert ok and q == p


def test_invisible(encoded):
    src, enc = encoded
    assert psnr(src, enc) > 38


def test_lossless(encoded):
    check(encoded[1])


@pytest.mark.parametrize("q", [95, 85, 75])
def test_jpeg(encoded, q):
    check(jpeg(encoded[1], q))


@pytest.mark.parametrize("box", [(13, 7, 300, 250), (101, 59, 400, 380), (3, 3, 120, 120)])
def test_crop(encoded, box):
    check(encoded[1].crop(box))


def test_crop_plus_jpeg(encoded):
    check(jpeg(encoded[1].crop((37, 21, 337, 291)), 85))


def test_unmarked_image_not_detected():
    r = decode(natural_image(seed=5))
    assert not (r.found and r.crc_ok)
