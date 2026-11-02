"""Print-and-scan robustness for the print profile (C2), using the simulator."""
import pytest

from iqr import Payload, psnr
from iqr.print_codec import decode_print, encode_print
from iqr.simulate import print_scan

from test_codec import natural_image

ID = 0x1F2E3D4C5B6A7988

CASES = {
    "straight scan": dict(),
    "scanned larger + crooked": dict(scale=1.3, angle=3),
    "scanned smaller + noisy + jpeg": dict(scale=0.85, angle=-5, noise=4, jpeg_quality=70),
    "rotated 30 deg": dict(scale=1.0, angle=30),
    "dark exposure + gamma": dict(scale=1.2, angle=8, exposure=0.85, gamma=1.25, contrast=0.9),
}


@pytest.fixture(scope="module")
def marked():
    src = natural_image(1200, 900, seed=3)          # typical print-file size
    return src, encode_print(src, Payload(identifier=ID))


def test_print_profile_invisible(marked):
    src, enc = marked
    assert psnr(src, enc) > 38


@pytest.mark.parametrize("name", list(CASES))
def test_print_scan(marked, name):
    r = decode_print(print_scan(marked[1], seed=1, **CASES[name]))
    assert r.found and r.crc_ok and r.payload.identifier == ID, name


def test_print_unmarked_not_detected():
    r = decode_print(print_scan(natural_image(1200, 900, seed=9), scale=1.1, angle=2))
    assert not (r.found and r.crc_ok)


def test_digital_profile_breaks_on_rescale(marked):
    """Documents why the print profile exists."""
    from iqr import decode, encode
    enc = encode(marked[0], Payload(identifier=ID))
    r = decode(print_scan(enc, scale=1.3, angle=3))
    assert not (r.found and r.crc_ok)
