"""Phone photos of a printed poster (print profile + photo pipeline)."""
import numpy as np
import pytest
from PIL import Image

pytest.importorskip("cv2")

from iqr import Payload, decode_any                      # noqa: E402
from iqr.camera import photograph                        # noqa: E402
from iqr.photo import decode_photo                       # noqa: E402
from iqr.print_codec import encode_print                 # noqa: E402
from iqr.sizing import reading_guide                     # noqa: E402

from test_codec import natural_image                     # noqa: E402

ID = 0x2C4E6A8B0D1F3355

PHOTOS = {
    "straight on, half the frame": dict(fill=0.55, yaw=0, pitch=0, roll=2),
    "from the side (25°), tilted": dict(fill=0.5, yaw=25, pitch=-8, roll=4),
    "from below, WhatsApp-size JPEG 70": dict(fill=0.6, yaw=-15, pitch=15, roll=-3, jpeg_quality=70),
}


@pytest.fixture(scope="module")
def poster():
    src = natural_image(1240, 1754, seed=5)                  # A4 at 150 dpi
    return encode_print(src, Payload(identifier=ID, data_type=0x03))


@pytest.mark.parametrize("name", list(PHOTOS))
def test_phone_photo(poster, name):
    ph = photograph(poster, frame=(3000, 4000), out_long=1600, seed=2, **PHOTOS[name])
    r = decode_photo(ph)
    assert r.found and r.crc_ok and r.payload.identifier == ID, name


def test_decode_any_falls_back_to_photo(poster):
    ph = photograph(poster, frame=(3000, 4000), fill=0.5, yaw=20, seed=4)
    r, profile = decode_any(ph)
    assert r.crc_ok and r.payload.identifier == ID and profile == "print-photo"


def test_unmarked_photo_not_detected():
    ph = photograph(natural_image(1240, 1754, seed=8), frame=(3000, 4000), fill=0.5, yaw=20)
    r = decode_photo(ph)
    assert not (r.found and r.crc_ok)


def test_pattern_scales_with_artwork():
    """Same artwork at two resolutions -> same number of cells (pattern scales)."""
    from iqr.sizing import cell_px_for
    small, big = cell_px_for((1240, 1754), 48), cell_px_for((4960, 7016), 48)
    assert abs(1240 / small - 4960 / big) < 2


def test_reading_guide_bigger_print_reads_from_farther():
    a4 = reading_guide((2480, 3508), 52, "A4")
    a2 = reading_guide((2480, 3508), 52, "A2")
    assert a4["size_mm"] == [210, 297]
    assert a2["phone_max_distance_m"] > 1.9 * a4["phone_max_distance_m"]
    assert np.isclose(a2["cell_mm"], 2 * a4["cell_mm"], rtol=0.02)
