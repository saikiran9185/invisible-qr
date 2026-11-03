"""InvisibleQR — perceptually invisible, camera-readable image codes.

Profiles:
  digital (C1) — strongest invisibility; survives JPEG and cropping
  print   (C2) — survives printing, scanning, rescaling and rotation,
                 and phone photos of the print (perspective, light, JPEG)
"""
from .payload import Payload, DATA_TYPES
from .codec import encode, decode, psnr
from .print_codec import encode_print, decode_print

__all__ = ["Payload", "DATA_TYPES", "encode", "decode", "encode_print", "decode_print", "psnr"]
__version__ = "0.3.0"


def decode_any(img, photo_only: bool = False):
    """Try the digital profile (fast), then the print profile on the whole
    image (scans), then phone-photo mode (find poster, fix perspective)."""
    if not photo_only:
        r = decode(img)
        if r.found and r.crc_ok:
            return r, "digital"
        r = decode_print(img)
        if r.found and r.crc_ok:
            return r, "print"
    try:
        from .photo import decode_photo
    except ImportError:                      # OpenCV not installed
        return r, "print"
    return decode_photo(img), "print-photo"
