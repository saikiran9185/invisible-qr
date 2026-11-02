"""InvisibleQR — perceptually invisible, camera-readable image codes.

Profiles:
  digital (C1) — strongest invisibility; survives JPEG and cropping
  print   (C2) — survives printing, scanning, rescaling and rotation
"""
from .payload import Payload, DATA_TYPES
from .codec import encode, decode, psnr
from .print_codec import encode_print, decode_print

__all__ = ["Payload", "DATA_TYPES", "encode", "decode", "encode_print", "decode_print", "psnr"]
__version__ = "0.2.0"


def decode_any(img):
    """Try the digital profile first (fast), then the print profile."""
    r = decode(img)
    if r.found and r.crc_ok:
        return r, "digital"
    return decode_print(img), "print"
