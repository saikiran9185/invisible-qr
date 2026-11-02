"""Command line: python -m iqr encode|decode ..."""
import argparse
import json
import secrets

from PIL import Image

from . import Payload, decode, encode, psnr
from .codec import DEFAULT_DELTA
from .payload import DATA_TYPES

TYPE_NAMES = {v: k for k, v in DATA_TYPES.items()}


def main() -> None:
    ap = argparse.ArgumentParser(prog="iqr", description="InvisibleQR encoder/decoder")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="embed an invisible code in an image")
    e.add_argument("input")
    e.add_argument("output", help="use .png (lossless) for best results")
    e.add_argument("--id", help="64-bit identifier in hex (random if omitted)")
    e.add_argument("--type", default="image", choices=sorted(TYPE_NAMES))
    e.add_argument("--delta", type=float, default=DEFAULT_DELTA, help="strength")

    d = sub.add_parser("decode", help="read an invisible code from an image")
    d.add_argument("input")
    d.add_argument("--delta", type=float, default=DEFAULT_DELTA)

    a = ap.parse_args()
    if a.cmd == "encode":
        ident = int(a.id, 16) if a.id else secrets.randbits(64)
        src = Image.open(a.input)
        out = encode(src, Payload(identifier=ident, data_type=TYPE_NAMES[a.type]), a.delta)
        out.save(a.output, quality=95) if a.output.lower().endswith((".jpg", ".jpeg")) else out.save(a.output)
        print(json.dumps({"id": f"{ident:016X}", "type": a.type, "psnr_db": round(psnr(src, out), 2)}))
    else:
        print(json.dumps(decode(Image.open(a.input), a.delta).to_dict(), indent=2))


if __name__ == "__main__":
    main()
