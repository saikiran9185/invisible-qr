"""Command line: python -m iqr encode|decode|simulate ..."""
import argparse
import json
import secrets

from PIL import Image

from . import Payload, decode_any, encode, encode_print, psnr
from .codec import DEFAULT_DELTA
from .payload import DATA_TYPES
from .simulate import print_scan

TYPE_NAMES = {v: k for k, v in DATA_TYPES.items()}


def main() -> None:
    ap = argparse.ArgumentParser(prog="iqr", description="InvisibleQR encoder/decoder")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="embed an invisible code in an image")
    e.add_argument("input")
    e.add_argument("output", help="use .png (lossless) for best results")
    e.add_argument("--profile", choices=["digital", "print"], default="digital",
                   help="digital: screens/files; print: survives print + scan")
    e.add_argument("--id", help="64-bit identifier in hex (random if omitted)")
    e.add_argument("--type", default="image", choices=sorted(TYPE_NAMES))
    e.add_argument("--strength", type=float, default=None,
                   help="digital: QIM step (default 18); print: multiplier (default 1.0)")
    e.add_argument("--cell", type=int, default=None, help="print: data cell size in px (overrides --cells)")
    e.add_argument("--cells", type=int, default=48,
                   help="print: data cells across the shorter side (default 48). The pattern scales "
                        "with the artwork; fewer = coarser = readable from farther, more visible up close")
    e.add_argument("--print-size", default=None,
                   help="print: physical size, e.g. A4, A3, A2 or 420x594mm — reports pattern size "
                        "and how far away a phone can read it")

    d = sub.add_parser("decode", help="read a code from an image, photo or scan")
    d.add_argument("input")
    d.add_argument("--photo", action="store_true",
                   help="go straight to phone-photo mode (find the poster, fix perspective)")

    s = sub.add_parser("simulate", help="simulate printing and scanning an image")
    s.add_argument("input")
    s.add_argument("output")
    s.add_argument("--scale", type=float, default=1.2)
    s.add_argument("--angle", type=float, default=3.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--photo", action="store_true",
                   help="simulate a phone photo of the print on a wall instead of a flatbed scan")
    s.add_argument("--fill", type=float, default=0.45, help="photo: poster width / frame width")
    s.add_argument("--yaw", type=float, default=20, help="photo: sideways viewing angle (deg)")
    s.add_argument("--scene", default=None, help="photo: background photo to place the print in")

    a = ap.parse_args()
    if a.cmd == "encode":
        ident = int(a.id, 16) if a.id else secrets.randbits(64)
        src = Image.open(a.input).convert("RGB")
        p = Payload(identifier=ident, data_type=TYPE_NAMES[a.type])
        if a.profile == "print":
            out = encode_print(src, p, cell_px=a.cell, strength=a.strength or 1.0, cells=a.cells)
        else:
            out = encode(src, p, a.strength or DEFAULT_DELTA)
        if a.output.lower().endswith((".jpg", ".jpeg")):
            out.save(a.output, quality=95)
        else:
            out.save(a.output)
        info = {"id": f"{ident:016X}", "type": a.type, "profile": a.profile,
                "psnr_db": round(psnr(src, out), 2)}
        if a.profile == "print":
            from .sizing import cell_px_for, reading_guide
            cpx = a.cell or cell_px_for(src.size, a.cells)
            info["cell_px"] = cpx
            if a.print_size:
                info["print"] = reading_guide(src.size, cpx, a.print_size)
        print(json.dumps(info, indent=2))
    elif a.cmd == "decode":
        r, prof = decode_any(Image.open(a.input), photo_only=a.photo)
        out = r.to_dict()
        if r.found:
            out["profile"] = prof
        print(json.dumps(out, indent=2))
    else:
        if a.photo:
            from .camera import photograph
            corners, scene = None, None
            if a.scene:                       # replace the poster already in the scene photo
                from .locate import find_quads
                scene = Image.open(a.scene).convert("RGB")
                qs = find_quads(scene)
                if qs:
                    corners = qs[0] * (3000 / scene.width)
            photograph(Image.open(a.input), fill=a.fill, yaw=a.yaw, scene=scene, corners=corners,
                       frame=(3000, round(3000 * (scene.height / scene.width if scene else 4 / 3))),
                       seed=a.seed).save(a.output, quality=92)
            print(f"saved simulated phone photo to {a.output}")
        else:
            print_scan(Image.open(a.input), scale=a.scale, angle=a.angle, seed=a.seed).save(a.output)
            print(f"saved simulated scan to {a.output}")


if __name__ == "__main__":
    main()
