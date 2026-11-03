"""Builds the demo: poster -> InvisibleQR (print profile) -> simulated print+scan -> decode.

Run from the repo root:  python demo/make_demo.py
(needs scikit-image for the sample photo: pip install scikit-image)
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iqr import Payload, decode_any, psnr                      # noqa: E402
from iqr.print_codec import encode_print                       # noqa: E402
from iqr.simulate import print_scan                            # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
ID = 0x8F32A91C72B4D610


def font(name, size):
    for p in (f"/usr/share/fonts/opentype/inter/{name}.otf", f"/Library/Fonts/{name}.otf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def make_poster():
    from skimage import data
    photo = Image.fromarray(data.coffee()).resize((1800, 1200), Image.LANCZOS)
    d = ImageDraw.Draw(photo)
    d.text((90, 840), "Morning Ritual", font=font("InterDisplay-Bold", 120), fill=(250, 246, 238))
    d.text((96, 990), "SINGLE-ORIGIN · SMALL BATCH · ROASTED IN DELHI",
           font=font("Inter-Medium", 34), fill=(250, 246, 238))
    return photo


if __name__ == "__main__":
    poster = make_poster()
    poster.save(f"{OUT}/1_poster.png")

    marked = encode_print(poster, Payload(identifier=ID, data_type=0x03))
    marked.save(f"{OUT}/2_poster_invisibleqr.png")

    diff = np.abs(np.asarray(marked, float) - np.asarray(poster, float)).mean(-1)
    Image.fromarray(np.clip(diff * 40, 0, 255).astype(np.uint8)).save(f"{OUT}/3_hidden_pattern_x40.png")

    scan = print_scan(marked, scale=1.15, angle=4, exposure=0.92, gamma=1.1, noise=3, seed=7)
    scan.save(f"{OUT}/4_simulated_scan.jpg", quality=90)

    result, profile = decode_any(Image.open(f"{OUT}/4_simulated_scan.jpg"))
    out = result.to_dict()
    out["profile"] = profile
    out["psnr_db"] = round(psnr(poster, marked), 2)
    with open(f"{OUT}/5_decode_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
