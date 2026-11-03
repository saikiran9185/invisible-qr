"""Renders demo/explainer.mp4 from the demo assets (run make_demo.py first).

python demo/make_video.py            # full render
python demo/make_video.py --frames   # write key frames only, for checking
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from iqr.print_codec import PT, _frame, _luma            # noqa: E402
from iqr.payload import Payload                           # noqa: E402

W, H, FPS = 1280, 720, 30
BG, INK, MUTED, ACC, OK = (13, 13, 15), (236, 235, 231), (150, 147, 140), (127, 155, 255), (95, 211, 154)


def F(name, size):
    for p in (f"/usr/share/fonts/opentype/inter/{name}.otf",
              f"/usr/share/fonts/truetype/dejavu/{name}.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


TITLE = F("InterDisplay-Bold", 92)
H2 = F("InterDisplay-SemiBold", 44)
BODY = F("Inter-Regular", 26)
SMALL = F("Inter-Medium", 18)
MONO = F("DejaVuSansMono", 26)
MONO_S = F("DejaVuSansMono", 17)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def fit(img, w, h):
    im = img.copy()
    im.thumbnail((w, h), Image.LANCZOS)
    return im


def caption(d, step, title, body, alpha=1.0):
    c = lambda col: tuple(int(BG[i] + (col[i] - BG[i]) * alpha) for i in range(3))
    d.text((64, 50), step, font=SMALL, fill=c(ACC))
    d.text((64, 76), title, font=H2, fill=c(INK))
    y = 136
    for line in body:
        d.text((64, y), line, font=BODY, fill=c(MUTED))
        y += 36


def canvas():
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


# ------------------------------------------------------------------ assets
poster = Image.open(f"{HERE}/1_poster.png").convert("RGB")
marked = Image.open(f"{HERE}/2_poster_invisibleqr.png").convert("RGB")
pattern = Image.open(f"{HERE}/3_hidden_pattern_x40.png").convert("RGB")
scan = Image.open(f"{HERE}/4_simulated_scan.jpg").convert("RGB")
result = json.load(open(f"{HERE}/5_decode_result.json"))

PW, PH = 760, 507                       # poster display size
POS = ((W - PW) // 2 + 120, 196)
p_small, m_small, pat_small = (im.resize((PW, PH), Image.LANCZOS) for im in (poster, marked, pattern))


def spectrum_image():
    y = _luma(scan)
    hp = y - gaussian_filter(y, 4.0)
    h, w = hp.shape
    n = 1 << int(np.ceil(np.log2(max(h, w))))
    Fm = np.abs(np.fft.fftshift(np.fft.fft2(hp * np.outer(np.hanning(h), np.hanning(w)), s=(n, n))))
    wh = Fm / (gaussian_filter(Fm, 3.0) + 1e-9)
    c = n // 2
    r = int(n * 0.22)
    crop = wh[c - r:c + r, c - r:c + r]
    from scipy.ndimage import maximum_filter
    disp = maximum_filter(crop, size=11)                  # make sharp peaks visible
    v = 0.22 * np.clip(crop / 2.5, 0, 1) + 0.78 * np.clip((disp - 2.3) / 0.9, 0, 1)
    img = Image.fromarray((v * 255).astype(np.uint8)).resize((470, 470), Image.LANCZOS)
    # place circles where the decoder found the pilot (4 peaks, 90° apart)
    from iqr.print_codec import PILOT, _pilot_candidates
    s, ang = _pilot_candidates(y, k=1)[0]
    rad = np.sqrt(2) / (PILOT * s) * n                       # in FFT bins
    peaks = []
    for q in range(4):
        a = ang + np.pi / 4 + q * np.pi / 2
        fx, fy = rad * np.cos(a), rad * np.sin(a)
        peaks.append(((fx + r) * 470 / (2 * r), (fy + r) * 470 / (2 * r)))
    return img.convert("RGB"), peaks


SPEC, PEAKS = spectrum_image()
FRAME_BITS = _frame(Payload(identifier=int(result["id"], 16), data_type=3).to_bits())


# ------------------------------------------------------------------ scenes
def s_title(t, T):
    im, d = canvas()
    a = ease(t / 0.8)
    # dotted mark
    for i in range(3):
        for j in range(3):
            on = (i, j) in [(0, 0), (1, 1), (2, 0), (2, 2), (0, 2)]
            col = INK if on else (40, 40, 44)
            col = tuple(int(BG[k] + (col[k] - BG[k]) * a) for k in range(3))
            d.rectangle([64 + j * 22, 250 + i * 22, 64 + j * 22 + 16, 250 + i * 22 + 16], fill=col)
    c = lambda col, al: tuple(int(BG[k] + (col[k] - BG[k]) * al) for k in range(3))
    d.text((60, 330), "InvisibleQR", font=TITLE, fill=c(INK, a))
    d.text((64, 446), "A QR code you can't see.", font=H2, fill=c(MUTED, ease((t - 0.6) / 0.8)))
    d.text((64, 640), "Saikiran Chepuri · B.Des Interaction Design, USDI", font=SMALL,
           fill=c(MUTED, ease((t - 1.2) / 0.8)))
    return im


def s_compare(t, T):
    im, d = canvas()
    caption(d, "01 · SPOT THE DIFFERENCE", "One of these hides a code.",
            ["Left of the line: original poster. Right: the same poster",
             "carrying a hidden 92-bit InvisibleQR code."])
    x = 0.5 + 0.38 * np.sin(2 * np.pi * (t - 0.5) / 4.0) if t > 0.5 else 0.5
    cut = int(PW * x)
    frame = p_small.copy()
    frame.paste(m_small.crop((cut, 0, PW, PH)), (cut, 0))
    fx, fy = 64, 210
    im.paste(frame, (fx, fy))
    d.line([fx + cut, fy, fx + cut, fy + PH], fill=(255, 255, 255), width=3)
    d.text((fx + 12, fy + PH - 34), "ORIGINAL", font=SMALL, fill=(255, 255, 255))
    d.text((fx + PW - 112, fy + PH - 34), "ENCODED", font=SMALL, fill=(255, 255, 255))
    d.text((860, 300), "42 dB", font=TITLE, fill=INK)
    d.text((864, 410), "PSNR — the change averages", font=BODY, fill=MUTED)
    d.text((864, 446), "about one brightness step", font=BODY, fill=MUTED)
    d.text((864, 482), "out of 255.", font=BODY, fill=MUTED)
    return im


def s_pattern(t, T):
    im, d = canvas()
    caption(d, "02 · THE HIDDEN PATTERN", "Here it is, amplified 40×.",
            ["Strongest in textured areas (wood, foam), weakest in flat ones —",
             "so it hides where the eye can't notice it."])
    a = ease((t - 0.6) / 1.6)
    frame = Image.blend(m_small, pat_small, a)
    im.paste(frame, (64, 210))
    # zoom inset of the real pattern
    if t > 2.4:
        b = ease((t - 2.4) / 0.8)
        crop = pattern.crop((700, 300, 860, 460)).resize((360, 360), Image.NEAREST)
        crop = Image.blend(Image.new("RGB", crop.size, BG), crop, b)
        im.paste(crop, (860, 240))
        d.rectangle([860, 240, 1220, 600], outline=tuple(int(BG[k] + (ACC[k] - BG[k]) * b) for k in range(3)), width=2)
        d.text((860, 616), "zoomed: smooth cells, one bit each", font=SMALL,
               fill=tuple(int(BG[k] + (MUTED[k] - BG[k]) * b) for k in range(3)))
    return im


def s_tile(t, T):
    im, d = canvas()
    caption(d, "03 · HOW IT WORKS", "Distributed, not stamped.",
            ["The image is split into cells. Each cell carries one bit.",
             "A 16×16-cell tile holds the whole code — and it repeats",
             "across the entire image, so every patch holds a full copy."])
    n = PT * PT
    shown = int(n * ease(t / 3.0))
    cs, ox, oy = 22, 64, 300
    sync_mask = np.zeros(n, bool)
    from iqr.print_codec import _PERM
    sync_mask[_PERM[:32]] = True
    for k in range(n):
        i, j = divmod(k, PT)
        if k < shown:
            col = ACC if sync_mask[k] else (INK if FRAME_BITS[k] else (55, 55, 60))
        else:
            col = (28, 28, 32)
        d.rectangle([ox + j * cs, oy + i * cs, ox + j * cs + cs - 3, oy + i * cs + cs - 3], fill=col)
    lx = ox + PT * cs + 40
    for y, col, lab in ((310, ACC, "sync — lets the decoder lock on"),
                        (350, INK, "code bit 1"), (390, (55, 55, 60), "code bit 0")):
        d.rectangle([lx, y + 4, lx + 18, y + 22], fill=col)
        d.text((lx + 32, y), lab, font=BODY, fill=MUTED)
    if t > 3.2:
        a = ease((t - 3.2) / 1.0)
        c = tuple(int(BG[k] + (INK[k] - BG[k]) * a) for k in range(3))
        m = tuple(int(BG[k] + (MUTED[k] - BG[k]) * a) for k in range(3))
        d.text((lx, 470), "92-bit ID  →  error-correcting code", font=BODY, fill=c)
        d.text((lx, 506), "→  196 bits, scattered across the tile", font=BODY, fill=c)
        d.text((lx, 560), "Damage in one area can't wipe out", font=BODY, fill=m)
        d.text((lx, 596), "neighbouring bits.", font=BODY, fill=m)
    return im


def s_scan(t, T):
    im, d = canvas()
    caption(d, "04 · PRINT AND SCAN", "Printed, then scanned crooked.",
            ["Simulated: ink dots, ink spread, 15% larger, 4° rotation,",
             "darker exposure, sensor noise and JPEG compression."])
    a = ease(t / 1.2)
    sc = fit(scan, 640, 470)
    left = fit(marked, 400, 300)
    im.paste(left, (64, 260))
    d.text((64, 572), "FILE SENT TO PRINTER", font=SMALL, fill=MUTED)
    # arrow
    d.text((488, 380), "→", font=TITLE, fill=tuple(int(BG[k] + (ACC[k] - BG[k]) * a) for k in range(3)))
    x = int(600 + (1 - a) * 80)
    shadow = Image.new("RGB", sc.size, (0, 0, 0))
    im.paste(Image.blend(Image.new("RGB", sc.size, BG), sc, a), (x, 214))
    d.text((x, 214 + sc.height + 12), "WHAT THE SCANNER SAW", font=SMALL,
           fill=tuple(int(BG[k] + (MUTED[k] - BG[k]) * a) for k in range(3)))
    return im


def s_spectrum(t, T):
    im, d = canvas()
    caption(d, "05 · FINDING THE CODE AGAIN", "A faint pilot texture leaves fingerprints.",
            ["In the scan's frequency spectrum it makes four sharp peaks.",
             "Where they land tells the decoder the new size and rotation."])
    sx, sy = 64, 216
    im.paste(SPEC, (sx, sy))
    if t > 1.0:
        a = ease((t - 1.0) / 0.8)
        col = tuple(int(BG[k] + (ACC[k] - BG[k]) * a) for k in range(3))
        for (px, py) in PEAKS:
            d.ellipse([sx + px - 18, sy + py - 18, sx + px + 18, sy + py + 18], outline=col, width=3)
    tx = 600
    lines = [("Scale", "≈ 1.15×"), ("Rotation", "≈ 4°"), ("Then", "straighten → lock on sync"),
             ("", "→ sum every tile copy"), ("", "→ fix errors (Viterbi)"), ("", "→ check CRC")]
    for i, (k, v) in enumerate(lines):
        if t > 1.8 + i * 0.45:
            a = ease((t - 1.8 - i * 0.45) / 0.5)
            ck = tuple(int(BG[q] + (MUTED[q] - BG[q]) * a) for q in range(3))
            cv = tuple(int(BG[q] + (INK[q] - BG[q]) * a) for q in range(3))
            d.text((tx, 250 + i * 56), k, font=BODY, fill=ck)
            d.text((tx + 140, 250 + i * 56), v, font=MONO, fill=cv)
    return im


def s_result(t, T):
    im, d = canvas()
    caption(d, "06 · DECODED", "The exact ID, verified.",
            ["The ID points to a record — product details, ownership,",
             "provenance — that a signed registry can confirm."])
    rows = [("protocol", result["protocol"]), ("id", result["id"]), ("type", result["type"]),
            ("verification", result["verification"])]
    y = 280
    d.rounded_rectangle([64, 250, 760, 560], radius=14, fill=(22, 22, 26), outline=(45, 45, 50))
    d.text((92, y - 14), "{", font=MONO, fill=MUTED)
    for i, (k, v) in enumerate(rows):
        a = ease((t - 0.4 - i * 0.5) / 0.5)
        if a <= 0:
            continue
        y = 300 + i * 52
        ck = tuple(int(BG[q] + (ACC[q] - BG[q]) * a) for q in range(3))
        tgt = OK if k == "verification" else INK
        cv = tuple(int(BG[q] + (tgt[q] - BG[q]) * a) for q in range(3))
        d.text((120, y), f'"{k}":', font=MONO, fill=ck)
        d.text((410, y), f'"{v}"', font=MONO, fill=cv)
    d.text((92, 510), "}", font=MONO, fill=MUTED)
    if t > 2.8:
        a = ease((t - 2.8) / 0.6)
        c = tuple(int(BG[q] + (OK[q] - BG[q]) * a) for q in range(3))
        d.text((820, 300), "✓", font=TITLE, fill=c)
        m = tuple(int(BG[q] + (MUTED[q] - BG[q]) * a) for q in range(3))
        d.text((820, 420), "Same ID that was hidden", font=BODY, fill=m)
        d.text((820, 456), "in the poster file.", font=BODY, fill=m)
    return im


def s_end(t, T):
    im, d = canvas()
    a = ease(t / 0.8)
    c = lambda col, al: tuple(int(BG[k] + (col[k] - BG[k]) * al) for k in range(3))
    d.text((64, 150), "No special scanner needed.", font=H2, fill=c(INK, a))
    d.text((64, 214), "Any scanner or phone camera captures the image;", font=BODY, fill=c(MUTED, a))
    d.text((64, 250), "the InvisibleQR decoder reads the code.", font=BODY, fill=c(MUTED, a))
    b = ease((t - 0.9) / 0.8)
    items = [("Digital profile", "screens, files — survives JPEG and crops"),
             ("Print profile", "posters, packaging — survives print + scan"),
             ("Next", "phone-camera app with perspective correction")]
    for i, (k, v) in enumerate(items):
        d.text((64, 340 + i * 52), k, font=BODY, fill=c(ACC, b))
        d.text((330, 340 + i * 52), v, font=BODY, fill=c(INK, b))
    d.text((64, 620), "InvisibleQR · IQR-1 · research prototype", font=SMALL, fill=c(MUTED, ease((t - 1.6) / 0.8)))
    return im


SCENES = [("title", s_title, 4.0), ("compare", s_compare, 7.0), ("pattern", s_pattern, 6.5),
          ("tile", s_tile, 8.0), ("scan", s_scan, 6.5), ("spectrum", s_spectrum, 7.5),
          ("result", s_result, 6.0), ("end", s_end, 6.0)]
FADE = 0.4


def frames(fn, T):
    n = int(round(T * FPS))
    for i in range(n):
        t = i / FPS
        im = fn(t, T)
        # fade from / to black at scene edges
        f = min(1.0, t / FADE, (T - t) / FADE)
        if f < 1.0:
            im = Image.blend(Image.new("RGB", (W, H), BG), im, max(f, 0.0))
        yield im


def render(out):
    tmp = os.path.join(HERE, ".video_tmp")
    os.makedirs(tmp, exist_ok=True)
    parts = []
    for name, fn, T in SCENES:
        part, part_tmp = f"{tmp}/{name}.mp4", f"{tmp}/{name}.partial.mp4"
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                              "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                              "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", part_tmp],
                             stdin=subprocess.PIPE)
        for im in frames(fn, T):
            p.stdin.write(im.tobytes())
        p.stdin.close()
        if p.wait() != 0:
            raise SystemExit(f"ffmpeg failed on {name}")
        os.replace(part_tmp, part)
        parts.append(part)
        print("rendered", name)
    lst = f"{tmp}/list.txt"
    with open(lst, "w") as f:
        f.writelines(f"file '{os.path.abspath(p)}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c", "copy", "-movflags", "+faststart", out], check=True)
    for p in parts:
        os.remove(p)
    os.remove(lst)
    os.rmdir(tmp)


if __name__ == "__main__":
    if "--frames" in sys.argv:
        out = sys.argv[sys.argv.index("--frames") + 1] if len(sys.argv) > 2 else HERE
        for name, fn, T in SCENES:
            fn(T * 0.8, T).save(f"{out}/frame_{name}.jpg", quality=88)
        print("frames written to", out)
    else:
        render(os.path.join(HERE, "explainer.mp4"))
