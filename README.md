# InvisibleQR

**A QR code you can't see.** InvisibleQR hides a small, recoverable code inside an ordinary image — a photo, poster, package or artwork — by repeating tiny encoded patterns across the whole picture. People see the normal image; a scanner reads the code.

The goal is *perceptually invisible, machine-detectable* encoding. A camera can only read what changes the pixels measurably, so the signal is kept below what the eye notices but above what a decoder can recover.

## Two profiles

| | **Digital (C1)** | **Print (C2)** |
|---|---|---|
| Use for | files, screens, social media | posters, packaging, prints |
| How | QIM on DCT coefficients, 8×8 blocks | masked additive carrier + pilot pattern, convolutional code |
| Invisibility | ~43 dB PSNR | ~42 dB PSNR (adjustable) |
| Survives | JPEG ≥ 65, cropping | printing, halftoning, scanning, rescale 0.75–1.6×, any rotation, exposure shifts, JPEG |
| Doesn't survive | resize, rotation, print | strong perspective (phone at an angle), very low-res scans |

Both carry the same 92-bit **IQR-1** payload — a compact ID, not the content. The ID points to a record in a registry.

```json
{ "protocol": "IQR-1", "id": "8F32A91C72B4D610", "type": "product", "verification": "valid" }
```

## Does it need its own scanner?

It needs its own **decoder software**, not special hardware. Any flatbed scanner, phone camera or webcam captures the picture; `python -m iqr decode` (later, a phone app) reads the hidden code from it. A normal camera app won't show anything on its own — the same way a normal camera doesn't read Digimarc codes unless the app supports them.

## How the print profile survives a scanner

1. **Pilot pattern** — a faint, regular diagonal texture adds four sharp peaks to the image's frequency spectrum. After print + scan, where those peaks land tells the decoder the new **scale and rotation**.
2. **Resample** the scan back to the original grid.
3. **Lock on** — search cell offset, 90° orientation and tile phase using a 32-bit sync pattern.
4. **Vote** — the 16×16-cell tile repeats across the whole image; every copy is summed.
5. **Correct errors** — a rate-½ convolutional code with soft Viterbi decoding fixes ~20 of 196 bits; bits are scattered so flat regions don't wipe out neighbours.
6. **Verify** — CRC-16 rejects anything that isn't a real code.

Strength follows the image: more signal in textured areas (hair, fabric, foliage), less in flat areas (sky, plain packaging) where it would show.

## Results

Simulated print → scan (halftone, ink spread, crooked placement, scanner resolution, blur, exposure/gamma, noise, JPEG) on real photos at print-file size (~1500 px):

| Strength | PSNR | Decoded |
|---|---|---|
| 1.0 (default) | 42.4 dB | **12 / 12** |
| 0.7 | 45.4 dB | 11 / 12 |

Small files (~500 px) have fewer pattern copies: 9 / 12 at default strength. Unmarked images: **0 false codes in 27 tries**.

Digital profile: JPEG 95/85/75/65 ✅, ≤55 ❌, crops to 120×120 px ✅.

> These are simulations. The next step is real print-and-scan tests on your own printer and scanner — see the roadmap.

## Quick start

```bash
pip install -r requirements.txt

# digital
python -m iqr encode photo.png out.png --id 8F32A91C72B4D610 --type image

# print (use for anything that will be printed)
python -m iqr encode poster.png poster_iqr.png --profile print --type product

# simulate printing + scanning, then decode
python -m iqr simulate poster_iqr.png scan.jpg --scale 1.15 --angle 4
python -m iqr decode scan.jpg          # tries both profiles automatically

# tests (~2 min)
pip install pytest && python -m pytest -q
```

Options: `--strength` (print: 0.6–1.4; lower = more invisible), `--cell` (print: cell size in px; bigger survives lower-res scans).

### Real print tips
- Encode the **full-resolution** print file, not a preview — more copies of the pattern means more reliable decoding.
- Export as PNG/TIFF; print at 200–300 dpi or more.
- Scan at **300 dpi or higher**, straight, colour or greyscale.

## Roadmap

- **Phase 1 — Digital encoding** ✅ payload format, C1 codec, CLI, tests
- **Phase 1b — Print & scan** ✅ C2 codec, pilot-based geometry recovery, convolutional code, print-scan simulator
- **Phase 2 — Camera decoding** perspective correction for phone photos; neural encoder/decoder (PyTorch, StegaStamp-style) trained on camera distortions; Android scanner (CameraX + LiteRT)
- **Phase 3 — Physical print testing** real printers and paper, posters, packaging
- **Phase 4 — Universal protocol** open spec, reference decoder, registry (FastAPI + PostgreSQL), signed records, C2PA link

See [`docs/SPEC.md`](docs/SPEC.md) for the protocol draft.

## Prior work

- Tancik, Mildenhall, Ng — *StegaStamp: Invisible Hyperlinks in Physical Photographs*, CVPR 2020.
- Digimarc — commercial invisible codes printed on retail packaging.
- C2PA — signed content provenance; InvisibleQR can point to a C2PA manifest.

## Structure

```
iqr/payload.py      IQR-1 payload + CRC
iqr/codec.py        digital profile (C1)
iqr/print_codec.py  print profile (C2)
iqr/ecc.py          convolutional code + Viterbi
iqr/simulate.py     print-and-scan simulator
tests/              robustness tests
docs/               GitHub Pages site + SPEC.md
```

---
Saikiran Chepuri · B.Des Interaction Design, USDI
