# InvisibleQR

**A QR code you can't see.** InvisibleQR hides a small, recoverable code inside an ordinary image — a photo, poster, package or artwork — by repeating tiny encoded patterns across the whole picture. People see the normal image; a scanner app reads the code.

The goal is *perceptually invisible, machine-detectable* encoding. A camera can only read what changes the pixels measurably, so the signal is kept below what the eye notices but above what a decoder can recover.

## How it works

1. **Divide** the image into 8×8-pixel blocks, grouped into repeating 12×12-block tiles.
2. **Embed** one bit per block by nudging two mid-frequency DCT coefficients of the brightness channel (quantisation index modulation).
3. **Repeat** the 144-slot frame (sync pattern + 92-bit payload) across every tile, so any ~96×96 px region holds a full copy.
4. **Decode** by searching every pixel offset and tile phase, locking onto the sync pattern, then majority-voting all copies and checking the CRC.

The payload is a compact ID, not the content itself — the ID points to a record in a registry.

```json
{ "protocol": "IQR-1", "id": "8F32A91C72B4D610", "type": "image", "verification": "valid" }
```

## Status — Phase 1 baseline (classical codec)

Measured on the bundled test image (`tests/test_codec.py`):

| Test | Result |
|---|---|
| Invisibility (PSNR) | ~43 dB |
| Lossless PNG | ✅ decoded |
| JPEG quality 95 / 85 / 75 / 65 | ✅ ✅ ✅ ✅ |
| JPEG quality ≤ 55 | ❌ |
| Crops (any offset, down to 120×120 px) | ✅ |
| Crop + JPEG 85 | ✅ |
| Unmarked image | correctly not detected |
| Resize, rotation, perspective, print-and-photograph | ❌ — Phase 2 (neural codec) |

## Quick start

```bash
pip install -r requirements.txt

# embed (random 64-bit ID if --id is omitted)
python -m iqr encode examples/sample.png out.png --id 8F32A91C72B4D610 --type product

# read it back
python -m iqr decode out.png

# tests
pip install pytest && python -m pytest -q
```

`--delta` sets strength: higher survives more compression but is less invisible (default 18).

## Roadmap

- **Phase 1 — Digital encoding** ✅ baseline codec, payload format, CLI, tests
- **Phase 2 — Camera decoding** neural encoder/decoder (PyTorch, StegaStamp-style) trained with simulated camera distortions; Android scanner (CameraX + LiteRT)
- **Phase 3 — Physical printing** print-and-capture tests on paper, posters, packaging
- **Phase 4 — Universal protocol** open spec, reference decoder, optional registry (FastAPI + PostgreSQL), signed records, C2PA link

See [`docs/SPEC.md`](docs/SPEC.md) for the protocol draft.

## Prior work

- Tancik, Mildenhall, Ng — *StegaStamp: Invisible Hyperlinks in Physical Photographs*, CVPR 2020 — showed 56-bit codes recovered from printed photos via phone camera.
- C2PA — signed content provenance; InvisibleQR can act as the camera-readable pointer to a C2PA manifest.

## Structure

```
iqr/          payload format, codec, CLI
tests/        robustness tests
docs/         GitHub Pages site + SPEC.md
examples/     sample image
```

---
Saikiran Chepuri · B.Des Interaction Design, USDI
