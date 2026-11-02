# IQR Protocol — Draft 0.2

Status: working draft. Everything here may change before 1.0.

## 1. Layers

| Layer | Role |
|---|---|
| **IQR Protocol** | Open spec: payload, framing, error handling, decoding rules |
| **IQR Decoder** | Open-source reference decoder any app or phone maker can embed |
| **IQR Registry** | Optional service resolving IDs to signed records |

The protocol does not depend on one company or one model. Codecs (classical or neural) are versioned profiles of the same payload.

## 2. Payload — IQR-1 (92 bits)

| Field | Bits | Notes |
|---|---|---|
| Protocol version | 4 | `1` for this draft |
| Identifier | 64 | Opaque; resolved by a registry |
| Data type | 8 | see table below |
| CRC-16/CCITT | 16 | over the first 76 bits; detects errors only |

Data types: `0x00` generic, `0x01` url, `0x02` image, `0x03` product, `0x04` certificate, `0x05` provenance, `0x06` artwork, `0x07` document. `0x80–0xFF` reserved for private use.

**Security:** the CRC does not prove authenticity. Anyone can copy a marked image. Trusted records must be signed by the issuer (e.g. Ed25519) and verified by the scanner against the registry record — the code only says *which* record to check.

## 3. Codec profile C1 — classical DCT/QIM (reference implementation)

- Colour: convert to YCbCr, embed in Y only.
- Blocks: 8×8 px; tile: 12×12 blocks = 144 slots.
- Frame: `SYNC(16) ‖ PAYLOAD(92) ‖ SYNC(16) ‖ PAYLOAD[0:20]`
- SYNC: `1011 0010 1110 0010`
- Slot of block (r, c): `(r mod 12)·12 + (c mod 12)`
- Embedding: orthonormal 8×8 DCT; coefficients (2,3) and (3,2) quantised with step Δ (default 18) to lattice `kΔ` for bit 0 and `kΔ + Δ/2` for bit 1.
- Decoding: for each of 64 pixel offsets, soft value per block = mean of `cos(2π·c/Δ)`; fold onto the tile; for each of 144 phases correlate with SYNC; take the best (threshold 0.2); sum soft votes per payload bit; sign → bit; check CRC.

Robust to: lossless formats, JPEG ≥ ~65, cropping to ≥ 96×96 px.
Not robust to: scaling, rotation, perspective, print-and-capture (use C2).

## 4. Codec profile C2 — print (reference implementation)

Canonical units: 1 cell = 16 canonical px. The encoder chooses `cell_px` (default `clip(round(min(h,w)/64), 12, 20)`); everything below is in canonical px, so the decoder needs no settings.

- **Signal** (added to all RGB channels equally = luminance only):
  `Δ = strength · M(x) · (A_d · d · sin(2πu/8)·sin(2πv/8) + A_p · (cos(2π(u+v)/13) + cos(2π(u−v)/13)))`
  with `A_d = 3.4`, `A_p = 0.9`, `d = +1` for bit 0 and `−1` for bit 1.
- **Mask** `M`: local std over one cell → `clip(0.5 + sd/16, 0.5, 1.4)`, reduced near black/white.
- **Tile**: 16×16 cells = 256 slots. Logical frame = `SYNC‖SYNC` (32) + K=7 rate-½ convolutional code of the 92-bit payload (generators 171/133 octal, 6 tail bits → 196) + first 28 coded bits repeated. Logical → physical slot by a fixed permutation (`numpy default_rng(0x1A2B).permutation(256)`).
- **Decoding**
  1. High-pass the luminance; whitened FFT magnitude; find pilot peaks at radius `√2/(13·s)` → scale `s` and rotation (mod 90°).
  2. Resample to canonical; for 4 orientations × 64 cell offsets, correlate each cell with the carrier (four quadrature integral images), fold onto the tile, correlate all 256 phases with the sync template (FFT).
  3. Best lock with `|sync| ≥ 0.5` and repeated-bit agreement ≥ 0.3 → combine repeats → soft Viterbi → CRC.
  4. Small scale/angle refinement around each pilot candidate.

Robust to: halftone printing, ink spread, scanning at 0.75–1.6× file resolution, any rotation, exposure/gamma shifts, noise, JPEG ≥ 70.
Not robust to: strong perspective, scans far below file resolution with large blur, heavy cropping below one tile.

## 5. Codec profile N1 — neural (planned, Phase 2)

- Encoder: U-Net producing a residual Δ added to the image, `I_enc = I + Δ`, trained with perceptual (LPIPS) + L2 losses.
- Decoder: spatial-transformer alignment → CNN → 92 soft bits.
- Training noise layer: perspective warp, blur, lighting/colour shifts, sensor noise, JPEG, partial occlusion, resize.
- Payload is protected with BCH or a short LDPC code before embedding; tiling gives crop robustness as in C1.
- Deployment: PyTorch → LiteRT for Android (CameraX), Core ML for iOS; multi-frame accumulation for low light.

## 6. Scanner behaviour

1. Capture frames; run detection on a downscaled stream.
2. Accumulate soft votes across frames until the CRC passes or a timeout.
3. Show confidence; never present an unsigned record as "verified".
4. Resolve the ID through the registry over HTTPS; verify the signature.

## 7. Open questions

- Profile negotiation when one image carries both C1 and N1.
- How stock camera apps could adopt the decoder (system-level integration vs. a dedicated app).
- Interplay with C2PA manifests: embed the manifest hash vs. a registry ID.
