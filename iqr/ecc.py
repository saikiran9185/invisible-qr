"""Rate-1/2, constraint-length-7 convolutional code (generators 171/133 octal,
the NASA/Wi-Fi standard) with soft-decision Viterbi decoding."""
from __future__ import annotations

import numpy as np

K = 7
G = (0o171, 0o133)
NSTATES = 1 << (K - 1)


def _parity(x: int) -> int:
    return bin(x).count("1") & 1


# precompute trellis: for state s and input b -> (next_state, out0, out1)
_NEXT = np.zeros((NSTATES, 2), dtype=np.int64)
_OUT = np.zeros((NSTATES, 2, 2), dtype=np.int8)
for s in range(NSTATES):
    for b in (0, 1):
        reg = (b << (K - 1)) | s
        _NEXT[s, b] = reg >> 1
        _OUT[s, b] = [_parity(reg & G[0]), _parity(reg & G[1])]


def conv_encode(bits) -> list[int]:
    """Encode bits + (K-1) zero tail bits -> 2*(n+6) coded bits."""
    s, out = 0, []
    for b in list(bits) + [0] * (K - 1):
        out.extend(_OUT[s, b].tolist())
        s = _NEXT[s, b]
    return out


def viterbi_decode(soft, n_bits: int) -> list[int]:
    """soft: per coded bit, positive = likely 0, negative = likely 1."""
    soft = np.asarray(soft, dtype=np.float64).reshape(-1, 2)
    steps = n_bits + K - 1
    sgn = 1.0 - 2.0 * _OUT                       # 0 -> +1, 1 -> -1
    metric = np.full(NSTATES, -np.inf)
    metric[0] = 0.0
    back = np.zeros((steps, NSTATES), dtype=np.int64)
    bitp = np.zeros((steps, NSTATES), dtype=np.int8)
    for t in range(steps):
        new = np.full(NSTATES, -np.inf)
        allowed = (0,) if t >= n_bits else (0, 1)     # tail forces zeros
        for b in allowed:
            m = metric + sgn[:, b, 0] * soft[t, 0] + sgn[:, b, 1] * soft[t, 1]
            ns = _NEXT[:, b]
            for s in np.argsort(-m):                  # keep best per next state
                if m[s] > new[ns[s]]:
                    new[ns[s]] = m[s]
                    back[t, ns[s]] = s
                    bitp[t, ns[s]] = b
        metric = new
    s, bits = 0, []
    for t in range(steps - 1, -1, -1):
        bits.append(int(bitp[t, s]))
        s = back[t, s]
    return bits[::-1][:n_bits]
