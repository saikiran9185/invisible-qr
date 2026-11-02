"""IQR-1 payload format (92 bits).

| Field            | Bits |
|------------------|------|
| Protocol version |    4 |
| Identifier       |   64 |
| Data type        |    8 |
| CRC-16/CCITT     |   16 |

The CRC detects decoding errors. It does NOT prove authenticity — trusted
records must be signed and verified by the registry (see docs/SPEC.md).
"""
from __future__ import annotations

from dataclasses import dataclass

PAYLOAD_BITS = 92
VERSION = 1

DATA_TYPES = {
    0x00: "generic",
    0x01: "url",
    0x02: "image",
    0x03: "product",
    0x04: "certificate",
    0x05: "provenance",
    0x06: "artwork",
    0x07: "document",
}


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


def _int_to_bits(value: int, width: int) -> list[int]:
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


def _bits_to_int(bits) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


@dataclass(frozen=True)
class Payload:
    identifier: int          # 64-bit
    data_type: int = 0x02    # 8-bit
    version: int = VERSION   # 4-bit

    def _body(self) -> bytes:
        # 76 bits packed into 10 bytes (left-aligned) for the CRC
        body = (self.version << 72) | (self.identifier << 8) | self.data_type
        return (body << 4).to_bytes(10, "big")

    def to_bits(self) -> list[int]:
        if not 0 <= self.identifier < 2**64:
            raise ValueError("identifier must fit in 64 bits")
        if not 0 <= self.data_type < 256:
            raise ValueError("data_type must fit in 8 bits")
        if not 0 <= self.version < 16:
            raise ValueError("version must fit in 4 bits")
        crc = crc16_ccitt(self._body())
        return (
            _int_to_bits(self.version, 4)
            + _int_to_bits(self.identifier, 64)
            + _int_to_bits(self.data_type, 8)
            + _int_to_bits(crc, 16)
        )

    @classmethod
    def from_bits(cls, bits) -> tuple["Payload", bool]:
        """Return (payload, crc_ok)."""
        bits = list(bits)
        if len(bits) != PAYLOAD_BITS:
            raise ValueError(f"expected {PAYLOAD_BITS} bits")
        p = cls(
            version=_bits_to_int(bits[0:4]),
            identifier=_bits_to_int(bits[4:68]),
            data_type=_bits_to_int(bits[68:76]),
        )
        return p, crc16_ccitt(p._body()) == _bits_to_int(bits[76:92])

    def to_dict(self, crc_ok: bool | None = None) -> dict:
        d = {
            "protocol": f"IQR-{self.version}",
            "id": f"{self.identifier:016X}",
            "type": DATA_TYPES.get(self.data_type, f"0x{self.data_type:02X}"),
        }
        if crc_ok is not None:
            d["verification"] = "valid" if crc_ok else "invalid"
        return d
