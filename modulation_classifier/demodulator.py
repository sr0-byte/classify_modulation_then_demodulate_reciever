"""Receiver side: integrate-and-dump, phase synchronisation (preamble-resolved), slicing, BER counting."""
import numpy as np
from config import SPS, PREAMBLE, PREAMBLE_BITS


def _slice(zr, mod):
    if mod == "BPSK":
        return (zr.real < 0).astype(int)
    if mod == "QPSK":
        return np.column_stack([zr.real < 0, zr.imag < 0]).astype(int).ravel()
    if mod == "16QAM":
        s = zr * np.sqrt(10)

        def to_bits(v):
            return (v > 0).astype(int), (np.abs(v) < 2).astype(int)
        i1, i0 = to_bits(s.real)
        q1, q0 = to_bits(s.imag)
        return np.column_stack([i1, i0, q1, q0]).ravel()
    raise ValueError(mod)


def demodulate(iq, mod, sps=SPS, sync="preamble"):
    """Returns (bits incl. preamble, info). info['z'] holds the symbols used for the constellation plot."""
    n = len(iq) // sps
    blocks = np.asarray(iq[:n * sps], dtype=complex).reshape(n, sps)
    if mod == "BFSK":
        m = np.arange(sps)
        t1, t0 = np.exp(-2j * np.pi * m / sps), np.exp(2j * np.pi * m / sps)
        e1, e0 = np.abs(blocks @ t1), np.abs(blocks @ t0)
        return (e1 > e0).astype(int), {}
    z = blocks.mean(axis=1)
    if mod == "DPSK":
        d = z[1:] * np.conj(z[:-1])
        return np.column_stack([d.imag < 0, d.real < 0]).astype(int).ravel(), {"z": z}
    z = z / np.sqrt(np.mean(np.abs(z) ** 2) + 1e-12)
    if sync == "none":
        return _slice(z, mod), {"z": z}
    if mod == "BPSK":
        phi, amb = np.angle(np.sum(z ** 2)) / 2, [0, np.pi]
    else:
        phi, amb = (np.angle(np.sum(z ** 4)) - np.pi) / 4, [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    # resolve the phase ambiguity with the known preamble
    best = None
    for a in amb:
        zr = z * np.exp(-1j * (phi + a))
        b = _slice(zr, mod)
        err = int(np.sum(b[:PREAMBLE_BITS] != PREAMBLE[:len(b)]))
        if best is None or err < best[0]:
            best = (err, b, zr)
    return best[1], {"z": best[2]}


def count_bit_errors(tx_payload, rx_payload):
    """Bit errors; missing received bits count as half an error each (coin flip)."""
    n = min(len(tx_payload), len(rx_payload))
    return float(np.sum(np.asarray(tx_payload)[:n] != np.asarray(rx_payload)[:n])
                 + 0.5 * (len(tx_payload) - n))
