"""Transmitter side: bits -> symbols -> modulated IQ, plus the AWGN / phase-offset channel."""
import numpy as np
from config import SPS, K, PREAMBLE, PREAMBLE_BITS, MIN_SYMBOLS


def bits_to_symbols(mod, bits):
    b = np.asarray(bits, dtype=int)
    if mod == "BPSK":
        return (1 - 2 * b).astype(complex)
    if mod == "QPSK":
        b = b.reshape(-1, 2)
        return ((1 - 2 * b[:, 0]) + 1j * (1 - 2 * b[:, 1])) / np.sqrt(2)
    if mod == "16QAM":
        b = b.reshape(-1, 4)

        def lvl(b1, b0):
            return np.select([(b1 == 0) & (b0 == 0), (b1 == 0) & (b0 == 1),
                              (b1 == 1) & (b0 == 1), (b1 == 1) & (b0 == 0)],
                             [-3, -1, 1, 3])
        return (lvl(b[:, 0], b[:, 1]) + 1j * lvl(b[:, 2], b[:, 3])) / np.sqrt(10)
    if mod == "DPSK":
        b = b.reshape(-1, 2)
        dphi = np.select([(b[:, 0] == 0) & (b[:, 1] == 0), (b[:, 0] == 0) & (b[:, 1] == 1),
                          (b[:, 0] == 1) & (b[:, 1] == 1), (b[:, 0] == 1) & (b[:, 1] == 0)],
                         [np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4])
        return np.exp(1j * np.concatenate([[0.0], np.cumsum(dphi)]))
    raise ValueError("unsupported modulation " + mod)


def modulate(mod, bits, sps=SPS):
    if mod == "BFSK":
        fd = 1.0 / sps
        f = np.repeat(np.where(np.asarray(bits) == 1, fd, -fd), sps)
        return np.exp(1j * 2 * np.pi * np.cumsum(f))
    return np.repeat(bits_to_symbols(mod, bits), sps)


def build_tx_bits(mod, payload_bits, min_symbols=MIN_SYMBOLS, rng=None):
    """Preamble + payload + random filler, padded to a whole number of symbols."""
    rng = rng or np.random.default_rng()
    k = K[mod]
    payload = np.asarray(payload_bits, dtype=int)
    n = PREAMBLE_BITS + len(payload)
    n_target = int(np.ceil(max(n, min_symbols * k) / k) * k)
    filler = rng.integers(0, 2, n_target - n)
    return np.concatenate([PREAMBLE, payload, filler]), len(payload)


def channel(iq, snr_db, rng, phase=None):
    """Random (or given) carrier phase + complex AWGN. snr_db=None -> noiseless."""
    if phase is None:
        phase = rng.uniform(0, 2 * np.pi)
    y = iq * np.exp(1j * phase)
    if snr_db is not None:
        sigma2 = 10 ** (-snr_db / 10)
        y = y + np.sqrt(sigma2 / 2) * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))
    return y


def text_to_bits(s):
    return np.array([int(b) for ch in s.encode("utf-8") for b in format(ch, "08b")], dtype=int)


def bits_to_text(bits):
    bits = np.asarray(bits, dtype=int)
    n = len(bits) // 8 * 8
    by = bytes(int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, n, 8))
    return by.decode("utf-8", errors="replace")
