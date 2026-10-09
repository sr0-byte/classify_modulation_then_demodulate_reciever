"""Feature extraction: high-order cumulants, amplitude statistics, spectral/FSK features."""
import numpy as np
from config import SPS


def symbol_samples(x, sps=SPS):
    n = len(x) // sps
    return x[:n * sps].reshape(n, sps).mean(axis=1)


def extract_features(x, sps=SPS):
    x = np.asarray(x, dtype=complex)
    fd = 1.0 / sps
    z = symbol_samples(x, sps)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2) + 1e-12)

    # moments / cumulants
    m20, m21 = np.mean(z ** 2), np.mean(np.abs(z) ** 2)
    m40, m41, m42 = np.mean(z ** 4), np.mean(z ** 3 * np.conj(z)), np.mean(np.abs(z) ** 4)
    c20 = abs(m20)
    c40 = abs(m40 - 3 * m20 ** 2)
    c41 = abs(m41 - 3 * m20 * m21)
    c42 = float((m42 - abs(m20) ** 2 - 2 * m21 ** 2).real)

    # amplitude statistics
    a = np.abs(z)
    amp_std = a.std() / (a.mean() + 1e-12)
    amp_kurt = np.mean((a - a.mean()) ** 4) / (a.var() + 1e-12) ** 2

    # spectral features
    P = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    P = P / P.sum()
    f = np.fft.fftshift(np.fft.fftfreq(len(x)))
    bw = 0.03
    p_fsk = P[np.abs(f - fd) < bw].sum() + P[np.abs(f + fd) < bw].sum()
    p_dc = P[np.abs(f) < bw].sum()
    ratio = p_fsk / (p_fsk + p_dc + 1e-12)

    # instantaneous frequency
    finst = np.angle(x[1:] * np.conj(x[:-1])) / (2 * np.pi)
    f_med = np.median(np.abs(finst))
    return np.array([c20, c40, c41, c42, amp_std, amp_kurt, p_fsk, p_dc, ratio, f_med])
