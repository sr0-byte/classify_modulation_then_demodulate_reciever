"""Dataset synthesis: random frames -> channel -> feature vectors."""
import numpy as np
from config import MODS, K, PREAMBLE_BITS
from signal_generator import modulate, build_tx_bits, channel
from features import extract_features


def make_dataset(snrs, n_per, seed, nsym_choices=(128, 256, 512)):
    """Returns X (features), y (class index into MODS), s (SNR of each frame)."""
    rng = np.random.default_rng(seed)
    X, y, s = [], [], []
    for snr in snrs:
        for ci, mod in enumerate(MODS):
            for _ in range(n_per):
                nsym = int(rng.choice(nsym_choices))
                payload = rng.integers(0, 2, nsym * K[mod] - PREAMBLE_BITS)
                bits, _ = build_tx_bits(mod, payload, min_symbols=nsym, rng=rng)
                X.append(extract_features(channel(modulate(mod, bits), snr, rng)))
                y.append(ci)
                s.append(snr)
    return np.array(X), np.array(y), np.array(s)
