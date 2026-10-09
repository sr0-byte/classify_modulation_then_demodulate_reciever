"""Dataset synthesis: random frames -> channel -> feature vectors."""
import argparse
import os
import numpy as np
from config import MODS, K, PREAMBLE_BITS, FEATURE_NAMES, DATA_DIR
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


def save_dataset(out_dir, Xtr, ytr, str_, Xte, yte, ste):
    """Write the .npy files (X = features, y = class index into MODS, snr = SNR of each frame)."""
    os.makedirs(out_dir, exist_ok=True)
    files = {"X_train": Xtr, "y_train": ytr, "snr_train": str_,
             "X_test": Xte, "y_test": yte, "snr_test": ste,
             "X": np.vstack([Xtr, Xte]), "y": np.concatenate([ytr, yte]),
             "snr": np.concatenate([str_, ste])}
    for name, arr in files.items():
        np.save(os.path.join(out_dir, name + ".npy"), arr)
    with open(os.path.join(out_dir, "labels.txt"), "w", encoding="utf-8") as fh:
        fh.write("class index -> modulation\n")
        fh.writelines(f"{i} = {m}\n" for i, m in enumerate(MODS))
        fh.write("\nfeature columns of X\n")
        fh.writelines(f"{i} = {n}\n" for i, n in enumerate(FEATURE_NAMES))
    print(f"Saved to {out_dir}:")
    for name, arr in files.items():
        print(f"  {name + '.npy':15s} shape {arr.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the dataset and save it as .npy files "
                                             "(same defaults and seeds as train.py).")
    ap.add_argument("--snr-min", type=int, default=-10)
    ap.add_argument("--snr-max", type=int, default=20)
    ap.add_argument("--snr-step", type=int, default=2)
    ap.add_argument("--n-per", type=int, default=150, help="train frames per class per SNR")
    ap.add_argument("--n-test", type=int, default=40, help="test frames per class per SNR")
    ap.add_argument("--out", default=DATA_DIR, help="output folder (default: data/)")
    a = ap.parse_args()
    snrs = list(range(a.snr_min, a.snr_max + 1, a.snr_step))
    Xtr, ytr, str_ = make_dataset(snrs, a.n_per, seed=1)
    Xte, yte, ste = make_dataset(snrs, a.n_test, seed=2)
    save_dataset(a.out, Xtr, ytr, str_, Xte, yte, ste)
