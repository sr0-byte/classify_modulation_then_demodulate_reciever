"""ML classifier helpers (build / save / load / classify) and BER formulas + BER simulation."""
import os
import joblib
import numpy as np
from scipy.special import erfc
from sklearn.ensemble import RandomForestClassifier

from config import MODS, K, SPS, PREAMBLE_BITS, MODEL_PATH, FEATURE_NAMES
from signal_generator import modulate, build_tx_bits, channel
from features import extract_features
from demodulator import demodulate, count_bit_errors


# ---------------------------- model helpers ----------------------------
def build_model(n_estimators=300, seed=0):
    return RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)


def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": model, "mods": MODS, "features": FEATURE_NAMES}, path)


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run  python train.py  first.")
    model = joblib.load(path)["model"]
    # Single-threaded inference: worker threads garbage-collecting Tk plot objects
    # cause "main thread is not in main loop" errors, and 1-frame predictions are
    # faster without thread overhead anyway.
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    return model


def classify(model, iq):
    """Returns (predicted modulation name, probability vector over MODS)."""
    p = model.predict_proba(extract_features(iq)[None, :])[0]
    return MODS[int(np.argmax(p))], p


# ---------------------------- BER formulas ----------------------------
def theory_ber(mod, snr_db):
    g = 10 ** (np.asarray(snr_db) / 10) * SPS / K[mod]
    if mod in ("BPSK", "QPSK"):
        return 0.5 * erfc(np.sqrt(g))
    if mod == "16QAM":
        return 3 / 8 * erfc(np.sqrt(0.4 * g))
    if mod == "BFSK":
        return 0.5 * np.exp(-g / 2)
    return None   # no closed form used for DPSK


# ---------------------------- BER simulation ----------------------------
def simulate_ber(model, snrs, n_frames=40, payload_bits=512, seed=7, verbose=True):
    """BER with a known modulation ('oracle') and end-to-end (classifier -> demodulator)."""
    rng = np.random.default_rng(seed)
    R = {m: {"oracle": [], "e2e": []} for m in MODS}
    for snr in snrs:
        for mod in MODS:
            e_o = e_e = 0.0
            tot = 0
            for _ in range(n_frames):
                payload = rng.integers(0, 2, payload_bits)
                bits, _ = build_tx_bits(mod, payload, rng=rng)
                iq = channel(modulate(mod, bits), snr, rng)
                rb, _ = demodulate(iq, mod)
                eo = count_bit_errors(payload, rb[PREAMBLE_BITS:PREAMBLE_BITS + payload_bits])
                pm, _ = classify(model, iq)
                if pm == mod:
                    ee = eo
                else:
                    rb2, _ = demodulate(iq, pm)
                    ee = count_bit_errors(payload, rb2[PREAMBLE_BITS:PREAMBLE_BITS + payload_bits])
                e_o += eo
                e_e += ee
                tot += payload_bits
            R[mod]["oracle"].append(e_o / tot)
            R[mod]["e2e"].append(e_e / tot)
        if verbose:
            print(f"  SNR {snr:3d} dB done")
    return R
