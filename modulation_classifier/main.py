"""Interactive terminal demo: classify + demodulate a message and a set of IQ symbols,
then (optionally) produce the evaluation plots.

Run:   python main.py          (needs models/amc_model.joblib -> run  python train.py  first)
"""
import numpy as np

from config import MODS, SPS, MIN_SYMBOLS, PREAMBLE_BITS
from signal_generator import (bits_to_symbols, modulate, build_tx_bits, channel,
                              text_to_bits, bits_to_text)
from demodulator import demodulate, count_bit_errors
from classifier import load_model, classify
import plotting as plotting


# ----------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------
def run_message(model, mod, message, snr_db, seed, fname="fig_message_demo.png"):
    rng = np.random.default_rng(seed)
    payload = text_to_bits(message)
    if len(payload) == 0:
        print("Empty message - nothing to send.")
        return
    bits, npay = build_tx_bits(mod, payload, rng=rng)
    tx_iq = modulate(mod, bits)
    rx_iq = channel(tx_iq, snr_db, rng)
    pred, proba = classify(model, rx_iq)
    rbits, info = demodulate(rx_iq, pred)
    rpay = rbits[PREAMBLE_BITS:PREAMBLE_BITS + npay]
    errs = count_bit_errors(payload, rpay)
    text = bits_to_text(rpay)

    print("\n----- RESULT (message) -----")
    print(f"Classifier decision : {pred}  ({'CORRECT' if pred == mod else 'WRONG'})")
    print(f"Recovered message   : {text!r}")
    print(f"BER                 : {errs / npay:.4f}  ({errs:g}/{npay} bits)")

    plotting.plot_message(mod, pred, proba, bits, tx_iq, rx_iq, payload, rpay, info,
                          snr_db, message, text, errs, fname=fname)
    return {"pred": pred, "text": text, "errs": errs, "npay": npay, "ber": errs / npay}


def run_symbols(model, I_vals, Q_vals, snr_db, phase_deg, seed=3, repeat_to=MIN_SYMBOLS,
                fname="fig_symbols_demo.png"):
    rng = np.random.default_rng(seed)
    s = np.array(I_vals, float) + 1j * np.array(Q_vals, float)
    if np.mean(np.abs(s) ** 2) == 0:
        print("All symbols are zero - nothing to send.")
        return
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    S = np.tile(s, int(np.ceil(repeat_to / len(s))))
    tx_iq = np.repeat(S, SPS)
    rx_iq = channel(tx_iq, snr_db, rng, np.deg2rad(phase_deg))
    pred, proba = classify(model, rx_iq)
    bits, info = demodulate(rx_iq, pred, sync="none")

    print("\n----- RESULT (IQ symbols) -----")
    print(f"Classifier decision : {pred}")
    print(f"Demodulated bits    : {''.join(map(str, bits[:48]))}")
    ser = None
    if pred in ("BPSK", "QPSK", "16QAM"):
        dec = bits_to_symbols(pred, bits)
        n = min(len(dec), len(S))
        ser = np.mean(np.abs(dec[:n] - S[:n]) > 1e-6)
        print(f"Symbol error rate   : {ser:.4f}")

    sent = np.unique(np.round(s * np.exp(1j * np.deg2rad(phase_deg)), 6))
    plotting.plot_symbols(pred, proba, tx_iq, rx_iq, info, snr_db, sent, fname=fname)
    return {"pred": pred, "ser": ser}


# ----------------------------------------------------------------------
# Terminal input
# ----------------------------------------------------------------------
def ask(prompt, cast=str, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            return cast(raw)
        except ValueError:
            print("  invalid value, try again")


def ask_modulation():
    while True:
        m = input("TX_modulation (BPSK / BFSK / QPSK / 16QAM / DPSK): ").strip().upper()
        if m in MODS:
            return m
        print("  choose one of:", ", ".join(MODS))


def ask_numbers(prompt):
    while True:
        raw = input(prompt).replace(",", " ").split()
        try:
            return [float(v) for v in raw]
        except ValueError:
            print("  numbers only, separated by spaces")


def ask_evaluation(model):
    print("\n=== PART 3: evaluation plots (optional) ===")
    print("  1 = accuracy vs SNR + confusion matrices at -10, -2 and 12 dB (~10-60 s)")
    print("  2 = BER vs SNR, oracle and end-to-end (~1-3 min)")
    print("  3 = both of the above")
    print("  Enter = skip")
    raw = input("choice: ").strip()
    if raw in ("1", "3"):
        plotting.plot_classifier_eval(model, n_per=ask("frames per class per SNR (Enter = 40): ", int, default=40))
    if raw in ("2", "3"):
        plotting.plot_ber(model, n_frames=ask("frames per point (Enter = 40): ", int, default=40))


def main():
    model = load_model()

    print("\n=== PART 1: message ===")
    mod = ask_modulation()
    message = input("message: ")
    snr_db = ask("SNR_DB: ", float)
    seed = ask("SEED (Enter = 42): ", int, default=42)
    run_message(model, mod, message, snr_db, seed)

    print("\n=== PART 2: IQ symbols (press Enter on I_val to skip) ===")
    I_vals = ask_numbers("I_val: ")
    if not I_vals:
        ask_evaluation(model)
        plotting.hold()
        return
    while True:
        Q_vals = ask_numbers("Q_val: ")
        if len(Q_vals) == len(I_vals):
            break
        print(f"  Q_val needs {len(I_vals)} numbers (you gave {len(Q_vals)})")
    snr2 = ask("snr_DB2: ", float)
    phase2 = ask("phase_deg_2 (Enter = 0): ", float, default=0.0)
    run_symbols(model, I_vals, Q_vals, snr2, phase2)
    ask_evaluation(model)
    plotting.hold()


if __name__ == "__main__":
    main()
