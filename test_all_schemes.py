"""Run EVERY modulation scheme on the same message and save the demo figures for each one.

    python test_all_schemes.py                    # DEMO mode: low SNR per scheme, shows real bit errors
    python test_all_schemes.py --snr 0            # same SNR (dB) for every scheme
    python test_all_schemes.py --strict           # 12 dB, asserts exact recovery + correct classification
    python test_all_schemes.py --sym-snr 6 --seed 7
    python -m unittest test_all_schemes -v        # demo mode with defaults

Demo mode REPORTS what happened (classified as, bit errors, BER, recovered text) and only fails
if something crashes or the BER is impossible. Strict mode is the pass/fail regression test.

Outputs
    figures/all_schemes/fig_message_<SCHEME>.png   (5 message figures)
    figures/all_schemes/fig_symbols_<SCHEME>.png   (5 symbol figures)
    results/all_schemes_summary.csv

Needs the trained model:  python train.py
"""
import os
os.environ.setdefault("AMC_NO_GUI", "1")      # save-only; to see windows run:  set AMC_NO_GUI=0  (Windows)

import argparse
import csv
import sys
import unittest
import numpy as np

from config import MODS, K, MIN_SYMBOLS, RESULTS_DIR
from signal_generator import text_to_bits, bits_to_symbols, modulate, channel
from demodulator import demodulate
from classifier import load_model, classify
from main import run_message, run_symbols
import plotting

MESSAGE = "Naan thaan LEO LEO DASS!"
OUT = "all_schemes"                      # sub-folder of figures/

# ---- settings (changed by the command-line options) ----
# SNR per sample (dB). Chosen so every scheme is in its "a few errors" region.
DEMO_SNR = {"BPSK": -4.0, "QPSK": -2.0, "BFSK": -2.0, "16QAM": 2.0, "DPSK": 0.0}
STRICT_SNR = 12.0
CFG = {"strict": False, "snr": None, "sym_snr": 10.0, "seed": 42, "sym_seed": 3, "sym_phase": 0.0}


def msg_snr(mod):
    if CFG["snr"] is not None:
        return CFG["snr"]
    return STRICT_SNR if CFG["strict"] else DEMO_SNR[mod]


def symbol_inputs(mod):
    """IQ symbol sequence that represents this scheme, built from the message bits."""
    bits = text_to_bits(MESSAGE)
    bits = np.pad(bits, (0, (-len(bits)) % K[mod]))
    s = bits_to_symbols(mod, bits)
    return list(s.real), list(s.imag)


def run_bfsk_waveform(model, snr_db, seed, fname):
    """BFSK has no IQ-symbol representation, so the waveform itself is sent through the channel."""
    rng = np.random.default_rng(seed)
    bits = text_to_bits(MESSAGE)
    bits = np.tile(bits, int(np.ceil(MIN_SYMBOLS / len(bits))))
    tx = modulate("BFSK", bits)
    rx = channel(tx, snr_db, rng, 0.0)
    pred, proba = classify(model, rx)
    _, info = demodulate(rx, pred, sync="none")
    plotting.plot_symbols(pred, proba, tx, rx, info, snr_db, None, fname=fname)
    return {"pred": pred, "ser": None}


class TestAllSchemes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()
        cls.rows = {m: {"scheme": m} for m in MODS}

    # ------------------------------------------------------------------
    def test_message_all_schemes(self):
        for mod in MODS:
            with self.subTest(scheme=mod):
                snr = msg_snr(mod)
                r = run_message(self.model, mod, MESSAGE, snr, CFG["seed"],
                                fname=f"{OUT}/fig_message_{mod}.png")
                self.rows[mod].update(msg_snr=snr, msg_pred=r["pred"], msg_recovered=r["text"],
                                      msg_errors=r["errs"], msg_ber=r["ber"])
                self.assertTrue(0.0 <= r["ber"] <= 1.0)          # always: sanity
                if CFG["strict"]:                                 # strict: pass/fail
                    self.assertEqual(r["pred"], mod, "classifier picked the wrong scheme")
                    self.assertEqual(r["text"], MESSAGE, "message not recovered exactly")
                    self.assertEqual(r["errs"], 0)

    # ------------------------------------------------------------------
    def test_symbols_all_schemes(self):
        for mod in MODS:
            with self.subTest(scheme=mod):
                fname = f"{OUT}/fig_symbols_{mod}.png"
                snr = CFG["sym_snr"]
                if mod == "BFSK":
                    r = run_bfsk_waveform(self.model, snr, CFG["sym_seed"], fname)
                else:
                    I, Q = symbol_inputs(mod)
                    r = run_symbols(self.model, I, Q, snr, CFG["sym_phase"],
                                    seed=CFG["sym_seed"], fname=fname)
                self.rows[mod].update(sym_snr=snr, sym_pred=r["pred"], sym_ser=r["ser"])
                if CFG["strict"]:
                    self.assertEqual(r["pred"], mod, "classifier picked the wrong scheme")

    # ------------------------------------------------------------------
    @classmethod
    def tearDownClass(cls):
        mode = "STRICT" if CFG["strict"] else "DEMO"
        print("\n" + "=" * 92)
        print(f"{mode} mode   message = {MESSAGE!r}   seed = {CFG['seed']}")
        print(f"{'scheme':7s} {'SNR dB':>6s} {'classified':>10s} {'errors':>7s} {'BER':>8s}   recovered text")
        for m in MODS:
            r = cls.rows[m]
            if "msg_ber" not in r:
                continue
            flag = "" if r["msg_pred"] == m else "  (WRONG scheme)"
            print(f"{m:7s} {r['msg_snr']:6.1f} {r['msg_pred']:>10s} {r['msg_errors']:7g} "
                  f"{r['msg_ber']:8.4f}   {r['msg_recovered']!r}{flag}")
        print(f"\nSymbol test at {CFG['sym_snr']} dB -> classified as: "
              + ", ".join(f"{m}:{cls.rows[m].get('sym_pred')}" for m in MODS))
        path = os.path.join(RESULTS_DIR, "all_schemes_summary.csv")
        keys = ["scheme", "msg_snr", "msg_pred", "msg_recovered", "msg_errors", "msg_ber",
                "sym_snr", "sym_pred", "sym_ser"]
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for m in MODS:
                w.writerow({k: cls.rows[m].get(k, "") for k in keys})
        print(f"Summary saved: {path}")
        print("Figures in   : figures/" + OUT + "/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="12 dB, assert exact recovery and classification")
    ap.add_argument("--snr", type=float, help="use this SNR (dB) for the message test of every scheme")
    ap.add_argument("--sym-snr", type=float, default=CFG["sym_snr"], help="SNR (dB) for the symbol test")
    ap.add_argument("--seed", type=int, default=CFG["seed"], help="noise/phase seed for the message test")
    args, rest = ap.parse_known_args()
    CFG.update(strict=args.strict, snr=args.snr, sym_snr=args.sym_snr, seed=args.seed)
    unittest.main(argv=[sys.argv[0]] + rest, verbosity=2)


if __name__ == "__main__":
    main()
