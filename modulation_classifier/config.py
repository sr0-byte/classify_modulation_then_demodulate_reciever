"""Shared settings. Everything that must match between training and testing lives here."""
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(MODELS_DIR, "amc_model.joblib")

for _d in (MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

SPS = 8                                   # samples per symbol
MODS = ["BPSK", "BFSK", "QPSK", "16QAM", "DPSK"]
K = {"BPSK": 1, "BFSK": 1, "QPSK": 2, "16QAM": 4, "DPSK": 2}   # bits per symbol
PREAMBLE_BITS = 64
PREAMBLE = np.random.default_rng(12345).integers(0, 2, PREAMBLE_BITS)
MIN_SYMBOLS = 256
FEATURE_NAMES = ["|C20|", "|C40|", "|C41|", "C42", "amp_std", "amp_kurt",
                 "P_fsk", "P_dc", "FSK_ratio", "med|f_inst|"]
COL = {"BPSK": "#3A86FF", "BFSK": "#FF006E", "QPSK": "#06A77D",
       "16QAM": "#FB8500", "DPSK": "#8338EC"}
