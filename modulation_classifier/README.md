# Modulation Classifier (AMC + Demodulation)

Automatic modulation classification (BPSK, BFSK, QPSK, 16QAM, DPSK) with a
random-forest classifier on high-order cumulant and spectral features, followed
by a demodulator that recovers the transmitted message and reports the BER.


## Structure

```
modulation_classifier/
├── main.py                # Interactive entry point (message demo, IQ-symbol demo, evaluation menu)
├── config.py              # Shared constants (SPS, MODS, preamble, paths, colours)
├── signal_generator.py    # Bits -> symbols -> modulated IQ, AWGN/phase channel, text<->bits
├── features.py            # Cumulants (C20/C40/C41/C42), amplitude stats, spectral features
├── classifier.py          # Build/save/load/classify helpers, theory BER formulas, BER simulation
├── demodulator.py         # Integrate-and-dump, phase sync (preamble), slicing, bit-error counting
├── dataset.py             # Dataset synthesis (frames -> channel -> feature vectors)
├── plotting.py            # Plot style, backend selection, demo and evaluation plots
├── train.py               # Train the model and save it to models/
├── test.py                # Unit tests
├── models/                # Trained model (amc_model.joblib)
├── figures/               # Saved plots (.png)
├── results/               # CSV outputs (accuracy_vs_snr.csv, ber_vs_snr.csv)
├── requirements.txt
└── README.md
```

Module dependencies (no cycles):
`config` <- `signal_generator`, `features`, `demodulator` <- `dataset`, `classifier` <- `plotting` <- `main`

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python train.py            # trains and writes models/amc_model.joblib
python train.py --eval     # same, plus saves evaluation figures (no GUI)
python main.py             # interactive demo
python test.py             # unit tests
```

`train.py` options: `--snr-min`, `--snr-max`, `--snr-step`, `--n-per`, `--n-test`, `--trees`, `--out`.

### Demo flow (`main.py`)
1. **Message**: choose a modulation, type a message and SNR. The signal is sent through the
   channel, classified, demodulated, and the recovered text and BER are shown.
2. **IQ symbols**: enter I and Q values, SNR and a phase offset to see the classification
   and the received constellation.
3. **Evaluation plots** (optional): accuracy vs SNR, confusion matrices at -10 dB (`fig_cm_low`),
   -2 dB (`fig_cm_med`) and 12 dB (`fig_cm_high`), and BER vs SNR (oracle and end-to-end).

### Figures (`figures/`)
| File | Content |
|---|---|
| `fig_message_demo.png` | Overall description, TX baseband (first 10 symbols), passband (first 7 symbols), classifier output, received constellation, first 64 payload bits |
| `fig_symbols_demo.png` | Received symbols, modulated and received IQ, classifier decision |
| `fig_accuracy_vs_snr.png` | Accuracy per modulation and overall vs SNR |
| `fig_cm_low/med/high.png` | Confusion matrices at -10 / -2 / 12 dB |
| `fig_ber_vs_snr.png` | BER vs SNR, oracle (with theory) above and end-to-end below |

Plots open in a window when a GUI backend is available and are always saved in `figures/`.
Set `AMC_NO_GUI=1` to force save-only mode.

## Notes
- The feature settings in `config.py` and `features.py` must stay the same between training
  and testing, otherwise a saved model will not match.
- BER theory curves are drawn for BPSK, QPSK, 16QAM and BFSK (no closed form is used for DPSK).

```text

Test BPSK:
I_val: 1 -1 1 -1
Q_val: 0 0 0 0

Test QPSK:
I_val: 1 1 -1 -1
Q_val: 1 -1 -1 1

Test 16-QAM (Outer 4 corners):
I_val: 3 3 -3 -3
Q_val: 3 -3 -3 3

DPSK 
I_val: 1.0  0.7071  0.0  -0.7071  -1.0
Q_val: 0.0  0.7071  1.0   0.7071   0.0
```