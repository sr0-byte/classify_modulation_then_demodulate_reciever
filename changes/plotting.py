"""All matplotlib code: backend selection, live demo plots and evaluation plots.

Plots are shown in a window (non-blocking) when a GUI backend exists and are
always saved to figures/. Set AMC_NO_GUI=1 to force save-only mode.
"""
import os
import time
import csv
import numpy as np
import matplotlib

# ---- pick a GUI backend if possible, otherwise fall back to file-only (Agg) ----
GUI = True
try:
    if os.environ.get("AMC_NO_GUI") == "1":
        raise RuntimeError("GUI disabled by AMC_NO_GUI")
    if os.environ.get("MPLBACKEND") is None:
        for _b in ("TkAgg", "QtAgg", "Qt5Agg", "MacOSX"):
            try:
                matplotlib.use(_b, force=True)
                import matplotlib.pyplot as _p
                _p.figure()
                _p.close("all")
                break
            except Exception:
                continue
        else:
            raise RuntimeError("no GUI backend")
except Exception:
    matplotlib.use("Agg", force=True)
    GUI = False
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix

from config import MODS, K, SPS, COL, FIGURES_DIR, RESULTS_DIR
from signal_generator import modulate, bits_to_symbols
from dataset import make_dataset
from classifier import simulate_ber, theory_ber

if matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "ps", "cairo", "template"):
    GUI = False

# ----------------------------------------------------------------------
# Look: serif font, warm paper background, dotted grid, full light frame
# ----------------------------------------------------------------------
INK = "#1F2A44"            # text / main lines
PAPER = "#F7F4EC"          # figure background
I_COL, Q_COL = "#0B3C5D", "#E4572E"
MARK = {"BPSK": "s", "BFSK": "^", "QPSK": "D", "16QAM": "v", "DPSK": "P"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.monospace": ["DejaVu Sans Mono"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "figure.dpi": 100, "figure.facecolor": PAPER, "axes.facecolor": "white",
    "axes.edgecolor": "#9AA3B2", "axes.linewidth": 1.0,
    "axes.grid": True, "grid.linestyle": ":", "grid.color": "#9AA3B2", "grid.alpha": 0.55,
    "legend.frameon": False,
})


def _finish(fig, fname, tight=True):
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    path = os.path.join(FIGURES_DIR, fname)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Plot saved: {path}")
    if GUI:
        plt.show(block=False)
        plt.pause(0.5)
    else:
        plt.close(fig)


def hold():
    """Keep the plot windows open until the user closes them."""
    if GUI and plt.get_fignums():
        print("\nClose the plot windows to exit.")
        plt.show()


def _short(s, n=34):
    s = repr(s)
    return s if len(s) <= n else s[:n - 3] + "...'"


def _spectrum(ax, rx_iq):
    fr = np.fft.fftshift(np.fft.fftfreq(len(rx_iq)))
    ax.plot(fr, 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(rx_iq))) ** 2 + 1e-12),
            color=COL["BFSK"], lw=1)
    ax.set_xlabel("normalised frequency")
    ax.set_ylabel("power (dB)")


def _prob_bars(ax, proba, pred):
    y = np.arange(len(MODS))
    ax.barh(y, proba * 100, color=[COL[m] for m in MODS], height=0.6,
            edgecolor=[INK if m == pred else "none" for m in MODS], linewidth=1.8)
    ax.set_yticks(y)
    ax.set_yticklabels(MODS)
    for t, m in zip(ax.get_yticklabels(), MODS):
        if m == pred:
            t.set_fontweight("bold")
    for yi, p in zip(y, proba * 100):
        if p > 70:
            ax.text(p - 1.5, yi, f"{p:.1f}%", va="center", ha="right", fontsize=9, color="white",
                    fontweight="bold")
        else:
            ax.text(p + 1.5, yi, f"{p:.1f}%", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("probability (%)")
    ax.grid(axis="y", visible=False)


# ----------------------------------------------------------------------
# Demo plot 1: message
# ----------------------------------------------------------------------
def plot_message(mod, pred, proba, bits, tx_iq, rx_iq, payload, rpay, info,
                 snr_db, message, text, errs, fname="fig_message_demo.png"):
    npay = len(payload)
    ok = pred == mod
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(3, 6, height_ratios=[0.55, 1, 1.05], hspace=0.6, wspace=1.0)

    # --- overall description (top banner) ---
    ad = fig.add_subplot(gs[0, :])
    ad.set_xticks([]); ad.set_yticks([]); ad.grid(False)
    ad.set_title("Overall description")
    ad.text(0.02, 0.5,
            f"Transmitted modulation : {mod}\n"
            f"Classifier decision    : {pred}  [{'CORRECT' if ok else 'WRONG'}]\n"
            f"Channel SNR            : {snr_db} dB\n"
            f"Payload size           : {npay} bits",
            family="monospace", fontsize=10.5, va="center", transform=ad.transAxes)
    ad.text(0.52, 0.5,
            f"Sent message      : {_short(message)}\n"
            f"Recovered message : {_short(text)}\n"
            f"Bit errors        : {errs:g} / {npay}\n"
            f"BER               : {errs / npay:.4f}",
            family="monospace", fontsize=10.5, va="center", transform=ad.transAxes)
    ad.axvline(0.5, color="#9AA3B2", lw=1)

    # --- (1) transmitted baseband, first 10 symbols ---
    a1 = fig.add_subplot(gs[1, 0:3])
    n10 = 10 * SPS
    t = np.arange(n10) / SPS
    a1.plot(t, tx_iq.real[:n10], color=I_COL, lw=2, label="I")
    a1.plot(t, tx_iq.imag[:n10], color=Q_COL, lw=2, ls="--", label="Q")
    a1.set_title(f"Transmitted baseband ({mod}), first 10 symbols")
    a1.set_xlabel("time (symbols)")
    a1.set_ylabel("amplitude")
    a1.legend(loc="lower right", bbox_to_anchor=(1, 1.01), ncol=2)

    # --- (2) passband, first 7 symbols ---
    a2 = fig.add_subplot(gs[1, 3:6])
    sp = 64
    fc = 4 / sp
    base = modulate(mod, bits[:7 * K[mod]], sps=sp)[:7 * sp]
    pb = np.real(base * np.exp(2j * np.pi * fc * np.arange(len(base))))
    a2.plot(np.arange(len(base)) / sp, pb, color=INK, lw=1.2)
    a2.set_title("Passband signal, first 7 symbols")
    a2.set_xlabel("time (symbols)")
    a2.set_ylabel("amplitude")

    # --- (3) classifier output ---
    a3 = fig.add_subplot(gs[2, 0:2])
    _prob_bars(a3, proba, pred)
    a3.set_title(f"Classifier output -> {pred}")

    # --- (4) received constellation ---
    a4 = fig.add_subplot(gs[2, 2:4])
    if "z" in info:
        z = info["z"]
        a4.scatter(z.real, z.imag, s=14, alpha=0.65, color=COL.get(pred, INK), label="received")
        if pred in ("BPSK", "QPSK", "16QAM"):
            ideal = np.unique(np.round(bits_to_symbols(
                pred, np.random.default_rng(0).integers(0, 2, 4000 * K[pred])), 6))
            a4.scatter(ideal.real, ideal.imag, marker="+", s=140, lw=2, color=INK, label="ideal")
        a4.set_aspect("equal")
        a4.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
        a4.set_xlabel("I")
        a4.set_ylabel("Q")
        a4.set_title("Received constellation")
    else:
        _spectrum(a4, rx_iq)
        a4.set_title("Received spectrum (FSK)")

    # --- (5) first 64 payload bits: sent vs recovered as cell strips ---
    a5 = fig.add_subplot(gs[2, 4:6])
    nb = min(64, npay)
    sent = np.asarray(payload)[:nb]
    rec = np.pad(np.asarray(rpay)[:nb], (0, max(0, nb - len(rpay[:nb]))))
    a5.imshow(np.vstack([sent, rec]), aspect="auto", cmap=ListedColormap(["#E9E4D4", INK]),
              vmin=0, vmax=1, interpolation="nearest")
    bad = np.where(sent != rec)[0]
    a5.plot(bad, np.full(len(bad), 1.0), marker="o", ls="none", mfc="none", mec="#E4002B", ms=10, mew=2)
    a5.set_yticks([0, 1])
    a5.set_yticklabels(["sent", "recovered"])
    a5.set_xticks(np.arange(0, nb, 8))
    a5.set_xlabel("bit index")
    a5.grid(False)
    a5.set_title(f"First {nb} payload bits  (dark = 1, red ring = error)")
    _finish(fig, fname, tight=False)


# ----------------------------------------------------------------------
# Demo plot 2: IQ symbols (constellation-centred layout)
# ----------------------------------------------------------------------
def plot_symbols(pred, proba, tx_iq, rx_iq, info, snr_db, sent=None, fname="fig_symbols_demo.png"):
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1], hspace=0.45, wspace=0.35)

    # left, tall: received symbols (or spectrum)
    ac = fig.add_subplot(gs[:, 0])
    if "z" in info:
        z = info["z"]
        ac.scatter(z.real, z.imag, s=18, alpha=0.6, color=COL.get(pred, INK), label="received")
        if sent is not None:
            ac.scatter(sent.real, sent.imag, marker="X", s=120, color=INK, edgecolor="white",
                       label="sent (with channel phase)")
        ac.axhline(0, color="#9AA3B2", lw=0.8)
        ac.axvline(0, color="#9AA3B2", lw=0.8)
        ac.set_aspect("equal")
        ac.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9)
        ac.set_xlabel("I")
        ac.set_ylabel("Q")
        ac.set_title("Received symbols (integrate & dump)")
    else:
        _spectrum(ac, rx_iq)
        ac.set_title("Received spectrum")

    n8 = 8 * SPS
    t = np.arange(n8) / SPS
    at = fig.add_subplot(gs[0, 1])
    at.plot(t, tx_iq.real[:n8], color=I_COL, lw=2, label="I")
    at.plot(t, tx_iq.imag[:n8], color=Q_COL, lw=2, ls="--", label="Q")
    at.set_title("Modulated IQ, first 8 symbols")
    at.set_xlabel("time (symbols)")
    at.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)

    ar = fig.add_subplot(gs[0, 2], sharey=at)
    ar.plot(t, rx_iq.real[:n8], color=I_COL, lw=1.3, alpha=0.85, label="I")
    ar.plot(t, rx_iq.imag[:n8], color=Q_COL, lw=1.3, alpha=0.85, ls="--", label="Q")
    ar.set_title(f"Received IQ at {snr_db} dB SNR")
    ar.set_xlabel("time (symbols)")

    ab = fig.add_subplot(gs[1, 1:3])
    _prob_bars(ab, proba, pred)
    ab.set_title(f"Classifier decision: {pred}")
    _finish(fig, fname, tight=False)


# ----------------------------------------------------------------------
# Evaluation plots
# ----------------------------------------------------------------------
CM_SNRS = [(-10, "fig_cm_low.png"), (-2, "fig_cm_med.png"), (12, "fig_cm_high.png")]


def _plot_cm(c, sv, fname):
    n = len(MODS)
    cn = c / np.maximum(c.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.6, 5.8))
    im = ax.imshow(cn, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(MODS, rotation=35, ha="left")
    ax.set_yticks(range(n)); ax.set_yticklabels(MODS)
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_xticks(np.arange(-0.5, n), minor=True); ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cn[i, j] * 100:.0f}%", ha="center", va="center", fontsize=10,
                    fontweight="bold" if i == j else "normal",
                    color="white" if cn[i, j] > 0.6 else INK)
    ax.set_xlabel("Predicted modulation")
    ax.set_ylabel("True modulation")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.06)
    cb.set_label("fraction of true class")
    fig.suptitle(f"Confusion matrix at SNR = {sv} dB", x=0.02, ha="left", fontweight="bold", y=0.99)
    _finish(fig, fname)


def plot_classifier_eval(model, n_per=40):
    """Accuracy vs SNR and confusion matrices at -10, -2 and 12 dB (fresh test set)."""
    snr_list = list(range(-10, 21, 2))
    print(f"Building test set ({len(snr_list) * len(MODS) * n_per} frames)...")
    t0 = time.time()
    X, y, snr = make_dataset(snr_list, n_per, seed=2)
    pred = model.predict(X)
    print(f"  done in {time.time() - t0:.1f}s, overall accuracy = {np.mean(pred == y) * 100:.2f} %")

    # accuracy vs SNR
    acc = {m: [] for m in MODS}
    acc["Overall"] = []
    for sv in snr_list:
        idx = snr == sv
        acc["Overall"].append(np.mean(pred[idx] == y[idx]))
        for ci, m in enumerate(MODS):
            j = idx & (y == ci)
            acc[m].append(np.mean(pred[j] == ci))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.axhline(90, color="#9AA3B2", lw=1.2, ls="--")
    ax.text(snr_list[-1], 91, "90 %", ha="right", va="bottom", fontsize=9, color="#6B7385")
    for m in MODS:
        ax.plot(snr_list, np.array(acc[m]) * 100, marker=MARK[m], color=COL[m], lw=1.6,
                ms=6, alpha=0.9, label=m)
    ax.plot(snr_list, np.array(acc["Overall"]) * 100, color=INK, lw=3.2, label="Overall", zorder=5)
    ax.set_xlabel("SNR per sample (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(snr_list[0] - 1, snr_list[-1] + 1)
    ax.set_ylim(0, 104)
    ax.set_title("Classification accuracy across SNR")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=6)
    _finish(fig, "fig_accuracy_vs_snr.png")

    # save the numbers to results/
    path = os.path.join(RESULTS_DIR, "accuracy_vs_snr.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["snr_db"] + MODS + ["Overall"])
        for i, sv in enumerate(snr_list):
            w.writerow([sv] + [f"{acc[m][i]:.4f}" for m in MODS + ["Overall"]])
    print(f"Results saved: {path}")

    # confusion matrices: -10 dB (low), -2 dB (med), 12 dB (high)
    for sv, fname in CM_SNRS:
        idx = snr == sv
        c = confusion_matrix(y[idx], pred[idx], labels=range(len(MODS)))
        _plot_cm(c, sv, fname)


def plot_ber(model, n_frames=40):
    snrs = list(range(-10, 17, 2))
    print(f"Simulating BER ({len(snrs)} SNR points x {len(MODS)} modulations x {n_frames} frames)...")
    t0 = time.time()
    BER = simulate_ber(model, snrs, n_frames)
    print(f"  finished in {time.time() - t0:.1f}s")
    floor = 1e-5
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 9.5), sharex=True, sharey=True)
    for ax, key, title in zip(axes, ["oracle", "e2e"],
                              ["Known modulation (oracle demodulator)",
                               "End-to-end (ML classifier -> demodulator)"]):
        for m in MODS:
            ax.semilogy(snrs, np.maximum(BER[m][key], floor), marker=MARK[m], color=COL[m],
                        lw=1.8, ms=6, label=m)
            t = theory_ber(m, np.array(snrs))
            if t is not None and key == "oracle":
                ax.semilogy(snrs, np.maximum(t, floor), ls="--", color=COL[m], alpha=0.55, lw=1.3)
        ax.set_title(title)
        ax.set_ylim(floor, 0.7)
        ax.set_ylabel("BER")
    axes[1].set_xlabel("SNR per sample (dB)")
    axes[0].plot([], [], ls="--", color="#6B7385", label="theory")
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h, l, loc="upper right", ncol=2, fontsize=9)
    fig.text(0.01, 0.005, "BER floor 1e-5 = no errors seen", fontsize=8.5, color="#6B7385")
    _finish(fig, "fig_ber_vs_snr.png")

    print("\nEnd-to-end BER table (rows = SNR dB)")
    print("SNR   " + "  ".join(f"{m:>9s}" for m in MODS))
    for i, sv in enumerate(snrs):
        print(f"{sv:4d}  " + "  ".join(f"{BER[m]['e2e'][i]:9.2e}" for m in MODS))

    path = os.path.join(RESULTS_DIR, "ber_vs_snr.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["snr_db"] + [f"{m}_{k}" for m in MODS for k in ("oracle", "e2e")])
        for i, sv in enumerate(snrs):
            w.writerow([sv] + [f"{BER[m][k][i]:.3e}" for m in MODS for k in ("oracle", "e2e")])
    print(f"Results saved: {path}")
