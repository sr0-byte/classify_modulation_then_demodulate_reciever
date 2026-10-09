"""Train the modulation classifier and save it to models/amc_model.joblib.

Usage:
    python train.py                       # default: SNR -10..20 dB, 150 frames/class/SNR
    python train.py --n-per 300 --eval    # bigger set + save evaluation plots to figures/
"""
import argparse
import os
import time
import numpy as np

from config import MODS, MODEL_PATH


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snr-min", type=int, default=-10)
    ap.add_argument("--snr-max", type=int, default=20)
    ap.add_argument("--snr-step", type=int, default=2)
    ap.add_argument("--n-per", type=int, default=150, help="training frames per class per SNR")
    ap.add_argument("--n-test", type=int, default=40, help="test frames per class per SNR")
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--eval", action="store_true", help="also save evaluation figures (no GUI)")
    ap.add_argument("--out", default=MODEL_PATH)
    args = ap.parse_args()

    if args.eval:
        os.environ["AMC_NO_GUI"] = "1"   # must be set before plotting is imported

    from dataset import make_dataset
    from classifier import build_model, save_model

    snrs = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    print(f"Building training set: {len(snrs)} SNRs x {len(MODS)} classes x {args.n_per} frames...")
    t0 = time.time()
    Xtr, ytr, _ = make_dataset(snrs, args.n_per, seed=1)
    Xte, yte, _ = make_dataset(snrs, args.n_test, seed=2)
    print(f"  done in {time.time() - t0:.1f}s  (train {len(ytr)}, test {len(yte)})")

    print("Training random forest...")
    model = build_model(n_estimators=args.trees)
    model.fit(Xtr, ytr)
    acc = np.mean(model.predict(Xte) == yte)
    print(f"Test accuracy (all SNRs): {acc * 100:.2f} %")

    save_model(model, args.out)
    print(f"Model saved: {args.out}")

    if args.eval:
        import modulation_classifier.plotting as plotting
        plotting.plot_classifier_eval(model, n_per=args.n_test)


if __name__ == "__main__":
    main()
