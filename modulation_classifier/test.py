"""Unit tests.   Run:  python test.py      (or: python -m unittest test -v)"""
import unittest
import numpy as np

from config import MODS, K, SPS, PREAMBLE_BITS, MIN_SYMBOLS, FEATURE_NAMES
from signal_generator import (bits_to_symbols, modulate, build_tx_bits, channel,
                              text_to_bits, bits_to_text)
from features import extract_features, symbol_samples
from demodulator import demodulate, count_bit_errors
from dataset import make_dataset
from classifier import build_model, classify, theory_ber


class TestSignalGenerator(unittest.TestCase):
    def test_text_roundtrip(self):
        for s in ["hello MAKKALEY", "Test 123 !", "héllo ✓"]:
            self.assertEqual(bits_to_text(text_to_bits(s)), s)

    def test_symbol_power_is_unit(self):
        rng = np.random.default_rng(0)
        for mod in ["BPSK", "QPSK", "16QAM"]:
            s = bits_to_symbols(mod, rng.integers(0, 2, 20000 * K[mod]))
            self.assertAlmostEqual(np.mean(np.abs(s) ** 2), 1.0, delta=0.05)

    def test_constellation_sizes(self):
        rng = np.random.default_rng(0)
        for mod, n in [("BPSK", 2), ("QPSK", 4), ("16QAM", 16)]:
            s = bits_to_symbols(mod, rng.integers(0, 2, 20000 * K[mod]))
            self.assertEqual(len(np.unique(np.round(s, 6))), n)

    def test_modulate_length_and_constant_envelope(self):
        bits = np.random.default_rng(1).integers(0, 2, 64)
        self.assertEqual(len(modulate("BPSK", bits)), 64 * SPS)
        self.assertTrue(np.allclose(np.abs(modulate("BFSK", bits)), 1.0))

    def test_build_tx_bits_padding(self):
        rng = np.random.default_rng(2)
        for mod in MODS:
            bits, n = build_tx_bits(mod, [1, 0, 1], rng=rng)
            self.assertEqual(n, 3)
            self.assertEqual(len(bits) % K[mod], 0)
            self.assertGreaterEqual(len(bits), MIN_SYMBOLS * K[mod])

    def test_channel_noiseless_preserves_power(self):
        rng = np.random.default_rng(3)
        x = modulate("QPSK", rng.integers(0, 2, 200))
        y = channel(x, None, rng)
        self.assertAlmostEqual(np.mean(np.abs(y) ** 2), np.mean(np.abs(x) ** 2))

    def test_channel_noise_level(self):
        rng = np.random.default_rng(4)
        x = np.ones(200000, complex)
        y = channel(x, 10.0, rng, phase=0.0)
        self.assertAlmostEqual(np.var(y - x), 10 ** (-1.0), delta=0.01)


class TestFeatures(unittest.TestCase):
    def test_shape_and_names(self):
        x = modulate("QPSK", np.random.default_rng(0).integers(0, 2, 512))
        f = extract_features(x)
        self.assertEqual(f.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.all(np.isfinite(f)))

    def test_symbol_samples(self):
        self.assertEqual(len(symbol_samples(np.ones(80, complex))), 80 // SPS)

    def test_cumulants_separate_bpsk_and_qpsk(self):
        rng = np.random.default_rng(5)
        fb = extract_features(modulate("BPSK", rng.integers(0, 2, 2048)))
        fq = extract_features(modulate("QPSK", rng.integers(0, 2, 4096)))
        self.assertGreater(fb[0], 0.9)    # |C20| ~ 1 for BPSK
        self.assertLess(fq[0], 0.1)       # |C20| ~ 0 for QPSK


class TestDemodulator(unittest.TestCase):
    def test_noiseless_roundtrip_all_modulations(self):
        for mod in MODS:
            rng = np.random.default_rng(10)
            payload = rng.integers(0, 2, 400)
            bits, n = build_tx_bits(mod, payload, rng=rng)
            iq = channel(modulate(mod, bits), None, rng)      # random phase, no noise
            rb, _ = demodulate(iq, mod)
            rx = rb[PREAMBLE_BITS:PREAMBLE_BITS + n]
            self.assertEqual(count_bit_errors(payload, rx), 0, msg=mod)

    def test_high_snr_message_recovered(self):
        rng = np.random.default_rng(11)
        payload = text_to_bits("Hi there")
        bits, n = build_tx_bits("16QAM", payload, rng=rng)
        iq = channel(modulate("16QAM", bits), 30.0, rng)
        rb, _ = demodulate(iq, "16QAM")
        self.assertEqual(bits_to_text(rb[PREAMBLE_BITS:PREAMBLE_BITS + n]), "Hi there")

    def test_count_bit_errors(self):
        self.assertEqual(count_bit_errors([0, 1, 1, 0], [0, 1, 1, 0]), 0)
        self.assertEqual(count_bit_errors([0, 1, 1, 0], [1, 1, 1, 0]), 1)
        self.assertEqual(count_bit_errors([0, 1, 1, 0], [0, 1]), 1.0)   # 2 missing bits = 1 error


class TestBerFormulas(unittest.TestCase):
    def test_monotonic_decreasing(self):
        snr = np.arange(-10, 15, 2)
        for mod in ["BPSK", "QPSK", "16QAM", "BFSK"]:
            t = theory_ber(mod, snr)
            self.assertTrue(np.all(np.diff(t) <= 0), msg=mod)

    def test_dpsk_has_no_formula(self):
        self.assertIsNone(theory_ber("DPSK", 5))

    def test_bpsk_better_than_16qam(self):
        self.assertLess(theory_ber("BPSK", 5), theory_ber("16QAM", 5))


class TestDatasetAndClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snrs = [0, 10, 20]
        cls.Xtr, cls.ytr, _ = make_dataset(snrs, 20, seed=1)
        cls.Xte, cls.yte, _ = make_dataset(snrs, 10, seed=2)
        cls.model = build_model(n_estimators=100).fit(cls.Xtr, cls.ytr)

    def test_dataset_shapes(self):
        self.assertEqual(self.Xtr.shape, (3 * len(MODS) * 20, len(FEATURE_NAMES)))
        self.assertEqual(set(np.unique(self.ytr)), set(range(len(MODS))))

    def test_dataset_is_reproducible(self):
        X2, _, _ = make_dataset([0], 3, seed=99)
        X3, _, _ = make_dataset([0], 3, seed=99)
        np.testing.assert_allclose(X2, X3)

    def test_accuracy_reasonable(self):
        acc = np.mean(self.model.predict(self.Xte) == self.yte)
        self.assertGreater(acc, 0.85)

    def test_classify_returns_valid_output(self):
        rng = np.random.default_rng(7)
        bits, _ = build_tx_bits("QPSK", rng.integers(0, 2, 300), rng=rng)
        pred, p = classify(self.model, channel(modulate("QPSK", bits), 20, rng))
        self.assertIn(pred, MODS)
        self.assertAlmostEqual(p.sum(), 1.0)
        self.assertEqual(pred, "QPSK")


if __name__ == "__main__":
    unittest.main(verbosity=2)
