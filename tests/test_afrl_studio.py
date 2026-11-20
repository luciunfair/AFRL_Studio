"""Automated regression and round-trip unit tests for AFRL Studio v1.7.

Run with:
    python -m unittest discover -s tests -v
"""

import csv
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _sub in (
    "",
    "fragment_extractor",
    "feature_extractor",
    os.path.join("feature_extractor", "NIST"),
    os.path.join("feature_extractor", "HEDGE_specials"),
    os.path.join("feature_extractor", "BLFCP_specials"),
    os.path.join("feature_extractor", "SOTA_specials"),
    os.path.join("feature_extractor", "AFRL_specials"),
    os.path.join("feature_extractor", "EnCoD_specials"),
    os.path.join("feature_extractor", "Triumvir_specials"),
    "combine_csv",
):
    p = os.path.join(REPO_ROOT, _sub) if _sub else REPO_ROOT
    if p not in sys.path:
        sys.path.insert(0, p)

from AFRL_fspecials import AFRL_test, local_nibble_variance
from HEDGE_fspecials import HEDGE_test, chi2_uniform, chi_test_absolute, chi_test_confidence
from BLFCP_fspecials import (
    byte_frequency_distribution,
    discrete_fourier_transform,
    lempel_ziv_complexity,
    monte_carlo_pi_approximation,
)
from SOTA_fspecials import (
    byte_statistical_moments,
    ks_uniformity_test,
    wasserstein_uniform_distance,
    index_of_coincidence,
    generalized_entropies,
    serial_and_lag_autocorrelation,
    markov_bigram_features,
    byte_difference_features,
    recompression_ratios,
    haar_wavelet_detail_ratio,
    spectral_flatness_measure,
    structural_run_and_ascii,
)
from nist_suite import (
    monobit_test,
    frequency_within_block_test,
    runs_test,
    longest_run_ones_in_a_block_test,
    binary_matrix_rank_test,
    dft_test,
    non_overlapping_template_matching_test,
    overlapping_template_matching_test,
    maurers_universal_test,
    linear_complexity_test,
    serial_test,
    approximate_entropy_test,
    cumulative_sums_test,
    random_excursion_test,
    random_excursion_variant_test,
)
from EnCoD_fspecials import (
    train_encod_model,
    load_encod_model,
    inspect_encod_model_summary,
    predict_encod_batch,
)
from BFLCP_dnn import (
    train_bflcp_model,
    load_bflcp_model,
    inspect_bflcp_model_summary,
    predict_bflcp_batch,
)
from Triumvir_ensemble import (
    train_triumvir_model,
    load_triumvir_model,
    inspect_triumvir_model_summary,
    predict_triumvir_batch,
)
from extract_whole_folder_fragments import extract_whole_folder_fragments
from extract_each_file_fragments import extract_random_fragments_file
from combine import concatenate_rows_gui, concatenate_columns_gui


class _DummyElement:
    def update(self, *args, **kwargs):
        return None


class _DummyWindow:
    """Headless mock for FreeSimpleGUI Window during Stage 1 & Stage 4 tests."""

    def __init__(self):
        self._el = _DummyElement()

    def __getitem__(self, key):
        return self._el

    def read(self, timeout=0):
        return ("__TIMEOUT__", {})

    def refresh(self):
        return None


def _write_uint8_csv(path: str, arr: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in arr:
            writer.writerow(row.tolist())


class TestStage1FragmentExtraction(unittest.TestCase):
    def test_whole_folder_and_per_file_slicing(self):
        rng = np.random.default_rng(12345)
        with tempfile.TemporaryDirectory() as tmp:
            src_dir = os.path.join(tmp, "raw")
            os.makedirs(src_dir, exist_ok=True)
            for idx in range(3):
                payload = rng.integers(0, 256, size=2048, dtype=np.uint8).tobytes()
                with open(os.path.join(src_dir, f"file_{idx}.bin"), "wb") as f:
                    f.write(payload)

            win = _DummyWindow()
            out_whole = os.path.join(tmp, "whole.csv")
            with patch("extract_whole_folder_fragments.psg.popup"), patch("extract_whole_folder_fragments.psg.popup_error"):
                ok = extract_whole_folder_fragments(win, src_dir, fragment_size=256, total_fragments=10, dest_csv=out_whole)
            self.assertTrue(ok)
            arr_whole = np.loadtxt(out_whole, delimiter=",", dtype=np.int64)
            self.assertEqual(arr_whole.shape, (10, 256))
            self.assertTrue(np.all((arr_whole >= 0) & (arr_whole <= 255)))

            out_per_file = os.path.join(tmp, "per_file.csv")
            with patch("extract_each_file_fragments.psg.popup"), patch("extract_each_file_fragments.psg.popup_error"):
                extract_random_fragments_file(win, src_dir, fragment_size=256, num_fragments=2, dest_csv=out_per_file)
            arr_pf = np.loadtxt(out_per_file, delimiter=",", dtype=np.int64)
            self.assertEqual(arr_pf.shape, (6, 256))
            self.assertTrue(np.all((arr_pf >= 0) & (arr_pf <= 255)))


class TestStage2FeaturesAndSampleRegression(unittest.TestCase):
    def test_any_length_feature_extraction_no_nan(self):
        rng = np.random.default_rng(2026)
        for frag_len in (256, 512, 4096):
            enc = rng.integers(0, 256, size=frag_len, dtype=np.uint8)
            unenc = np.tile(np.arange(32, dtype=np.uint8), frag_len // 32)
            for vec in (enc, unenc):
                bits = np.unpackbits(vec)
                # Core NIST checks
                for fn in (
                    monobit_test,
                    frequency_within_block_test,
                    runs_test,
                    longest_run_ones_in_a_block_test,
                    binary_matrix_rank_test,
                    dft_test,
                    non_overlapping_template_matching_test,
                    overlapping_template_matching_test,
                    maurers_universal_test,
                    linear_complexity_test,
                    serial_test,
                    approximate_entropy_test,
                    cumulative_sums_test,
                    random_excursion_test,
                    random_excursion_variant_test,
                ):
                    res = fn(bits)
                    self.assertIsNotNone(res)

                # HEDGE, BFLCp, SOTA, AFRL
                c2 = chi2_uniform(vec, Length=frag_len)
                self.assertEqual(len(c2), 2)
                self.assertTrue(np.all(np.isfinite(c2)))
                self.assertIn(chi_test_absolute(c2[0], length=frag_len), (0, 1))
                self.assertIn(chi_test_confidence(c2[1]), (0, 1))

                bfd = byte_frequency_distribution(vec)
                self.assertEqual(bfd.shape, (256,))
                self.assertAlmostEqual(float(np.sum(bfd)), 1.0, places=5)

                dft = discrete_fourier_transform(vec)
                self.assertEqual(dft.shape, (10,))
                self.assertTrue(np.all(np.isfinite(dft)))

                lz = lempel_ziv_complexity(vec)
                self.assertTrue(np.isfinite(lz) and lz > 0.0)
                mc = monte_carlo_pi_approximation(vec)
                self.assertTrue(np.isfinite(mc))

                for vals in (
                    byte_statistical_moments(vec),
                    ks_uniformity_test(vec),
                    [wasserstein_uniform_distance(vec)],
                    [index_of_coincidence(vec)],
                    generalized_entropies(vec),
                    serial_and_lag_autocorrelation(vec),
                    markov_bigram_features(vec),
                    byte_difference_features(vec),
                    recompression_ratios(vec),
                    [haar_wavelet_detail_ratio(vec)],
                    [spectral_flatness_measure(vec)],
                    structural_run_and_ascii(vec),
                ):
                    self.assertTrue(np.all(np.isfinite(vals)))

                lnv_pass, lnv_stat, _ = local_nibble_variance(vec)
                self.assertTrue(np.isfinite(lnv_stat))
                self.assertIn(int(lnv_pass), (0, 1))

            # Structured repeating fragment must be flagged unencrypted (0)
            self.assertEqual(AFRL_test(unenc), 0)
            self.assertEqual(HEDGE_test(unenc), 0)

    def test_sample_dataset_matches_published_outputs(self):
        sample_csv = os.path.join(REPO_ROOT, "sample_dataset", "sample_dataset.csv")
        ref_afrl_csv = os.path.join(REPO_ROOT, "output_sample", "AFRL_detection", "AFRL_detection.csv")
        ref_hedge_csv = os.path.join(REPO_ROOT, "output_sample", "HEDGE_detection", "HEDGE_detection.csv")
        self.assertTrue(os.path.isfile(sample_csv))

        data = np.loadtxt(sample_csv, delimiter=",", dtype=np.uint8)
        self.assertEqual(data.shape, (20, 4096))

        afrl_preds = [int(AFRL_test(row)) for row in data]
        hedge_preds = [int(HEDGE_test(row)) for row in data]

        # Rows 0..9 = Unencrypted (expect 8/10 = 0), Rows 10..19 = Encrypted (expect 10/10 = 1)
        self.assertEqual(sum(p == 0 for p in afrl_preds[:10]), 8)
        self.assertEqual(sum(p == 1 for p in afrl_preds[10:]), 10)
        self.assertEqual(sum(p == 0 for p in hedge_preds[:10]), 8)
        self.assertEqual(sum(p == 1 for p in hedge_preds[10:]), 10)

        with open(ref_afrl_csv, "r", encoding="utf-8") as f:
            ref_afrl = [int(line.strip()) for line in f if line.strip().isdigit()]
        with open(ref_hedge_csv, "r", encoding="utf-8") as f:
            ref_hedge = [int(line.strip()) for line in f if line.strip().isdigit()]
        self.assertEqual(afrl_preds, ref_afrl)
        self.assertEqual(hedge_preds, ref_hedge)


class TestStage3TrainableDetectorsRoundTrip(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(777)
        self.frag_len = 512
        self.n_per_class = 24
        # Easy synthetic separation: uniform bytes vs biased structured ASCII/low-byte payloads
        self.unenc_tr = rng.integers(32, 127, size=(self.n_per_class, self.frag_len), dtype=np.uint8)
        self.enc_tr = rng.integers(0, 256, size=(self.n_per_class, self.frag_len), dtype=np.uint8)
        self.unenc_te = rng.integers(32, 127, size=(8, self.frag_len), dtype=np.uint8)
        self.enc_te = rng.integers(0, 256, size=(8, self.frag_len), dtype=np.uint8)
        self.test_block = np.vstack([self.unenc_te, self.enc_te])
        self.test_labels = np.array([0] * 8 + [1] * 8, dtype=np.int64)

    def test_encod_train_save_load_predict(self):
        with tempfile.TemporaryDirectory() as tmp:
            u_tr, e_tr = os.path.join(tmp, "u_tr.csv"), os.path.join(tmp, "e_tr.csv")
            u_te, e_te = os.path.join(tmp, "u_te.csv"), os.path.join(tmp, "e_te.csv")
            mdl_path = os.path.join(tmp, "encod_test.npz")
            _write_uint8_csv(u_tr, self.unenc_tr)
            _write_uint8_csv(e_tr, self.enc_tr)
            _write_uint8_csv(u_te, self.unenc_te)
            _write_uint8_csv(e_te, self.enc_te)

            res = train_encod_model(
                unenc_train_csv=u_tr,
                enc_train_csv=e_tr,
                save_model_path=mdl_path,
                arch_name="BinaryAll_SELU",
                scaler_mode="minmax",
                epochs=4,
                batch_size=8,
                learning_rate=1e-3,
                val_ratio=0.2,
                early_stop_patience=4,
                test_unenc_csv=u_te,
                test_enc_csv=e_te,
                seed=42,
            )
            self.assertFalse(res["cancelled"])
            self.assertTrue(os.path.isfile(mdl_path))
            summary = inspect_encod_model_summary(mdl_path)
            self.assertIn("BinaryAll_SELU", summary)

            loaded = load_encod_model(mdl_path)
            preds = predict_encod_batch(self.test_block, loaded)
            self.assertEqual(preds.shape, (16,))
            self.assertTrue(set(np.unique(preds)).issubset({0, 1}))
            acc = float(np.mean(preds == self.test_labels))
            self.assertGreaterEqual(acc, 0.90)

    def test_bflcp_train_save_load_predict(self):
        with tempfile.TemporaryDirectory() as tmp:
            u_tr, e_tr = os.path.join(tmp, "u_tr.csv"), os.path.join(tmp, "e_tr.csv")
            u_te, e_te = os.path.join(tmp, "u_te.csv"), os.path.join(tmp, "e_te.csv")
            mdl_path = os.path.join(tmp, "bflcp_test.npz")
            _write_uint8_csv(u_tr, self.unenc_tr)
            _write_uint8_csv(e_tr, self.enc_tr)
            _write_uint8_csv(u_te, self.unenc_te)
            _write_uint8_csv(e_te, self.enc_te)

            res = train_bflcp_model(
                unenc_train_csv=u_tr,
                enc_train_csv=e_tr,
                save_model_path=mdl_path,
                variant="BFLCp_270",
                scaler_mode="standard",
                epochs=4,
                batch_size=8,
                learning_rate=1e-3,
                val_ratio=0.2,
                early_stop_patience=4,
                test_unenc_csv=u_te,
                test_enc_csv=e_te,
                cpu_percent=10,
                seed=42,
            )
            self.assertFalse(res["cancelled"])
            self.assertTrue(os.path.isfile(mdl_path))
            summary = inspect_bflcp_model_summary(mdl_path)
            self.assertIn("BFLCp_270", summary)

            loaded = load_bflcp_model(mdl_path)
            preds = predict_bflcp_batch(self.test_block, loaded, pool=None)
            self.assertEqual(preds.shape, (16,))
            self.assertTrue(set(np.unique(preds)).issubset({0, 1}))
            acc = float(np.mean(preds == self.test_labels))
            self.assertGreaterEqual(acc, 0.90)

    def test_triumvir_train_save_load_predict(self):
        with tempfile.TemporaryDirectory() as tmp:
            u_tr, e_tr = os.path.join(tmp, "u_tr.csv"), os.path.join(tmp, "e_tr.csv")
            u_te, e_te = os.path.join(tmp, "u_te.csv"), os.path.join(tmp, "e_te.csv")
            mdl_path = os.path.join(tmp, "triumvir_test.npz")
            _write_uint8_csv(u_tr, self.unenc_tr)
            _write_uint8_csv(e_tr, self.enc_tr)
            _write_uint8_csv(u_te, self.unenc_te)
            _write_uint8_csv(e_te, self.enc_te)

            res = train_triumvir_model(
                unenc_train_csv=u_tr,
                enc_train_csv=e_tr,
                save_model_path=mdl_path,
                variant="Triumvir_Fast",
                scaler_mode="standard",
                epochs=3,
                batch_size=8,
                learning_rate=1e-3,
                val_ratio=0.2,
                early_stop_patience=3,
                test_unenc_csv=u_te,
                test_enc_csv=e_te,
                cpu_percent=10,
                seed=42,
            )
            self.assertFalse(res["cancelled"])
            self.assertTrue(os.path.isfile(mdl_path))
            summary = inspect_triumvir_model_summary(mdl_path)
            self.assertIn("Triumvir_Fast", summary)

            loaded = load_triumvir_model(mdl_path)
            self.assertEqual(len(loaded["temperatures"]), 3)
            for t_val in loaded["temperatures"]:
                self.assertTrue(0.1 <= float(t_val) <= 10.0)
            preds = predict_triumvir_batch(self.test_block, loaded, pool=None)
            self.assertEqual(preds.shape, (16,))
            self.assertTrue(set(np.unique(preds)).issubset({0, 1}))
            acc = float(np.mean(preds == self.test_labels))
            self.assertGreaterEqual(acc, 0.90)


class TestStage4CSVCombiner(unittest.TestCase):
    def test_row_and_column_concatenation(self):
        with tempfile.TemporaryDirectory() as tmp:
            f1 = os.path.join(tmp, "a.csv")
            f2 = os.path.join(tmp, "b.csv")
            f_col = os.path.join(tmp, "c.csv")
            out_rows = os.path.join(tmp, "merged_rows.csv")
            out_cols = os.path.join(tmp, "merged_cols.csv")

            with open(f1, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows([["c1", "c2"], [1, 2], [3, 4]])
            with open(f2, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows([["c1", "c2"], [5, 6], [7, 8]])
            with open(f_col, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows([["c3"], [9], [10]])

            win = _DummyWindow()
            with patch("combine.psg.popup"), patch("combine.psg.popup_error"):
                ok_r = concatenate_rows_gui(win, [f1, f2], out_rows, chunk_size=2, header=True)
                ok_c = concatenate_columns_gui(win, [f1, f_col], out_cols, chunk_size=2, header=True)

            self.assertTrue(ok_r)
            self.assertTrue(ok_c)

            with open(out_rows, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows, [["c1", "c2"], ["1", "2"], ["3", "4"], ["5", "6"], ["7", "8"]])

            with open(out_cols, "r", encoding="utf-8") as f:
                cols = list(csv.reader(f))
            self.assertEqual(cols, [["c1", "c2", "c3"], ["1", "2", "9"], ["3", "4", "10"]])


if __name__ == "__main__":
    unittest.main()
