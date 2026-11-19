import math
import os
import time
import zlib
import multiprocessing as mp
import numpy as np
from scipy.fft import rfft
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, skew, kurtosis

from nn_utils import (
    iter_uint8_csv_chunks,
    fit_scaler,
    apply_scaler,
    stratified_split,
    compute_binary_metrics,
    softmax,
    fmt_hms,
    LinearLayer,
    BatchNorm1dLayer,
    DropoutLayer,
    ActivationLayer,
    SequentialMLP,
)
from BLFCP_fspecials import lempel_ziv_complexity
from SOTA_fspecials import (
    ks_uniformity_test,
    wasserstein_uniform_distance,
    markov_bigram_features,
    structural_run_and_ascii,
)
from AFRL_fspecials import local_nibble_variance

TRIUMVIR_VARIANTS = {
    "Triumvir_Full": {
        "include_ks_wass": True,
        "stat_dim": 269,
        "seq_dim": 14,
        "spat_dim": 22,
        "label": "Triumvir-Style Full (305 dims: Stat 269 + Seq 14 + Spatial/Spectral 22 + Calibrated Gate)",
    },
    "Triumvir_Fast": {
        "include_ks_wass": False,
        "stat_dim": 266,
        "seq_dim": 14,
        "spat_dim": 22,
        "label": "Triumvir-Style Fast (302 dims: omits KS/Wasserstein for faster large-scale runs)",
    },
}


# =====================================================================
# ROW-LEVEL HELPER FOR MULTIPROCESSING (TOP-LEVEL PICKLABLE)
# =====================================================================
def _triumvir_row_worker(args):
    row_u8, include_ks_wass = args
    n = row_u8.size

    # Sequential scalars: Markov(4), LZ76(1), zlib ratio(1), norm max run(1)
    mk = markov_bigram_features(row_u8)
    lz = lempel_ziv_complexity(row_u8)
    z_ratio = float(len(zlib.compress(row_u8.tobytes(), level=6)) / float(max(1, n)))
    max_run_norm, _ = structural_run_and_ascii(row_u8)

    # Spatial LNV + mean window chi2
    _, lnv_val, _ = local_nibble_variance(row_u8)
    if n >= 64:
        n_win = (n - 64) // 32 + 1
        stride = row_u8.strides[0]
        wins = np.lib.stride_tricks.as_strided(row_u8, shape=(n_win, 64), strides=(stride * 32, stride))
        nibs = np.concatenate((wins >> 4, wins & 0x0F), axis=1)
        row_off = (np.arange(n_win, dtype=np.int32) * 16)[:, None]
        w_counts = np.bincount((nibs.astype(np.int32) + row_off).ravel(), minlength=n_win * 16).reshape(n_win, 16)
        w_chi2_mean = float(np.mean(np.sum(((w_counts - 8.0) ** 2) / 8.0, axis=1)))
    else:
        w_chi2_mean = 0.0

    if include_ks_wass:
        ks_s, ks_p = ks_uniformity_test(row_u8)
        wass = wasserstein_uniform_distance(row_u8)
        return (ks_s, ks_p, wass, mk[0], mk[1], mk[2], mk[3], lz, z_ratio, max_run_norm, lnv_val, w_chi2_mean)
    return (mk[0], mk[1], mk[2], mk[3], lz, z_ratio, max_run_norm, lnv_val, w_chi2_mean)


# =====================================================================
# MULTI-MODAL FEATURE EXTRACTION (STATISTICAL, SEQUENTIAL, SPATIAL)
# =====================================================================
def extract_triumvir_modalities_batch(
    byte_block: np.ndarray,
    variant: str = "Triumvir_Full",
    pool=None,
):
    """
    Extracts 3 complementary modalities for any fragment length L:
      - X_stat: Statistical & distributional geometry
      - X_seq:  Sequential, Markov, autocorrelation & compressibility
      - X_spat: Spatial 2D grid, multi-scale block heterogeneity, LNV & spectral
    """
    u8 = np.asarray(byte_block, dtype=np.uint8)
    if u8.ndim == 1:
        u8 = u8[None, :]
    n_rows, frag_len = u8.shape
    spec = TRIUMVIR_VARIANTS[variant]
    include_kw = spec["include_ks_wass"]

    if n_rows == 0 or frag_len == 0:
        return (
            np.zeros((n_rows, spec["stat_dim"]), dtype=np.float32),
            np.zeros((n_rows, spec["seq_dim"]), dtype=np.float32),
            np.zeros((n_rows, spec["spat_dim"]), dtype=np.float32),
        )

    f64 = u8.astype(np.float64)

    # --- Vectorized BFD + Chi2 + Moments + Entropies + IoC ---
    idx_dtype = np.int32 if (n_rows * 256) < 2_000_000_000 else np.int64
    row_offsets = (np.arange(n_rows, dtype=idx_dtype) * 256)[:, None]
    counts = np.bincount((u8.astype(idx_dtype) + row_offsets).ravel(), minlength=n_rows * 256).reshape(n_rows, 256)
    counts_f64 = counts.astype(np.float64)
    bfd = (counts_f64 / float(frag_len)).astype(np.float32)

    expected = float(frag_len) / 256.0
    chi2_stats = np.sum(((counts_f64 - expected) ** 2) / expected, axis=1)
    chi2_pvals = chi2.sf(chi2_stats, df=255)

    means = np.mean(f64, axis=1)
    vars_ = np.var(f64, axis=1)
    safe_mask = vars_ > 1e-12
    skews = np.zeros(n_rows, dtype=np.float64)
    kurts = np.zeros(n_rows, dtype=np.float64)
    if np.any(safe_mask):
        skews[safe_mask] = skew(f64[safe_mask], axis=1)
        kurts[safe_mask] = kurtosis(f64[safe_mask], axis=1)

    ioc = np.sum(counts_f64 * (counts_f64 - 1.0), axis=1) / float(max(1, frag_len * (frag_len - 1))) if frag_len >= 2 else np.zeros(n_rows)
    probs = counts_f64 / float(frag_len)
    sum_p2 = np.sum(probs * probs, axis=1)
    h_coll = np.where(sum_p2 > 0.0, -np.log2(np.maximum(sum_p2, 1e-12)), 0.0)
    h_min = -np.log2(np.maximum(np.max(probs, axis=1), 1e-12))
    h_tsal = 1.0 - sum_p2

    # --- Vectorized Sequential (Autocorr lags 1,2,4,8 + ByteDiff + ASCII) ---
    xc = f64 - means[:, None]
    denom = np.sum(xc * xc, axis=1)
    valid_ac = (denom > 1e-12) & (frag_len >= 10)
    ac1 = np.zeros(n_rows, dtype=np.float64)
    ac2 = np.zeros(n_rows, dtype=np.float64)
    ac4 = np.zeros(n_rows, dtype=np.float64)
    ac8 = np.zeros(n_rows, dtype=np.float64)
    if np.any(valid_ac):
        d_ok = denom[valid_ac]
        xc_ok = xc[valid_ac]
        ac1[valid_ac] = (np.sum(xc_ok[:, :-1] * xc_ok[:, 1:], axis=1) + xc_ok[:, -1] * xc_ok[:, 0]) / d_ok
        ac2[valid_ac] = np.sum(xc_ok[:, :-2] * xc_ok[:, 2:], axis=1) / d_ok
        ac4[valid_ac] = np.sum(xc_ok[:, :-4] * xc_ok[:, 4:], axis=1) / d_ok
        ac8[valid_ac] = np.sum(xc_ok[:, :-8] * xc_ok[:, 8:], axis=1) / d_ok

    if frag_len >= 2:
        mod_diff = np.diff(u8, axis=1)
        d_off = (np.arange(n_rows, dtype=idx_dtype) * 256)[:, None]
        d_counts = np.bincount((mod_diff.astype(idx_dtype) + d_off).ravel(), minlength=n_rows * 256).reshape(n_rows, 256)
        d_probs = d_counts.astype(np.float64) / float(mod_diff.shape[1])
        h_diff = -np.sum(np.where(d_probs > 0.0, d_probs * np.log2(np.maximum(d_probs, 1e-12)), 0.0), axis=1)
        abs_diff_mean = np.mean(np.abs(np.diff(f64, axis=1)), axis=1)
    else:
        h_diff = np.zeros(n_rows, dtype=np.float64)
        abs_diff_mean = np.zeros(n_rows, dtype=np.float64)

    ascii_ratio = np.mean((u8 >= 32) & (u8 <= 126), axis=1)

    # --- Vectorized Spatial / Spectral (DFT + Haar + SFM + 2D Grid + 8-Block) ---
    dft_feats = np.zeros((n_rows, 10), dtype=np.float32)
    if frag_len >= 2:
        mags = np.abs(rfft(f64, axis=1))
        actual_k = min(5, mags.shape[1])
        if actual_k > 0:
            top_idx = np.argsort(mags, axis=1)[:, -actual_k:]
            top_mags = np.take_along_axis(mags, top_idx, axis=1)
            tot_m = np.sum(mags, axis=1, keepdims=True)
            safe_tot = np.where(tot_m > 0.0, tot_m, 1.0)
            dft_feats[:, :actual_k] = np.where(tot_m > 0.0, top_mags / safe_tot, top_mags).astype(np.float32)
            dft_feats[:, 5:5 + actual_k] = top_idx.astype(np.float32) / float(max(1, frag_len // 2))

    even_len = frag_len & ~1
    if even_len >= 2:
        hc = f64[:, :even_len] - np.mean(f64[:, :even_len], axis=1, keepdims=True)
        ev_c, od_c = hc[:, 0::2], hc[:, 1::2]
        ea = np.sum((ev_c + od_c) ** 2, axis=1)
        ed = np.sum((ev_c - od_c) ** 2, axis=1)
        et = ea + ed
        haar_ratio = np.where(et > 1e-12, ed / np.maximum(et, 1e-12), 0.0)
    else:
        haar_ratio = np.zeros(n_rows, dtype=np.float64)

    if frag_len >= 4:
        ps = (np.abs(rfft(xc, axis=1))[:, 1:]) ** 2
        am = np.mean(ps, axis=1) if ps.shape[1] > 0 else np.zeros(n_rows)
        ps_safe = np.maximum(ps, 1e-12)
        sfm = np.where(am > 1e-12, np.exp(np.mean(np.log(ps_safe), axis=1)) / np.maximum(am, 1e-12), 0.0)
    else:
        sfm = np.zeros(n_rows, dtype=np.float64)

    # 2D byte matrix spatial features (grid width = 32 bytes)
    grid_w = 32 if frag_len >= 64 else max(2, frag_len // 2)
    grid_h = frag_len // grid_w
    if grid_h >= 2 and grid_w >= 2:
        grid = f64[:, :grid_h * grid_w].reshape(n_rows, grid_h, grid_w)
        h_jump = np.mean(np.abs(grid[:, :, 1:] - grid[:, :, :-1]), axis=(1, 2))
        v_jump = np.mean(np.abs(grid[:, 1:, :] - grid[:, :-1, :]), axis=(1, 2))
        hv_ratio = (h_jump + 1e-6) / (v_jump + 1e-6)

        g0 = grid[:, :-1, :].reshape(n_rows, -1)
        g1 = grid[:, 1:, :].reshape(n_rows, -1)
        g0c = g0 - np.mean(g0, axis=1, keepdims=True)
        g1c = g1 - np.mean(g1, axis=1, keepdims=True)
        g_den = np.sqrt(np.sum(g0c * g0c, axis=1) * np.sum(g1c * g1c, axis=1))
        v_corr = np.where(g_den > 1e-12, np.sum(g0c * g1c, axis=1) / np.maximum(g_den, 1e-12), 0.0)
    else:
        h_jump = np.zeros(n_rows)
        v_jump = np.zeros(n_rows)
        hv_ratio = np.zeros(n_rows)
        v_corr = np.zeros(n_rows)

    # 8-block spatial heterogeneity
    blk_len = frag_len // 8
    if blk_len >= 4:
        b8 = f64[:, :8 * blk_len].reshape(n_rows, 8, blk_len)
        b8_u8 = u8[:, :8 * blk_len].reshape(n_rows, 8, blk_len)
        b8_means = np.mean(b8, axis=2)
        b8_vars = np.var(b8, axis=2)
        b8_ascii = np.mean((b8_u8 >= 32) & (b8_u8 <= 126), axis=2)
        blk_mean_std = np.std(b8_means, axis=1)
        blk_var_std = np.std(b8_vars, axis=1)
        blk_ascii_std = np.std(b8_ascii, axis=1)
        blk_mean_gap = np.max(b8_means, axis=1) - np.min(b8_means, axis=1)
    else:
        blk_mean_std = np.zeros(n_rows)
        blk_var_std = np.zeros(n_rows)
        blk_ascii_std = np.zeros(n_rows)
        blk_mean_gap = np.zeros(n_rows)

    # --- Per-row sequential/scipy helpers (parallelized when pool is active) ---
    tasks = [(u8[i], include_kw) for i in range(n_rows)]
    if pool is not None and n_rows >= 32:
        row_scalars = np.asarray(pool.map(_triumvir_row_worker, tasks, chunksize=max(1, n_rows // 16)), dtype=np.float32)
    else:
        row_scalars = np.asarray([_triumvir_row_worker(t) for t in tasks], dtype=np.float32)

    if include_kw:
        ks_s = row_scalars[:, 0:1]
        ks_p = row_scalars[:, 1:2]
        wass = row_scalars[:, 2:3]
        seq_tail = row_scalars[:, 3:10]   # mk(4), lz(1), zlib(1), max_run(1)
        spat_lnv = row_scalars[:, 10:12]  # lnv_val, w_chi2_mean
    else:
        seq_tail = row_scalars[:, 0:7]
        spat_lnv = row_scalars[:, 7:9]

    stat_cols = [
        bfd,
        chi2_stats[:, None].astype(np.float32),
        chi2_pvals[:, None].astype(np.float32),
        means[:, None].astype(np.float32),
        vars_[:, None].astype(np.float32),
        skews[:, None].astype(np.float32),
        kurts[:, None].astype(np.float32),
    ]
    if include_kw:
        stat_cols.extend([ks_s, ks_p, wass])
    stat_cols.extend([
        ioc[:, None].astype(np.float32),
        h_coll[:, None].astype(np.float32),
        h_min[:, None].astype(np.float32),
        h_tsal[:, None].astype(np.float32),
    ])
    X_stat = np.hstack(stat_cols).astype(np.float32, copy=False)

    X_seq = np.hstack([
        ac1[:, None].astype(np.float32),
        ac2[:, None].astype(np.float32),
        ac4[:, None].astype(np.float32),
        ac8[:, None].astype(np.float32),
        h_diff[:, None].astype(np.float32),
        abs_diff_mean[:, None].astype(np.float32),
        ascii_ratio[:, None].astype(np.float32),
        seq_tail,
    ]).astype(np.float32, copy=False)

    X_spat = np.hstack([
        dft_feats,
        haar_ratio[:, None].astype(np.float32),
        sfm[:, None].astype(np.float32),
        spat_lnv,
        h_jump[:, None].astype(np.float32),
        v_jump[:, None].astype(np.float32),
        hv_ratio[:, None].astype(np.float32),
        v_corr[:, None].astype(np.float32),
        blk_mean_std[:, None].astype(np.float32),
        blk_var_std[:, None].astype(np.float32),
        blk_ascii_std[:, None].astype(np.float32),
        blk_mean_gap[:, None].astype(np.float32),
    ]).astype(np.float32, copy=False)

    return X_stat, X_seq, X_spat


def load_triumvir_modalities_from_csv(
    file_path: str,
    variant: str = "Triumvir_Full",
    pool=None,
    chunk_size: int = 800,
    cancel_event=None,
    progress_cb=None,
):
    stat_chunks, seq_chunks, spat_chunks = [], [], []
    rows_read = 0
    detected_len = 0

    for u8_block, total_rows in iter_uint8_csv_chunks(file_path, chunk_size=chunk_size):
        if cancel_event is not None and cancel_event.is_set():
            return None, None, None, detected_len
        detected_len = u8_block.shape[1]
        xs, xq, xp = extract_triumvir_modalities_batch(u8_block, variant=variant, pool=pool)
        stat_chunks.append(xs)
        seq_chunks.append(xq)
        spat_chunks.append(xp)
        rows_read += u8_block.shape[0]
        if progress_cb is not None:
            progress_cb(rows_read, total_rows)

    if not stat_chunks:
        raise ValueError(f"No data rows found in CSV file: {file_path}")
    return (
        np.vstack(stat_chunks).astype(np.float32, copy=False),
        np.vstack(seq_chunks).astype(np.float32, copy=False),
        np.vstack(spat_chunks).astype(np.float32, copy=False),
        int(detected_len),
    )


# =====================================================================
# THREE EXPERTS + TEMPERATURE CALIBRATION + LEARNED GATE NETWORK
# =====================================================================
def build_triumvir_experts_and_gate(stat_dim: int, seq_dim: int, spat_dim: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    stat_mlp = SequentialMLP([
        LinearLayer(stat_dim, 256, "lecun_normal", name="stat.1", rng=rng),
        ActivationLayer("selu"),
        LinearLayer(256, 128, "lecun_normal", name="stat.2", rng=rng),
        ActivationLayer("selu"),
        LinearLayer(128, 64, "lecun_normal", name="stat.3", rng=rng),
        ActivationLayer("selu"),
        LinearLayer(64, 2, "lecun_normal", name="stat.4", rng=rng),
    ])
    seq_mlp = SequentialMLP([
        LinearLayer(seq_dim, 128, "he_normal", name="seq.1", rng=rng),
        ActivationLayer("relu"),
        BatchNorm1dLayer(128, name="seq.bn1"),
        DropoutLayer(0.2, rng=rng),
        LinearLayer(128, 64, "he_normal", name="seq.2", rng=rng),
        ActivationLayer("relu"),
        BatchNorm1dLayer(64, name="seq.bn2"),
        LinearLayer(64, 32, "he_normal", name="seq.3", rng=rng),
        ActivationLayer("relu"),
        LinearLayer(32, 2, "glorot_uniform", name="seq.4", rng=rng),
    ])
    spat_mlp = SequentialMLP([
        LinearLayer(spat_dim, 128, "he_normal", name="spat.1", rng=rng),
        ActivationLayer("relu"),
        BatchNorm1dLayer(128, name="spat.bn1"),
        DropoutLayer(0.2, rng=rng),
        LinearLayer(128, 64, "he_normal", name="spat.2", rng=rng),
        ActivationLayer("relu"),
        BatchNorm1dLayer(64, name="spat.bn2"),
        LinearLayer(64, 32, "he_normal", name="spat.3", rng=rng),
        ActivationLayer("relu"),
        LinearLayer(32, 2, "glorot_uniform", name="spat.4", rng=rng),
    ])
    # Gate takes 9 inputs: calibrated logits (3 * 2) + confidence margins (3) -> 3 softmax expert weights
    gate_mlp = SequentialMLP([
        LinearLayer(9, 32, "he_normal", name="gate.1", rng=rng),
        ActivationLayer("relu"),
        LinearLayer(32, 16, "he_normal", name="gate.2", rng=rng),
        ActivationLayer("relu"),
        LinearLayer(16, 3, "glorot_uniform", name="gate.3", rng=rng),
    ])
    return stat_mlp, seq_mlp, spat_mlp, gate_mlp


def _fit_temperature_scalar(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Finds optimal temperature T in [0.1, 10.0] minimizing NLL on calibration logits."""
    y_i = y_true.astype(np.int64)
    idx_ar = np.arange(len(y_i))

    def _nll(temp: float) -> float:
        p = softmax(logits / float(temp))
        return -float(np.mean(np.log(np.clip(p[idx_ar, y_i], 1e-12, 1.0))))

    res = minimize_scalar(_nll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x) if res.success else 1.0


def _build_gate_inputs_and_probs(logits_list, temps):
    cal_logits = [lg / float(t) for lg, t in zip(logits_list, temps)]
    cal_probs = [softmax(clg) for clg in cal_logits]
    margins = [np.abs(p[:, 1] - p[:, 0])[:, None] for p in cal_probs]
    gate_x = np.hstack(cal_logits + margins).astype(np.float32, copy=False)
    stacked_probs = np.stack(cal_probs, axis=1).astype(np.float32, copy=False)  # (N, 3, 2)
    return gate_x, stacked_probs


def _train_gate_network(
    gate_mlp: SequentialMLP,
    gate_x_cal: np.ndarray,
    stacked_p_cal: np.ndarray,
    y_cal: np.ndarray,
    gate_x_val: np.ndarray,
    stacked_p_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    seed: int = 123,
):
    param_refs = [p for ly in gate_mlp.layers if hasattr(ly, "params_and_grads") for p, _ in ly.params_and_grads()]
    m_states = [np.zeros_like(p) for p in param_refs]
    v_states = [np.zeros_like(p) for p in param_refs]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    adam_step = 0

    best_loss = float("inf")
    best_acc = -1.0
    best_state = gate_mlp._clone_state()
    rng = np.random.default_rng(seed)
    n_cal = gate_x_cal.shape[0]

    for _ in range(epochs):
        perm = rng.permutation(n_cal)
        for start in range(0, n_cal, batch_size):
            idx = perm[start:start + batch_size]
            bsz = idx.shape[0]
            gx = gate_x_cal[idx]
            sp = stacked_p_cal[idx]
            yb = y_cal[idx]

            gate_logits = gate_mlp.forward(gx, training=True)
            w = softmax(gate_logits)  # (B, 3)
            p_ens = np.sum(w[:, :, None] * sp, axis=1)  # (B, 2)
            p_y = np.clip(p_ens[np.arange(bsz), yb], 1e-12, 1.0)
            p_exp_y = sp[np.arange(bsz), :, yb]  # (B, 3)

            # Exact mixture cross-entropy gradient w.r.t. gate logits: (w - responsibility) / B
            resp = (w * p_exp_y) / p_y[:, None]
            ds = ((w - resp) / float(bsz)).astype(np.float32)
            gate_mlp.backward(ds)

            adam_step += 1
            lr_t = lr * math.sqrt(1.0 - (beta2 ** adam_step)) / (1.0 - (beta1 ** adam_step))
            p_idx = 0
            for ly in gate_mlp.layers:
                if hasattr(ly, "params_and_grads"):
                    for p, g in ly.params_and_grads():
                        g_eff = g + (weight_decay * p if p.ndim == 2 else 0.0)
                        m_states[p_idx] = beta1 * m_states[p_idx] + (1.0 - beta1) * g_eff
                        v_states[p_idx] = beta2 * v_states[p_idx] + (1.0 - beta2) * (g_eff * g_eff)
                        p -= lr_t * m_states[p_idx] / (np.sqrt(v_states[p_idx]) + eps)
                        p_idx += 1

        val_w = softmax(gate_mlp.forward(gate_x_val, training=False))
        val_p = np.sum(val_w[:, :, None] * stacked_p_val, axis=1)
        vm = compute_binary_metrics(y_val, val_p)
        if (vm["accuracy"] > best_acc) or (abs(vm["accuracy"] - best_acc) < 1e-6 and vm["loss"] < best_loss):
            best_acc = vm["accuracy"]
            best_loss = vm["loss"]
            best_state = gate_mlp._clone_state()

    gate_mlp.load_state(best_state)


def _eval_ensemble_probs(model: dict, X_stat: np.ndarray, X_seq: np.ndarray, X_spat: np.ndarray) -> np.ndarray:
    xs = apply_scaler(X_stat, model["stat_scale"], model["stat_min"])
    xq = apply_scaler(X_seq, model["seq_scale"], model["seq_min"])
    xp = apply_scaler(X_spat, model["spat_scale"], model["spat_min"])

    lg_stat = model["stat_mlp"].forward(xs, training=False)
    lg_seq = model["seq_mlp"].forward(xq, training=False)
    lg_spat = model["spat_mlp"].forward(xp, training=False)

    gate_x, stacked_p = _build_gate_inputs_and_probs([lg_stat, lg_seq, lg_spat], model["temperatures"])
    w = softmax(model["gate_mlp"].forward(gate_x, training=False))
    return np.sum(w[:, :, None] * stacked_p, axis=1).astype(np.float32)


# =====================================================================
# SAVE / LOAD / INSPECT TRAINED MODEL (.NPZ)
# =====================================================================
def save_triumvir_model(
    save_path: str,
    variant: str,
    scaler_mode: str,
    frag_len: int,
    stat_mlp: SequentialMLP,
    seq_mlp: SequentialMLP,
    spat_mlp: SequentialMLP,
    gate_mlp: SequentialMLP,
    stat_scale: np.ndarray,
    stat_min: np.ndarray,
    seq_scale: np.ndarray,
    seq_min: np.ndarray,
    spat_scale: np.ndarray,
    spat_min: np.ndarray,
    temperatures,
    expert_val_accs,
    val_metrics: dict,
    test_metrics: dict = None,
) -> str:
    parent = os.path.dirname(os.path.abspath(save_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not save_path.lower().endswith(".npz"):
        save_path = save_path + ".npz"

    val_metrics = val_metrics or {}
    payload = {
        "method": np.array("Triumvir"),
        "arch": np.array(variant),
        "scaler_mode": np.array(scaler_mode),
        "fragment_length": np.array(int(frag_len), dtype=np.int64),
        "stat_dim": np.array(TRIUMVIR_VARIANTS[variant]["stat_dim"], dtype=np.int64),
        "seq_dim": np.array(TRIUMVIR_VARIANTS[variant]["seq_dim"], dtype=np.int64),
        "spat_dim": np.array(TRIUMVIR_VARIANTS[variant]["spat_dim"], dtype=np.int64),
        "stat_scale": stat_scale.astype(np.float32),
        "stat_min": stat_min.astype(np.float32),
        "seq_scale": seq_scale.astype(np.float32),
        "seq_min": seq_min.astype(np.float32),
        "spat_scale": spat_scale.astype(np.float32),
        "spat_min": spat_min.astype(np.float32),
        "temperatures": np.asarray(temperatures, dtype=np.float32),
        "expert_val_accs": np.asarray(expert_val_accs, dtype=np.float64),
        "best_val_acc": np.array(float(val_metrics.get("accuracy", 0.0)), dtype=np.float64),
        "best_val_loss": np.array(float(val_metrics.get("loss", 0.0)), dtype=np.float64),
        "val_precision": np.array(float(val_metrics.get("precision", 0.0)), dtype=np.float64),
        "val_recall": np.array(float(val_metrics.get("recall", 0.0)), dtype=np.float64),
        "val_f1": np.array(float(val_metrics.get("f1", 0.0)), dtype=np.float64),
    }
    if test_metrics is not None:
        payload["test_acc"] = np.array(float(test_metrics.get("accuracy", 0.0)), dtype=np.float64)
        payload["test_f1"] = np.array(float(test_metrics.get("f1", 0.0)), dtype=np.float64)

    for mlp in (stat_mlp, seq_mlp, spat_mlp, gate_mlp):
        payload.update(mlp.export_state())

    np.savez(save_path, **payload)
    return save_path


def load_triumvir_model(model_path: str) -> dict:
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(f"Triumvir model file not found: {model_path}")
    if os.path.splitext(model_path)[1].lower() != ".npz":
        raise ValueError(f"Triumvir models in AFRL Studio use '.npz' format: {model_path}")

    with np.load(model_path, allow_pickle=False) as data:
        if "method" not in data or str(data["method"].item()) != "Triumvir":
            raise ValueError(f"Selected file is not a Triumvir ensemble model: {model_path}")

        variant = str(data["arch"].item())
        stat_dim = int(data["stat_dim"].item())
        seq_dim = int(data["seq_dim"].item())
        spat_dim = int(data["spat_dim"].item())

        stat_mlp, seq_mlp, spat_mlp, gate_mlp = build_triumvir_experts_and_gate(stat_dim, seq_dim, spat_dim)
        state = {k: np.asarray(data[k], dtype=np.float32) for k in data.files if k.startswith(("stat.", "seq.", "spat.", "gate."))}
        for mlp in (stat_mlp, seq_mlp, spat_mlp, gate_mlp):
            mlp.load_state(state)

        return {
            "path": model_path,
            "method": "Triumvir",
            "arch": variant,
            "scaler_mode": str(data["scaler_mode"].item()),
            "fragment_length": int(data["fragment_length"].item()),
            "stat_mlp": stat_mlp,
            "seq_mlp": seq_mlp,
            "spat_mlp": spat_mlp,
            "gate_mlp": gate_mlp,
            "stat_scale": np.asarray(data["stat_scale"], dtype=np.float32),
            "stat_min": np.asarray(data["stat_min"], dtype=np.float32),
            "seq_scale": np.asarray(data["seq_scale"], dtype=np.float32),
            "seq_min": np.asarray(data["seq_min"], dtype=np.float32),
            "spat_scale": np.asarray(data["spat_scale"], dtype=np.float32),
            "spat_min": np.asarray(data["spat_min"], dtype=np.float32),
            "temperatures": np.asarray(data["temperatures"], dtype=np.float32).tolist(),
            "best_val_acc": float(data["best_val_acc"].item()) if "best_val_acc" in data else 0.0,
        }


def inspect_triumvir_model_summary(model_path: str) -> str:
    try:
        m = load_triumvir_model(model_path)
        acc_str = f"Val Acc: {m['best_val_acc'] * 100:.2f}%" if m["best_val_acc"] > 0 else "Loaded"
        len_str = f"Train L={m['fragment_length']}B" if m["fragment_length"] > 0 else "Any Length"
        return f"Ready  │  {m['arch']}  │  {m['scaler_mode']}  │  {len_str}  │  {acc_str}"
    except Exception as e:
        return f"Model error: {e}"


# =====================================================================
# TRAINING PIPELINE (CALLED BY GUI THREAD)
# =====================================================================
def train_triumvir_model(
    unenc_train_csv: str,
    enc_train_csv: str,
    save_model_path: str,
    test_unenc_csv: str = "",
    test_enc_csv: str = "",
    variant: str = "Triumvir_Full",
    scaler_mode: str = "standard",
    epochs: int = 45,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    val_ratio: float = 0.20,
    early_stop_patience: int = 12,
    cpu_percent: int = 80,
    seed: int = 123,
    cancel_event=None,
    status_cb=None,
) -> dict:
    if epochs < 1 or batch_size < 2:
        raise ValueError("Epochs must be >= 1 and Batch Size must be >= 2.")
    if not (0.01 <= val_ratio <= 0.90):
        raise ValueError("Validation split ratio must be between 0.01 and 0.90.")
    if not save_model_path or not save_model_path.strip():
        raise ValueError("Please specify a save path for the trained Triumvir model (.npz).")
    if bool(test_unenc_csv) != bool(test_enc_csv):
        raise ValueError("Provide both Unencrypted and Encrypted Test CSVs, or leave both empty.")

    def _notify(cur, total, status_txt, time_txt):
        if status_cb is not None:
            status_cb(cur, total, status_txt, time_txt)

    t_start = time.time()
    num_cores = max(1, int(mp.cpu_count() * cpu_percent / 100))
    _notify(0, 100, f"Stage 1/5: Extracting 3 modalities from unencrypted CSV ({num_cores} cores)...", "Elapsed: 00:00:00")

    pool_ctx = mp.Pool(processes=num_cores) if num_cores > 1 else None
    try:
        xs0, xq0, xp0, len0 = load_triumvir_modalities_from_csv(
            unenc_train_csv, variant=variant, pool=pool_ctx, cancel_event=cancel_event,
            progress_cb=lambda r, tot: _notify(min(20, int(20 * r / max(1, tot))), 100, f"Stage 1/5: Unencrypted features ({r}/{tot})...", f"Elapsed: {fmt_hms(time.time() - t_start)}"),
        )
        if xs0 is None:
            return {"cancelled": True}

        xs1, xq1, xp1, len1 = load_triumvir_modalities_from_csv(
            enc_train_csv, variant=variant, pool=pool_ctx, cancel_event=cancel_event,
            progress_cb=lambda r, tot: _notify(20 + min(20, int(20 * r / max(1, tot))), 100, f"Stage 1/5: Encrypted features ({r}/{tot})...", f"Elapsed: {fmt_hms(time.time() - t_start)}"),
        )
        if xs1 is None:
            return {"cancelled": True}

        txs, txq, txp, y_test = None, None, None, None
        if test_unenc_csv and test_enc_csv and os.path.isfile(test_unenc_csv) and os.path.isfile(test_enc_csv):
            _notify(40, 100, "Stage 1/5: Extracting test CSV modalities...", f"Elapsed: {fmt_hms(time.time() - t_start)}")
            txs0, txq0, txp0, _ = load_triumvir_modalities_from_csv(test_unenc_csv, variant=variant, pool=pool_ctx, cancel_event=cancel_event)
            if txs0 is None:
                return {"cancelled": True}
            txs1, txq1, txp1, _ = load_triumvir_modalities_from_csv(test_enc_csv, variant=variant, pool=pool_ctx, cancel_event=cancel_event)
            if txs1 is None:
                return {"cancelled": True}
            txs = np.vstack([txs0, txs1]).astype(np.float32, copy=False)
            txq = np.vstack([txq0, txq1]).astype(np.float32, copy=False)
            txp = np.vstack([txp0, txp1]).astype(np.float32, copy=False)
            y_test = np.concatenate([np.zeros(len(txs0), dtype=np.int64), np.ones(len(txs1), dtype=np.int64)])
    finally:
        if pool_ctx is not None:
            pool_ctx.close()
            pool_ctx.join()

    X_stat_all = np.vstack([xs0, xs1]).astype(np.float32, copy=False)
    X_seq_all = np.vstack([xq0, xq1]).astype(np.float32, copy=False)
    X_spat_all = np.vstack([xp0, xp1]).astype(np.float32, copy=False)
    y_all = np.concatenate([np.zeros(len(xs0), dtype=np.int64), np.ones(len(xs1), dtype=np.int64)])
    n_total = len(y_all)
    if n_total < 6:
        raise ValueError("Need at least 6 total training fragments for 3-way train/cal/val splitting.")
    frag_len = len0 if len0 == len1 else len0
    del xs0, xq0, xp0, xs1, xq1, xp1

    # 3-way stratified split via index array so all 3 modalities share identical splits
    all_indices = np.arange(n_total, dtype=np.int64)[:, None]
    dev_idx_2d, val_idx_2d, y_dev, y_val = stratified_split(all_indices, y_all, val_ratio=val_ratio, seed=seed)
    dev_idx, val_idx = dev_idx_2d.ravel(), val_idx_2d.ravel()

    if len(dev_idx) >= 20:
        tr_idx_2d, cal_idx_2d, y_tr, y_cal = stratified_split(dev_idx[:, None], y_dev, val_ratio=0.18, seed=seed + 1)
        tr_idx, cal_idx = tr_idx_2d.ravel(), cal_idx_2d.ravel()
    else:
        tr_idx, cal_idx, y_tr, y_cal = dev_idx, dev_idx, y_dev, y_dev

    # Fit per-modality scalers strictly on X_tr
    stat_scale, stat_min = fit_scaler(X_stat_all[tr_idx], mode=scaler_mode)
    seq_scale, seq_min = fit_scaler(X_seq_all[tr_idx], mode=scaler_mode)
    spat_scale, spat_min = fit_scaler(X_spat_all[tr_idx], mode=scaler_mode)

    stat_tr, stat_cal, stat_val = apply_scaler(X_stat_all[tr_idx], stat_scale, stat_min), apply_scaler(X_stat_all[cal_idx], stat_scale, stat_min), apply_scaler(X_stat_all[val_idx], stat_scale, stat_min)
    seq_tr, seq_cal, seq_val = apply_scaler(X_seq_all[tr_idx], seq_scale, seq_min), apply_scaler(X_seq_all[cal_idx], seq_scale, seq_min), apply_scaler(X_seq_all[val_idx], seq_scale, seq_min)
    spat_tr, spat_cal, spat_val = apply_scaler(X_spat_all[tr_idx], spat_scale, spat_min), apply_scaler(X_spat_all[cal_idx], spat_scale, spat_min), apply_scaler(X_spat_all[val_idx], spat_scale, spat_min)
    del X_stat_all, X_seq_all, X_spat_all

    stat_mlp, seq_mlp, spat_mlp, gate_mlp = build_triumvir_experts_and_gate(
        stat_tr.shape[1], seq_tr.shape[1], spat_tr.shape[1], seed=seed
    )

    expert_specs = [
        ("Stage 2/5: Statistical Expert", stat_mlp, stat_tr, stat_cal, 0),
        ("Stage 3/5: Sequential Expert", seq_mlp, seq_tr, seq_cal, 1),
        ("Stage 4/5: Spatial/Spectral Expert", spat_mlp, spat_tr, spat_cal, 2),
    ]
    for stage_lbl, mlp, x_e_tr, x_e_cal, stage_i in expert_specs:
        res_e = mlp.train_loop(
            X_train=x_e_tr, y_train=y_tr, X_val=x_e_cal, y_val=y_cal,
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            early_stop_patience=early_stop_patience, seed=seed + stage_i, cancel_event=cancel_event,
            progress_cb=lambda ep, ep_tot, st, ts, lbl=stage_lbl, si=stage_i: _notify(si * epochs + ep, 3 * epochs + 10, f"{lbl} │ {st}", ts),
        )
        if res_e.get("cancelled"):
            return {"cancelled": True}

    if cancel_event is not None and cancel_event.is_set():
        return {"cancelled": True}

    # Stage 5: Per-Expert Temperature Calibration + Learned Uncertainty Gate on held-out calibration split
    _notify(3 * epochs + 2, 3 * epochs + 10, "Stage 5/5: Calibrating temperatures & training uncertainty gate...", f"Elapsed: {fmt_hms(time.time() - t_start)}")
    lg_cal = [
        stat_mlp.forward(stat_cal, training=False),
        seq_mlp.forward(seq_cal, training=False),
        spat_mlp.forward(spat_cal, training=False),
    ]
    temperatures = [_fit_temperature_scalar(lg, y_cal) for lg in lg_cal]

    lg_val = [
        stat_mlp.forward(stat_val, training=False),
        seq_mlp.forward(seq_val, training=False),
        spat_mlp.forward(spat_val, training=False),
    ]
    expert_val_accs = [float(np.mean(np.argmax(lg, axis=1) == y_val)) for lg in lg_val]

    gate_x_cal, sp_cal = _build_gate_inputs_and_probs(lg_cal, temperatures)
    gate_x_val, sp_val = _build_gate_inputs_and_probs(lg_val, temperatures)
    _train_gate_network(
        gate_mlp, gate_x_cal, sp_cal, y_cal, gate_x_val, sp_val, y_val,
        epochs=80, batch_size=min(64, max(4, len(y_cal))), seed=seed,
    )

    val_w = softmax(gate_mlp.forward(gate_x_val, training=False))
    val_ens_probs = np.sum(val_w[:, :, None] * sp_val, axis=1)
    val_metrics = compute_binary_metrics(y_val, val_ens_probs)

    runtime_model = {
        "stat_mlp": stat_mlp, "seq_mlp": seq_mlp, "spat_mlp": spat_mlp, "gate_mlp": gate_mlp,
        "stat_scale": stat_scale, "stat_min": stat_min,
        "seq_scale": seq_scale, "seq_min": seq_min,
        "spat_scale": spat_scale, "spat_min": spat_min,
        "temperatures": temperatures,
    }

    test_metrics = None
    if txs is not None and y_test is not None:
        test_probs = _eval_ensemble_probs(runtime_model, txs, txq, txp)
        test_metrics = compute_binary_metrics(y_test, test_probs)

    saved_path = save_triumvir_model(
        save_model_path,
        variant=variant,
        scaler_mode=scaler_mode,
        frag_len=frag_len,
        stat_mlp=stat_mlp, seq_mlp=seq_mlp, spat_mlp=spat_mlp, gate_mlp=gate_mlp,
        stat_scale=stat_scale, stat_min=stat_min,
        seq_scale=seq_scale, seq_min=seq_min,
        spat_scale=spat_scale, spat_min=spat_min,
        temperatures=temperatures,
        expert_val_accs=expert_val_accs,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    _notify(3 * epochs + 10, 3 * epochs + 10, "Triumvir-Style Ensemble Training Complete!", f"Elapsed: {fmt_hms(time.time() - t_start)}")

    return {
        "cancelled": False,
        "model_path": saved_path,
        "arch": variant,
        "scaler_mode": scaler_mode,
        "fragment_length": frag_len,
        "train_samples": int(len(y_tr)),
        "cal_samples": int(len(y_cal)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)) if y_test is not None else 0,
        "expert_val_accs": expert_val_accs,
        "temperatures": temperatures,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "total_seconds": time.time() - t_start,
    }


# =====================================================================
# INFERENCE FUNCTIONS (USED BY DETECTION STUDIO)
# =====================================================================
def predict_triumvir_probs_batch(byte_block: np.ndarray, model: dict, pool=None) -> np.ndarray:
    xs, xq, xp = extract_triumvir_modalities_batch(byte_block, variant=model["arch"], pool=pool)
    return _eval_ensemble_probs(model, xs, xq, xp)


def predict_triumvir_batch(byte_block: np.ndarray, model: dict, pool=None) -> np.ndarray:
    probs = predict_triumvir_probs_batch(byte_block, model, pool=pool)
    return np.argmax(probs, axis=1).astype(np.uint8)


def Triumvir_test(byte_data, model: dict) -> int:
    row = np.asarray(byte_data, dtype=np.uint8)
    if row.size == 0:
        return 0
    return int(predict_triumvir_batch(row[None, :], model)[0])
