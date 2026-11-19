import os
import time
import multiprocessing as mp
import numpy as np
from scipy.fft import rfft
from scipy.stats import chi2

from BLFCP_fspecials import lempel_ziv_complexity
from nn_utils import (
    iter_uint8_csv_chunks,
    fit_scaler,
    apply_scaler,
    stratified_split,
    compute_binary_metrics,
    fmt_hms,
    LinearLayer,
    BatchNorm1dLayer,
    DropoutLayer,
    ActivationLayer,
    SequentialMLP,
)

BFLCP_VARIANTS = {
    "BFLCp_270": {
        "input_dim": 270,
        "label": "BFLCp-270 — BFD(256) + DFT(10) + LZ76(1) + Chi2 Stat/p-val(2) + MonteCarlo(1)",
    },
    "BFLCp_268": {
        "input_dim": 268,
        "label": "BFLCp-268 — BFD(256) + DFT(10) + LZ76(1) + Chi2 p-val(1) (Paper Table 5 Best)",
    },
}


# =====================================================================
# VECTORIZED BATCH FEATURE EXTRACTION (268 OR 270 FEATURES, ANY LENGTH)
# =====================================================================
def extract_bflcp_features_batch(
    byte_block: np.ndarray,
    variant: str = "BFLCp_270",
    pool=None,
) -> np.ndarray:
    """
    Extracts BFLCp features (268 or 270 columns) across a batch of uint8 fragments
    of ANY length L, using vectorized NumPy/SciPy + optional mp.Pool for LZ76.
    """
    u8 = np.asarray(byte_block, dtype=np.uint8)
    if u8.ndim == 1:
        u8 = u8[None, :]
    n_rows, frag_len = u8.shape
    dim = BFLCP_VARIANTS[variant]["input_dim"]
    if n_rows == 0 or frag_len == 0:
        return np.zeros((n_rows, dim), dtype=np.float32)

    # 1. Byte Frequency Distribution (256)
    idx_dtype = np.int32 if (n_rows * 256) < 2_000_000_000 else np.int64
    row_offsets = (np.arange(n_rows, dtype=idx_dtype) * 256)[:, None]
    counts = np.bincount(
        (u8.astype(idx_dtype) + row_offsets).ravel(),
        minlength=n_rows * 256,
    ).reshape(n_rows, 256)
    bfd = counts.astype(np.float32) / float(frag_len)

    # 2. Vectorized DFT top-5 magnitudes & normalized indices (10)
    dft_feats = np.zeros((n_rows, 10), dtype=np.float32)
    if frag_len >= 2:
        mags = np.abs(rfft(u8.astype(np.float64), axis=1))
        if mags.shape[1] > 0:
            actual_k = min(5, mags.shape[1])
            top_idx = np.argsort(mags, axis=1)[:, -actual_k:]
            top_mags = np.take_along_axis(mags, top_idx, axis=1)
            total_mags = np.sum(mags, axis=1, keepdims=True)
            safe_total = np.where(total_mags > 0.0, total_mags, 1.0)
            top_mags_norm = np.where(total_mags > 0.0, top_mags / safe_total, top_mags)
            half_len = float(max(1, frag_len // 2))
            dft_feats[:, :actual_k] = top_mags_norm.astype(np.float32)
            dft_feats[:, 5:5 + actual_k] = top_idx.astype(np.float32) / half_len

    # 3. Length-normalized Lempel-Ziv Complexity (1)
    if pool is not None and n_rows >= 64:
        lz_vals = np.asarray(
            pool.map(lempel_ziv_complexity, u8, chunksize=max(1, n_rows // 16)),
            dtype=np.float32,
        )
    else:
        lz_vals = np.fromiter(
            (lempel_ziv_complexity(row) for row in u8),
            dtype=np.float32,
            count=n_rows,
        )

    # 4. Chi-Square uniformity statistic & p-value (computed directly from counts)
    expected = float(frag_len) / 256.0
    chi2_stats = np.sum(((counts.astype(np.float64) - expected) ** 2) / expected, axis=1)
    chi2_pvals = chi2.sf(chi2_stats, df=255)

    if variant == "BFLCp_268":
        return np.hstack([
            bfd,
            dft_feats,
            lz_vals[:, None],
            chi2_pvals[:, None].astype(np.float32),
        ]).astype(np.float32, copy=False)

    # 5. Vectorized Monte Carlo Pi Approximation (1)
    num_pairs = frag_len // 2
    if num_pairs > 0:
        x = u8[:, 0:2 * num_pairs:2].astype(np.float64) / 255.0
        y = u8[:, 1:2 * num_pairs:2].astype(np.float64) / 255.0
        in_circle = np.sum(((x - 0.5) ** 2 + (y - 0.5) ** 2) <= 0.25, axis=1)
        mc_vals = (4.0 * in_circle / float(num_pairs)).astype(np.float32)
    else:
        mc_vals = np.zeros(n_rows, dtype=np.float32)

    return np.hstack([
        bfd,
        dft_feats,
        lz_vals[:, None],
        chi2_stats[:, None].astype(np.float32),
        chi2_pvals[:, None].astype(np.float32),
        mc_vals[:, None],
    ]).astype(np.float32, copy=False)


def load_bflcp_features_from_csv(
    file_path: str,
    variant: str = "BFLCp_270",
    pool=None,
    chunk_size: int = 1000,
    cancel_event=None,
    progress_cb=None,
):
    feat_chunks = []
    rows_read = 0
    detected_len = 0

    for u8_block, total_rows in iter_uint8_csv_chunks(file_path, chunk_size=chunk_size):
        if cancel_event is not None and cancel_event.is_set():
            return None, detected_len
        detected_len = u8_block.shape[1]
        feat_chunks.append(extract_bflcp_features_batch(u8_block, variant=variant, pool=pool))
        rows_read += u8_block.shape[0]
        if progress_cb is not None:
            progress_cb(rows_read, total_rows)

    if not feat_chunks:
        raise ValueError(f"No data rows found in CSV file: {file_path}")
    return np.vstack(feat_chunks).astype(np.float32, copy=False), int(detected_len)


# =====================================================================
# DNN-BFLCp ARCHITECTURE BUILDER (Skračić et al., 2023 — Fig. 1)
# =====================================================================
def build_bflcp_mlp(input_dim: int = 270, num_classes: int = 2, seed: int = 123) -> SequentialMLP:
    rng = np.random.default_rng(seed)
    return SequentialMLP([
        DropoutLayer(0.2, rng=rng),
        LinearLayer(input_dim, 1024, "he_normal", name="network.1", rng=rng),
        ActivationLayer("relu"),
        BatchNorm1dLayer(1024, name="network.3"),
        DropoutLayer(0.3, rng=rng),
        LinearLayer(1024, 1536, "he_normal", name="network.5", rng=rng),
        ActivationLayer("relu"),
        BatchNorm1dLayer(1536, name="network.7"),
        DropoutLayer(0.3, rng=rng),
        LinearLayer(1536, 1280, "he_normal", name="network.9", rng=rng),
        ActivationLayer("relu"),
        BatchNorm1dLayer(1280, name="network.11"),
        DropoutLayer(0.3, rng=rng),
        LinearLayer(1280, 2048, "he_normal", name="network.13", rng=rng),
        ActivationLayer("relu"),
        BatchNorm1dLayer(2048, name="network.15"),
        DropoutLayer(0.3, rng=rng),
        LinearLayer(2048, 1024, "he_normal", name="network.17", rng=rng),
        ActivationLayer("relu"),
        LinearLayer(1024, num_classes, "glorot_uniform", name="network.19", rng=rng),
    ])


# =====================================================================
# SAVE / LOAD / INSPECT TRAINED MODEL (.NPZ + OPTIONAL .PTH)
# =====================================================================
def save_bflcp_model(
    save_path: str,
    variant: str,
    mlp: SequentialMLP,
    scaler_mode: str,
    scaler_scale: np.ndarray,
    scaler_min: np.ndarray,
    frag_len: int,
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
        "method": np.array("BFLCp"),
        "arch": np.array(variant),
        "input_dim": np.array(BFLCP_VARIANTS[variant]["input_dim"], dtype=np.int64),
        "fragment_length": np.array(int(frag_len), dtype=np.int64),
        "scaler_mode": np.array(scaler_mode),
        "scaler_scale": scaler_scale.astype(np.float32),
        "scaler_min": scaler_min.astype(np.float32),
        "best_val_acc": np.array(float(val_metrics.get("accuracy", 0.0)), dtype=np.float64),
        "best_val_loss": np.array(float(val_metrics.get("loss", 0.0)), dtype=np.float64),
        "val_precision": np.array(float(val_metrics.get("precision", 0.0)), dtype=np.float64),
        "val_recall": np.array(float(val_metrics.get("recall", 0.0)), dtype=np.float64),
        "val_f1": np.array(float(val_metrics.get("f1", 0.0)), dtype=np.float64),
    }
    if test_metrics is not None:
        payload["test_acc"] = np.array(float(test_metrics.get("accuracy", 0.0)), dtype=np.float64)
        payload["test_f1"] = np.array(float(test_metrics.get("f1", 0.0)), dtype=np.float64)

    payload.update(mlp.export_state())
    np.savez(save_path, **payload)
    return save_path


def load_bflcp_model(model_path: str) -> dict:
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(f"BFLCp model file not found: {model_path}")

    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".pth":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Loading '.pth' checkpoints requires PyTorch, which is not installed in this environment.\n"
                "Please load the '.npz' weights file or train a new BFLCp model inside AFRL Studio."
            ) from exc

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        raw_state = ckpt.get("model_state_dict", ckpt)
        state = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in raw_state.items() if hasattr(v, "detach")}
        if "network.1.weight" not in state:
            raise ValueError(f"This .pth file is not a valid DNN_BFLCp checkpoint: {model_path}")

        in_dim = int(state["network.1.weight"].shape[1])
        variant = str(ckpt.get("arch", "BFLCp_268" if in_dim == 268 else "BFLCp_270"))
        scaler_mode = str(ckpt.get("scaler_mode", "standard"))

        if "scaler_min" in ckpt and "scaler_scale" in ckpt:
            scale_vec = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
            min_vec = np.asarray(ckpt["scaler_min"], dtype=np.float32)
        elif "scaler_mean" in ckpt and "scaler_scale" in ckpt:
            std_v = np.asarray(ckpt["scaler_scale"], dtype=np.float64)
            std_v = np.where(std_v == 0.0, 1.0, std_v)
            mean_v = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
            scale_vec = (1.0 / std_v).astype(np.float32)
            min_vec = (-mean_v / std_v).astype(np.float32)
        else:
            scale_vec = np.ones(in_dim, dtype=np.float32)
            min_vec = np.zeros(in_dim, dtype=np.float32)

        frag_len = int(ckpt.get("fragment_length", 0))
        best_val_acc = float(ckpt.get("best_val_acc", ckpt.get("val_accuracy", 0.0) / 100.0))
    else:
        with np.load(model_path, allow_pickle=False) as data:
            if "method" in data and str(data["method"].item()) != "BFLCp":
                raise ValueError(f"Selected model is '{data['method'].item()}', not a BFLCp model.")
            if "network.1.weight" not in data:
                raise ValueError(f"Invalid BFLCp model file (missing DNN_BFLCp layers): {model_path}")

            state = {k: np.asarray(data[k], dtype=np.float32) for k in data.files if k.startswith("network.")}
            in_dim = int(state["network.1.weight"].shape[1])
            variant = str(data["arch"].item()) if "arch" in data else ("BFLCp_268" if in_dim == 268 else "BFLCp_270")
            scaler_mode = str(data["scaler_mode"].item()) if "scaler_mode" in data else "standard"
            scale_vec = np.asarray(data["scaler_scale"], dtype=np.float32)
            min_vec = np.asarray(data["scaler_min"], dtype=np.float32)
            frag_len = int(data["fragment_length"].item()) if "fragment_length" in data else 0
            best_val_acc = float(data["best_val_acc"].item()) if "best_val_acc" in data else 0.0

    mlp = build_bflcp_mlp(input_dim=in_dim)
    mlp.load_state(state)
    return {
        "path": model_path,
        "method": "BFLCp",
        "arch": variant,
        "input_dim": in_dim,
        "mlp": mlp,
        "scaler_mode": scaler_mode,
        "scaler_scale": scale_vec,
        "scaler_min": min_vec,
        "fragment_length": frag_len,
        "best_val_acc": best_val_acc,
    }


def inspect_bflcp_model_summary(model_path: str) -> str:
    try:
        m = load_bflcp_model(model_path)
        acc_str = f"Val Acc: {m['best_val_acc'] * 100:.2f}%" if m["best_val_acc"] > 0 else "Loaded"
        len_str = f"Train L={m['fragment_length']}B" if m["fragment_length"] > 0 else "Any Length"
        return f"Ready  │  {m['arch']} ({m['input_dim']} feats)  │  {m['scaler_mode']}  │  {len_str}  │  {acc_str}"
    except Exception as e:
        return f"Model error: {e}"


# =====================================================================
# TRAINING PIPELINE (CALLED BY GUI THREAD)
# =====================================================================
def train_bflcp_model(
    unenc_train_csv: str,
    enc_train_csv: str,
    save_model_path: str,
    test_unenc_csv: str = "",
    test_enc_csv: str = "",
    variant: str = "BFLCp_270",
    scaler_mode: str = "standard",
    epochs: int = 60,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_ratio: float = 0.20,
    early_stop_patience: int = 15,
    cpu_percent: int = 80,
    seed: int = 42,
    cancel_event=None,
    status_cb=None,
) -> dict:
    if epochs < 1 or batch_size < 2:
        raise ValueError("Epochs must be >= 1 and Batch Size must be >= 2 (required for BatchNorm1d).")
    if not (0.01 <= val_ratio <= 0.90):
        raise ValueError("Validation split ratio must be between 0.01 and 0.90.")
    if not save_model_path or not save_model_path.strip():
        raise ValueError("Please specify a save path for the trained BFLCp model (.npz).")
    if bool(test_unenc_csv) != bool(test_enc_csv):
        raise ValueError("Provide both Unencrypted and Encrypted Test CSVs, or leave both empty.")

    def _notify(cur, total, status_txt, time_txt):
        if status_cb is not None:
            status_cb(cur, total, status_txt, time_txt)

    t_start = time.time()
    num_cores = max(1, int(mp.cpu_count() * cpu_percent / 100))

    _notify(0, 100, f"Stage 1/3: Extracting {variant} features ({num_cores} cores)...", "Elapsed: 00:00:00")
    pool_ctx = mp.Pool(processes=num_cores) if num_cores > 1 else None
    try:
        x0, len0 = load_bflcp_features_from_csv(
            unenc_train_csv,
            variant=variant,
            pool=pool_ctx,
            cancel_event=cancel_event,
            progress_cb=lambda r, tot: _notify(min(25, int(25 * r / max(1, tot))), 100, f"Stage 1/3: Extracting unencrypted features ({r}/{tot})...", f"Elapsed: {fmt_hms(time.time() - t_start)}"),
        )
        if x0 is None:
            return {"cancelled": True}

        _notify(25, 100, f"Stage 1/3: Extracting encrypted features ({num_cores} cores)...", f"Elapsed: {fmt_hms(time.time() - t_start)}")
        x1, len1 = load_bflcp_features_from_csv(
            enc_train_csv,
            variant=variant,
            pool=pool_ctx,
            cancel_event=cancel_event,
            progress_cb=lambda r, tot: _notify(25 + min(25, int(25 * r / max(1, tot))), 100, f"Stage 1/3: Extracting encrypted features ({r}/{tot})...", f"Elapsed: {fmt_hms(time.time() - t_start)}"),
        )
        if x1 is None:
            return {"cancelled": True}

        X_test_raw, y_test = None, None
        if test_unenc_csv and test_enc_csv and os.path.isfile(test_unenc_csv) and os.path.isfile(test_enc_csv):
            _notify(50, 100, "Stage 2/3: Extracting test evaluation features...", f"Elapsed: {fmt_hms(time.time() - t_start)}")
            tx0, _ = load_bflcp_features_from_csv(test_unenc_csv, variant=variant, pool=pool_ctx, cancel_event=cancel_event)
            if tx0 is None:
                return {"cancelled": True}
            tx1, _ = load_bflcp_features_from_csv(test_enc_csv, variant=variant, pool=pool_ctx, cancel_event=cancel_event)
            if tx1 is None:
                return {"cancelled": True}
            X_test_raw = np.vstack([tx0, tx1]).astype(np.float32, copy=False)
            y_test = np.concatenate([np.zeros(len(tx0), dtype=np.int64), np.ones(len(tx1), dtype=np.int64)])
    finally:
        if pool_ctx is not None:
            pool_ctx.close()
            pool_ctx.join()

    X_all = np.vstack([x0, x1]).astype(np.float32, copy=False)
    y_all = np.concatenate([np.zeros(len(x0), dtype=np.int64), np.ones(len(x1), dtype=np.int64)])
    if len(X_all) < 4:
        raise ValueError("Need at least 4 total training fragments to split train/validation sets with BatchNorm1d.")
    frag_len = len0 if len0 == len1 else len0
    del x0, x1

    # Stratified split BEFORE fitting scaler (zero data leakage)
    X_train_raw, X_val_raw, y_train, y_val = stratified_split(X_all, y_all, val_ratio=val_ratio, seed=seed)
    del X_all, y_all

    scale_vec, min_vec = fit_scaler(X_train_raw, mode=scaler_mode)
    X_train = apply_scaler(X_train_raw, scale_vec, min_vec)
    X_val = apply_scaler(X_val_raw, scale_vec, min_vec)
    del X_train_raw, X_val_raw

    X_test = apply_scaler(X_test_raw, scale_vec, min_vec) if X_test_raw is not None else None
    del X_test_raw

    input_dim = BFLCP_VARIANTS[variant]["input_dim"]
    mlp = build_bflcp_mlp(input_dim=input_dim, num_classes=2, seed=seed)
    _notify(
        0,
        epochs,
        f"Training {variant} ({len(y_train)} train, {len(y_val)} val)...",
        f"Elapsed: {fmt_hms(time.time() - t_start)} | LR: {learning_rate:.1e}",
    )

    train_res = mlp.train_loop(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stop_patience=early_stop_patience,
        seed=seed,
        cancel_event=cancel_event,
        progress_cb=_notify,
    )
    if train_res.get("cancelled"):
        return {"cancelled": True}

    test_metrics = None
    if X_test is not None and y_test is not None:
        test_probs = mlp.eval_probs(X_test)
        test_metrics = compute_binary_metrics(y_test, test_probs)

    saved_path = save_bflcp_model(
        save_model_path,
        variant=variant,
        mlp=mlp,
        scaler_mode=scaler_mode,
        scaler_scale=scale_vec,
        scaler_min=min_vec,
        frag_len=frag_len,
        val_metrics=train_res["val_metrics"],
        test_metrics=test_metrics,
    )

    return {
        "cancelled": False,
        "model_path": saved_path,
        "arch": variant,
        "scaler_mode": scaler_mode,
        "fragment_length": frag_len,
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)) if y_test is not None else 0,
        "val_metrics": train_res["val_metrics"],
        "test_metrics": test_metrics,
        "total_seconds": time.time() - t_start,
    }


# =====================================================================
# INFERENCE FUNCTIONS (USED BY DETECTION STUDIO)
# =====================================================================
def predict_bflcp_probs_batch(byte_block: np.ndarray, model: dict, pool=None) -> np.ndarray:
    feats = extract_bflcp_features_batch(byte_block, variant=model["arch"], pool=pool)
    x_scaled = apply_scaler(feats, model["scaler_scale"], model["scaler_min"])
    return model["mlp"].eval_probs(x_scaled)


def predict_bflcp_batch(byte_block: np.ndarray, model: dict, pool=None) -> np.ndarray:
    probs = predict_bflcp_probs_batch(byte_block, model, pool=pool)
    return np.argmax(probs, axis=1).astype(np.uint8)


def BFLCp_test(byte_data, model: dict) -> int:
    row = np.asarray(byte_data, dtype=np.uint8)
    if row.size == 0:
        return 0
    return int(predict_bflcp_batch(row[None, :], model)[0])
