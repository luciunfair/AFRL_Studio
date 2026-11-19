import os
import time
import numpy as np

from nn_utils import (
    iter_uint8_csv_chunks,
    fit_scaler,
    apply_scaler,
    stratified_split,
    compute_binary_metrics,
    fmt_hms,
    LinearLayer,
    ActivationLayer,
    SequentialMLP,
)

ENCOD_ARCHITECTURES = {
    "BinaryAll_SELU": {
        "dims": [256, 256, 256, 256, 128, 2],
        "activation": "selu",
        "init": "lecun_normal",
        "label": "BinaryAll — 5-Layer SELU + LeCun Normal (All Formats, Paper Fig. 2b)",
    },
    "BinarySingle_ReLU": {
        "dims": [256, 256, 256, 64, 2],
        "activation": "relu",
        "init": "glorot_uniform",
        "label": "BinarySingle — 4-Layer ReLU + Glorot Uniform (Single Format, Paper Fig. 2a)",
    },
}


# =====================================================================
# VECTORIZED BYTE HISTOGRAM EXTRACTION (ANY FRAGMENT LENGTH)
# =====================================================================
def extract_byte_histograms_batch(byte_block: np.ndarray) -> np.ndarray:
    """
    Computes normalized 256-bin byte probability histograms (sum = 1.0 per fragment)
    for ANY fragment length L using vectorized offset bincount.
    """
    u8 = np.asarray(byte_block, dtype=np.uint8)
    if u8.ndim == 1:
        u8 = u8[None, :]
    n_rows, frag_len = u8.shape
    if n_rows == 0 or frag_len == 0:
        return np.zeros((n_rows, 256), dtype=np.float32)

    idx_dtype = np.int32 if (n_rows * 256) < 2_000_000_000 else np.int64
    row_offsets = (np.arange(n_rows, dtype=idx_dtype) * 256)[:, None]
    counts = np.bincount(
        (u8.astype(idx_dtype) + row_offsets).ravel(),
        minlength=n_rows * 256,
    ).reshape(n_rows, 256)
    return counts.astype(np.float32) / float(frag_len)


def load_histograms_from_csv(
    file_path: str,
    chunk_size: int = 2000,
    cancel_event=None,
    progress_cb=None,
):
    """Streams a raw byte-fragment CSV and returns (histograms_float32, fragment_length)."""
    hist_chunks = []
    rows_read = 0
    detected_len = 0

    for u8_block, total_rows in iter_uint8_csv_chunks(file_path, chunk_size=chunk_size):
        if cancel_event is not None and cancel_event.is_set():
            return None, detected_len
        detected_len = u8_block.shape[1]
        hist_chunks.append(extract_byte_histograms_batch(u8_block))
        rows_read += u8_block.shape[0]
        if progress_cb is not None:
            progress_cb(rows_read, total_rows)

    if not hist_chunks:
        raise ValueError(f"No data rows found in CSV file: {file_path}")
    return np.vstack(hist_chunks).astype(np.float32, copy=False), int(detected_len)


# =====================================================================
# MODEL BUILDER, SAVE / LOAD / INSPECT
# =====================================================================
def build_encod_mlp(arch_name: str = "BinaryAll_SELU", seed: int = 123) -> SequentialMLP:
    spec = ENCOD_ARCHITECTURES[arch_name]
    dims = spec["dims"]
    act = spec["activation"]
    init_kind = spec["init"]
    rng = np.random.default_rng(seed)

    layers = []
    n_lin = len(dims) - 1
    for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:]), start=1):
        layers.append(LinearLayer(d_in, d_out, init_kind=init_kind, name=f"fc{i}", rng=rng))
        if i < n_lin:
            layers.append(ActivationLayer(act))
    return SequentialMLP(layers)


def save_encod_model(
    save_path: str,
    arch_name: str,
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
    activation = ENCOD_ARCHITECTURES[arch_name]["activation"]
    payload = {
        "method": np.array("EnCoD"),
        "arch": np.array(arch_name),
        "activation": np.array(activation),
        "num_layers": np.array(len(ENCOD_ARCHITECTURES[arch_name]["dims"]) - 1, dtype=np.int64),
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


def load_encod_model(model_path: str) -> dict:
    """Loads a trained EnCoD model (.npz or .pth if PyTorch is installed)."""
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(f"EnCoD model file not found: {model_path}")

    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".pth":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Loading '.pth' checkpoints requires PyTorch, which is not installed in this environment.\n"
                "Please load the '.npz' weights file or train a new EnCoD model inside AFRL Studio."
            ) from exc

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        raw_state = ckpt.get("model_state_dict", ckpt)
        state = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in raw_state.items() if hasattr(v, "detach")}
        if "fc1.weight" not in state:
            raise ValueError(f"This .pth file is not a valid EnCoD checkpoint: {model_path}")

        arch = str(ckpt.get("arch", "BinaryAll_SELU" if "fc5.weight" in state else "BinarySingle_ReLU"))
        scaler_mode = str(ckpt.get("scaler_mode", "minmax"))
        scale_vec = np.asarray(ckpt.get("scaler_scale", np.full(256, 2.0)), dtype=np.float32)
        min_vec = np.asarray(ckpt.get("scaler_min", np.zeros(256)), dtype=np.float32)
        frag_len = int(ckpt.get("fragment_length", 0))
        best_val_acc = float(ckpt.get("best_val_acc", 0.0))
    else:
        with np.load(model_path, allow_pickle=False) as data:
            if "method" in data and str(data["method"].item()) != "EnCoD":
                raise ValueError(f"Selected model is '{data['method'].item()}', not an EnCoD model.")
            if "fc1.weight" not in data:
                raise ValueError(f"Invalid EnCoD model file (missing fc1.weight): {model_path}")

            state = {k: np.asarray(data[k], dtype=np.float32) for k in data.files if k.startswith("fc")}
            arch = str(data["arch"].item()) if "arch" in data else ("BinaryAll_SELU" if "fc5.weight" in state else "BinarySingle_ReLU")
            scaler_mode = str(data["scaler_mode"].item()) if "scaler_mode" in data else "minmax"
            scale_vec = np.asarray(data["scaler_scale"], dtype=np.float32) if "scaler_scale" in data else np.full(256, 2.0, dtype=np.float32)
            min_vec = np.asarray(data["scaler_min"], dtype=np.float32) if "scaler_min" in data else np.zeros(256, dtype=np.float32)
            frag_len = int(data["fragment_length"].item()) if "fragment_length" in data else 0
            best_val_acc = float(data["best_val_acc"].item()) if "best_val_acc" in data else 0.0

    mlp = build_encod_mlp(arch)
    mlp.load_state(state)
    return {
        "path": model_path,
        "method": "EnCoD",
        "arch": arch,
        "mlp": mlp,
        "scaler_mode": scaler_mode,
        "scaler_scale": scale_vec,
        "scaler_min": min_vec,
        "fragment_length": frag_len,
        "best_val_acc": best_val_acc,
    }


def inspect_encod_model_summary(model_path: str) -> str:
    try:
        m = load_encod_model(model_path)
        acc_str = f"Val Acc: {m['best_val_acc'] * 100:.2f}%" if m["best_val_acc"] > 0 else "Loaded"
        len_str = f"Train L={m['fragment_length']}B" if m["fragment_length"] > 0 else "Any Length"
        return f"Ready  │  {m['arch']}  │  Scaler: {m['scaler_mode']}  │  {len_str}  │  {acc_str}"
    except Exception as e:
        return f"Model error: {e}"


# =====================================================================
# TRAINING PIPELINE (CALLED BY GUI THREAD)
# =====================================================================
def train_encod_model(
    unenc_train_csv: str,
    enc_train_csv: str,
    save_model_path: str,
    test_unenc_csv: str = "",
    test_enc_csv: str = "",
    arch_name: str = "BinaryAll_SELU",
    scaler_mode: str = "minmax",
    epochs: int = 60,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    val_ratio: float = 0.15,
    early_stop_patience: int = 15,
    seed: int = 123,
    cancel_event=None,
    status_cb=None,
) -> dict:
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive integers.")
    if not (0.01 <= val_ratio <= 0.90):
        raise ValueError("Validation split ratio must be between 0.01 and 0.90.")
    if not save_model_path or not save_model_path.strip():
        raise ValueError("Please specify a save path for the trained EnCoD model (.npz).")
    if bool(test_unenc_csv) != bool(test_enc_csv):
        raise ValueError("Provide both Unencrypted and Encrypted Test CSVs, or leave both empty.")

    def _notify(cur, total, status_txt, time_txt):
        if status_cb is not None:
            status_cb(cur, total, status_txt, time_txt)

    t_start = time.time()
    _notify(0, 100, "Stage 1/3: Loading unencrypted training fragments...", "Elapsed: 00:00:00")
    x0, len0 = load_histograms_from_csv(
        unenc_train_csv,
        cancel_event=cancel_event,
        progress_cb=lambda r, tot: _notify(min(25, int(25 * r / max(1, tot))), 100, f"Stage 1/3: Reading unencrypted CSV ({r}/{tot} rows)...", f"Elapsed: {fmt_hms(time.time() - t_start)}"),
    )
    if x0 is None:
        return {"cancelled": True}

    _notify(25, 100, "Stage 1/3: Loading encrypted training fragments...", f"Elapsed: {fmt_hms(time.time() - t_start)}")
    x1, len1 = load_histograms_from_csv(
        enc_train_csv,
        cancel_event=cancel_event,
        progress_cb=lambda r, tot: _notify(25 + min(25, int(25 * r / max(1, tot))), 100, f"Stage 1/3: Reading encrypted CSV ({r}/{tot} rows)...", f"Elapsed: {fmt_hms(time.time() - t_start)}"),
    )
    if x1 is None:
        return {"cancelled": True}

    X_all = np.vstack([x0, x1]).astype(np.float32, copy=False)
    y_all = np.concatenate([np.zeros(len(x0), dtype=np.int64), np.ones(len(x1), dtype=np.int64)])
    if len(X_all) < 2:
        raise ValueError("Need at least 2 total training fragments to split train/validation sets.")
    frag_len = len0 if len0 == len1 else len0
    del x0, x1

    X_train_raw, X_val_raw, y_train, y_val = stratified_split(X_all, y_all, val_ratio=val_ratio, seed=seed)
    del X_all, y_all

    scale_vec, min_vec = fit_scaler(X_train_raw, mode=scaler_mode)
    X_train = apply_scaler(X_train_raw, scale_vec, min_vec)
    X_val = apply_scaler(X_val_raw, scale_vec, min_vec)
    del X_train_raw, X_val_raw

    X_test, y_test = None, None
    if test_unenc_csv and test_enc_csv and os.path.isfile(test_unenc_csv) and os.path.isfile(test_enc_csv):
        _notify(50, 100, "Stage 2/3: Loading test evaluation CSVs...", f"Elapsed: {fmt_hms(time.time() - t_start)}")
        tx0, _ = load_histograms_from_csv(test_unenc_csv, cancel_event=cancel_event)
        if tx0 is None:
            return {"cancelled": True}
        tx1, _ = load_histograms_from_csv(test_enc_csv, cancel_event=cancel_event)
        if tx1 is None:
            return {"cancelled": True}
        X_test = apply_scaler(np.vstack([tx0, tx1]).astype(np.float32, copy=False), scale_vec, min_vec)
        y_test = np.concatenate([np.zeros(len(tx0), dtype=np.int64), np.ones(len(tx1), dtype=np.int64)])

    mlp = build_encod_mlp(arch_name, seed=seed)
    _notify(
        0,
        epochs,
        f"Training {arch_name} ({len(y_train)} train, {len(y_val)} val)...",
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

    saved_path = save_encod_model(
        save_model_path,
        arch_name=arch_name,
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
        "arch": arch_name,
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
def predict_encod_probs_batch(byte_block: np.ndarray, model: dict) -> np.ndarray:
    hists = extract_byte_histograms_batch(byte_block)
    x_scaled = apply_scaler(hists, model["scaler_scale"], model["scaler_min"])
    return model["mlp"].eval_probs(x_scaled)


def predict_encod_batch(byte_block: np.ndarray, model: dict) -> np.ndarray:
    probs = predict_encod_probs_batch(byte_block, model)
    return np.argmax(probs, axis=1).astype(np.uint8)


def EnCoD_test(byte_data, model: dict) -> int:
    row = np.asarray(byte_data, dtype=np.uint8)
    if row.size == 0:
        return 0
    return int(predict_encod_batch(row[None, :], model)[0])
