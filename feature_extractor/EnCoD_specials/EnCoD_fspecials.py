import math
import os
import time
import numpy as np
import pandas as pd

_SELU_ALPHA = 1.6732632423543772848170429916717
_SELU_SCALE = 1.0507009873554804934193349852946

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
# CSV HEADER DETECTION & VECTORIZED BYTE HISTOGRAM EXTRACTION
# =====================================================================
def _detect_header_and_frag_width(file_path: str):
    """Detects optional CSV header row and exact fragment byte width."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        line0 = f.readline().strip()
        line1 = f.readline().strip()

    if not line0:
        return 0, 0

    cols0 = [c for c in line0.split(",") if c != ""]
    if not line1:
        try:
            float(cols0[0].strip())
            return 0, len(cols0)
        except ValueError:
            return 1, len(cols0)

    cols1 = [c for c in line1.split(",") if c != ""]
    is_non_numeric = False
    try:
        float(cols0[0].strip())
    except ValueError:
        is_non_numeric = True

    if is_non_numeric or (len(cols0) != len(cols1)):
        return 1, len(cols1)
    return 0, len(cols0)


def _fast_count_lines(file_path: str) -> int:
    """Fast binary buffered newline counter for large CSV files (excludes detected header)."""
    skip_rows, _ = _detect_header_and_frag_width(file_path)
    count = 0
    with open(file_path, "rb") as f:
        while True:
            buf = f.read(1048576)
            if not buf:
                break
            count += buf.count(b"\n")
    return max(1, count - skip_rows)


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


def _chunk_df_to_uint8(chunk_df: pd.DataFrame) -> np.ndarray:
    arr = chunk_df.to_numpy()
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, 255).astype(np.uint8, copy=False)
    if np.issubdtype(arr.dtype, np.floating):
        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=0.0)
        return np.rint(np.clip(arr, 0.0, 255.0)).astype(np.uint8)
    num_df = chunk_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return np.rint(np.clip(num_df.to_numpy(dtype=np.float64), 0.0, 255.0)).astype(np.uint8)


def load_histograms_from_csv(
    file_path: str,
    chunk_size: int = 2000,
    cancel_event=None,
    progress_cb=None,
):
    """
    Streams a raw byte-fragment CSV in memory-safe chunks and returns
    (histograms_float32, fragment_length).
    """
    skip_rows, data_cols = _detect_header_and_frag_width(file_path)
    if data_cols <= 0:
        raise ValueError(f"Empty or invalid CSV file: {file_path}")

    total_rows = _fast_count_lines(file_path)
    usecols = list(range(data_cols))
    hist_chunks = []
    rows_read = 0
    detected_len = data_cols

    for chunk_df in pd.read_csv(
        file_path,
        header=None,
        skiprows=skip_rows,
        usecols=usecols,
        chunksize=chunk_size,
        engine="c",
    ):
        if cancel_event is not None and cancel_event.is_set():
            return None, detected_len
        if chunk_df.empty:
            continue

        u8 = _chunk_df_to_uint8(chunk_df)
        detected_len = u8.shape[1]
        hist_chunks.append(extract_byte_histograms_batch(u8))
        rows_read += u8.shape[0]
        if progress_cb is not None:
            progress_cb(rows_read, total_rows)

    if not hist_chunks:
        raise ValueError(f"No data rows found in CSV file: {file_path}")

    return np.vstack(hist_chunks).astype(np.float32, copy=False), int(detected_len)


# =====================================================================
# SCALER, SPLIT, ACTIVATIONS & METRICS (PURE NUMPY)
# =====================================================================
def _fit_scaler(X_train: np.ndarray, scaler_mode: str):
    if scaler_mode == "minmax":
        data_min = X_train.min(axis=0).astype(np.float32)
        data_max = X_train.max(axis=0).astype(np.float32)
        data_range = data_max - data_min
        data_range[data_range == 0.0] = 1.0
        scale_vec = (2.0 / data_range).astype(np.float32)
        min_vec = (-data_min * scale_vec).astype(np.float32)
    elif scaler_mode == "fixed":
        scale_vec = np.full(256, 2.0, dtype=np.float32)
        min_vec = np.zeros(256, dtype=np.float32)
    else:
        raise ValueError(f"Unknown scaler_mode: {scaler_mode}")
    return scale_vec, min_vec


def _apply_scaler(X: np.ndarray, scale_vec: np.ndarray, min_vec: np.ndarray) -> np.ndarray:
    return (X * scale_vec + min_vec).astype(np.float32, copy=False)


def _stratified_split(X: np.ndarray, y: np.ndarray, val_ratio: float, seed: int = 123):
    rng = np.random.default_rng(seed)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)

    if idx0.size < 2 or idx1.size < 2 or val_ratio <= 0.0 or val_ratio >= 1.0:
        all_idx = rng.permutation(len(y))
        n_val = max(1, min(len(y) - 1, int(round(len(y) * val_ratio))))
        return X[all_idx[n_val:]], X[all_idx[:n_val]], y[all_idx[n_val:]], y[all_idx[:n_val]]

    rng.shuffle(idx0)
    rng.shuffle(idx1)
    v0 = max(1, min(idx0.size - 1, int(round(idx0.size * val_ratio))))
    v1 = max(1, min(idx1.size - 1, int(round(idx1.size * val_ratio))))

    val_idx = np.concatenate([idx0[:v0], idx1[:v1]])
    train_idx = np.concatenate([idx0[v0:], idx1[v1:]])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def _activate(z: np.ndarray, kind: str) -> np.ndarray:
    dt = z.dtype if z.dtype in (np.float32, np.float64) else np.float32
    if kind == "selu":
        scale = dt.type(_SELU_SCALE)
        alpha = dt.type(_SELU_ALPHA)
        z_neg = np.minimum(z, dt.type(0.0))
        return np.where(z > 0.0, scale * z, scale * alpha * (np.exp(z_neg) - dt.type(1.0))).astype(dt, copy=False)
    if kind == "relu":
        return np.maximum(dt.type(0.0), z)
    raise ValueError(f"Unsupported activation: {kind}")


def _activate_grad(z: np.ndarray, a: np.ndarray, kind: str) -> np.ndarray:
    dt = z.dtype if z.dtype in (np.float32, np.float64) else np.float32
    if kind == "selu":
        scale = dt.type(_SELU_SCALE)
        sa = dt.type(_SELU_SCALE * _SELU_ALPHA)
        return np.where(z > 0.0, scale, a + sa)
    if kind == "relu":
        return (z > 0.0).astype(dt)
    raise ValueError(f"Unsupported activation: {kind}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    dt = logits.dtype if logits.dtype in (np.float32, np.float64) else np.float32
    z_shift = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return (exp_z / np.sum(exp_z, axis=1, keepdims=True)).astype(dt, copy=False)


def _init_weights(arch_name: str, seed: int = 123):
    spec = ENCOD_ARCHITECTURES[arch_name]
    dims = spec["dims"]
    init_kind = spec["init"]
    rng = np.random.default_rng(seed)

    weights = []
    biases = []
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        if init_kind == "lecun_normal":
            std = math.sqrt(1.0 / float(fan_in))
            w = rng.normal(loc=0.0, scale=std, size=(fan_in, fan_out)).astype(np.float32)
        elif init_kind == "glorot_uniform":
            limit = math.sqrt(6.0 / float(fan_in + fan_out))
            w = rng.uniform(low=-limit, high=limit, size=(fan_in, fan_out)).astype(np.float32)
        else:
            raise ValueError(f"Unknown init kind: {init_kind}")
        b = np.zeros(fan_out, dtype=np.float32)
        weights.append(w)
        biases.append(b)
    return weights, biases, spec["activation"]


def _forward_eval(X: np.ndarray, weights, biases, activation: str, batch_size: int = 2048):
    n = X.shape[0]
    n_layers = len(weights)
    probs_out = np.empty((n, weights[-1].shape[1]), dtype=np.float32)

    for start in range(0, n, batch_size):
        a = X[start:start + batch_size]
        for i in range(n_layers - 1):
            a = _activate(a @ weights[i] + biases[i], activation)
        logits = a @ weights[-1] + biases[-1]
        probs_out[start:start + batch_size] = _softmax(logits)

    return probs_out


def _compute_binary_metrics(y_true: np.ndarray, probs: np.ndarray):
    preds = np.argmax(probs, axis=1)
    y_true_i = y_true.astype(np.int64)
    tp = int(np.sum((preds == 1) & (y_true_i == 1)))
    fp = int(np.sum((preds == 1) & (y_true_i == 0)))
    tn = int(np.sum((preds == 0) & (y_true_i == 0)))
    fn = int(np.sum((preds == 0) & (y_true_i == 1)))

    total = max(1, len(y_true_i))
    acc = (tp + tn) / float(total)
    prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
    loss = -float(np.mean(np.log(np.clip(probs[np.arange(len(y_true_i)), y_true_i], 1e-12, 1.0))))
    return {
        "loss": loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


# =====================================================================
# SAVE / LOAD / INSPECT TRAINED MODEL (.NPZ + OPTIONAL .PTH)
# =====================================================================
def save_encod_model(
    save_path: str,
    arch_name: str,
    activation: str,
    weights,
    biases,
    scaler_mode: str,
    scaler_scale: np.ndarray,
    scaler_min: np.ndarray,
    frag_len: int,
    val_metrics: dict,
    test_metrics: dict = None,
):
    parent = os.path.dirname(os.path.abspath(save_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not save_path.lower().endswith(".npz"):
        save_path = save_path + ".npz"

    payload = {
        "arch": np.array(arch_name),
        "activation": np.array(activation),
        "num_layers": np.array(len(weights), dtype=np.int64),
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

    # PyTorch-compatible layer naming (fc1.weight transposed to (out, in))
    for idx, (w, b) in enumerate(zip(weights, biases), start=1):
        payload[f"fc{idx}.weight"] = w.T.astype(np.float32)
        payload[f"fc{idx}.bias"] = b.astype(np.float32)

    np.savez(save_path, **payload)
    return save_path


def load_encod_model(model_path: str) -> dict:
    """
    Loads a trained EnCoD model (.npz saved by AFRL Studio or EnCoD_train.py,
    or .pth if PyTorch is installed).
    """
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
        state = ckpt.get("model_state_dict", ckpt)
        arch = str(ckpt.get("arch", "BinaryAll_SELU"))
        activation = ENCOD_ARCHITECTURES.get(arch, {}).get("activation", "selu")
        scale_vec = np.asarray(ckpt.get("scaler_scale", np.full(256, 2.0)), dtype=np.float32)
        min_vec = np.asarray(ckpt.get("scaler_min", np.zeros(256)), dtype=np.float32)

        weights, biases = [], []
        idx = 1
        while f"fc{idx}.weight" in state:
            weights.append(state[f"fc{idx}.weight"].detach().cpu().numpy().T.astype(np.float32))
            biases.append(state[f"fc{idx}.bias"].detach().cpu().numpy().astype(np.float32))
            idx += 1

        return {
            "path": model_path,
            "arch": arch,
            "activation": activation,
            "weights": weights,
            "biases": biases,
            "scaler_mode": str(ckpt.get("scaler_mode", "minmax")),
            "scaler_scale": scale_vec,
            "scaler_min": min_vec,
            "fragment_length": int(ckpt.get("fragment_length", 0)),
            "best_val_acc": float(ckpt.get("best_val_acc", 0.0)),
        }

    with np.load(model_path, allow_pickle=False) as data:
        arch = str(data["arch"].item()) if "arch" in data else "BinaryAll_SELU"
        activation = (
            str(data["activation"].item())
            if "activation" in data
            else ENCOD_ARCHITECTURES.get(arch, {}).get("activation", "selu")
        )
        scaler_mode = str(data["scaler_mode"].item()) if "scaler_mode" in data else "minmax"
        scale_vec = np.asarray(data["scaler_scale"], dtype=np.float32) if "scaler_scale" in data else np.full(256, 2.0, dtype=np.float32)
        min_vec = np.asarray(data["scaler_min"], dtype=np.float32) if "scaler_min" in data else np.zeros(256, dtype=np.float32)
        frag_len = int(data["fragment_length"].item()) if "fragment_length" in data else 0
        best_val_acc = float(data["best_val_acc"].item()) if "best_val_acc" in data else 0.0

        weights, biases = [], []
        idx = 1
        while f"fc{idx}.weight" in data:
            weights.append(np.asarray(data[f"fc{idx}.weight"], dtype=np.float32).T.copy())
            biases.append(np.asarray(data[f"fc{idx}.bias"], dtype=np.float32).copy())
            idx += 1

    if not weights:
        raise ValueError(f"Invalid EnCoD model file (no layer weights found): {model_path}")

    return {
        "path": model_path,
        "arch": arch,
        "activation": activation,
        "weights": weights,
        "biases": biases,
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


def _fmt_hms(seconds: float) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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
    """
    Trains an EnCoD classifier directly from unencrypted & encrypted fragment CSVs
    and saves the resulting weights + MinMax scaler parameters to a .npz file.
    """
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
        progress_cb=lambda r, tot: _notify(min(25, int(25 * r / max(1, tot))), 100, f"Stage 1/3: Reading unencrypted CSV ({r}/{tot} rows)...", f"Elapsed: {_fmt_hms(time.time() - t_start)}"),
    )
    if x0 is None:
        return {"cancelled": True}

    _notify(25, 100, "Stage 1/3: Loading encrypted training fragments...", f"Elapsed: {_fmt_hms(time.time() - t_start)}")
    x1, len1 = load_histograms_from_csv(
        enc_train_csv,
        cancel_event=cancel_event,
        progress_cb=lambda r, tot: _notify(25 + min(25, int(25 * r / max(1, tot))), 100, f"Stage 1/3: Reading encrypted CSV ({r}/{tot} rows)...", f"Elapsed: {_fmt_hms(time.time() - t_start)}"),
    )
    if x1 is None:
        return {"cancelled": True}

    X_all = np.vstack([x0, x1]).astype(np.float32, copy=False)
    y_all = np.concatenate([
        np.zeros(len(x0), dtype=np.int64),
        np.ones(len(x1), dtype=np.int64),
    ])
    if len(X_all) < 2:
        raise ValueError("Need at least 2 total training fragments to split train/validation sets.")
    frag_len = len0 if len0 == len1 else len0
    del x0, x1

    X_train_raw, X_val_raw, y_train, y_val = _stratified_split(X_all, y_all, val_ratio=val_ratio, seed=seed)
    del X_all, y_all

    # Fit scaler strictly on training split (zero data leakage)
    scale_vec, min_vec = _fit_scaler(X_train_raw, scaler_mode=scaler_mode)
    X_train = _apply_scaler(X_train_raw, scale_vec, min_vec)
    X_val = _apply_scaler(X_val_raw, scale_vec, min_vec)
    del X_train_raw, X_val_raw

    # Optional test CSVs
    X_test, y_test = None, None
    if test_unenc_csv and test_enc_csv and os.path.isfile(test_unenc_csv) and os.path.isfile(test_enc_csv):
        _notify(50, 100, "Stage 2/3: Loading test evaluation CSVs...", f"Elapsed: {_fmt_hms(time.time() - t_start)}")
        tx0, _ = load_histograms_from_csv(test_unenc_csv, cancel_event=cancel_event)
        if tx0 is None:
            return {"cancelled": True}
        tx1, _ = load_histograms_from_csv(test_enc_csv, cancel_event=cancel_event)
        if tx1 is None:
            return {"cancelled": True}
        X_test = _apply_scaler(np.vstack([tx0, tx1]).astype(np.float32, copy=False), scale_vec, min_vec)
        y_test = np.concatenate([np.zeros(len(tx0), dtype=np.int64), np.ones(len(tx1), dtype=np.int64)])

    # Initialize model & Adam state
    weights, biases, activation = _init_weights(arch_name, seed=seed)
    n_layers = len(weights)
    m_w = [np.zeros_like(w) for w in weights]
    v_w = [np.zeros_like(w) for w in weights]
    m_b = [np.zeros_like(b) for b in biases]
    v_b = [np.zeros_like(b) for b in biases]

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lr = float(learning_rate)
    min_lr = 1e-6
    adam_step = 0

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_val_metrics = None
    best_weights = None
    best_biases = None
    lr_wait = 0
    best_lr_loss = float("inf")
    es_wait = 0

    n_train = X_train.shape[0]
    rng = np.random.default_rng(seed)
    train_start = time.time()
    _notify(
        0,
        epochs,
        f"Training {arch_name} ({n_train} train, {len(y_val)} val)...",
        f"Elapsed: {_fmt_hms(time.time() - t_start)} | LR: {lr:.1e}",
    )

    for epoch in range(epochs):
        if cancel_event is not None and cancel_event.is_set():
            return {"cancelled": True}

        perm = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            if cancel_event is not None and cancel_event.is_set():
                return {"cancelled": True}

            idx = perm[start:start + batch_size]
            xb = X_train[idx]
            yb = y_train[idx]
            bsz = xb.shape[0]

            # Forward
            a = xb
            pre_acts = []
            acts = [a]
            for i in range(n_layers - 1):
                z = a @ weights[i] + biases[i]
                pre_acts.append(z)
                a = _activate(z, activation)
                acts.append(a)
            logits = a @ weights[-1] + biases[-1]
            probs = _softmax(logits)

            # Cross-entropy gradient
            dz = probs
            dz[np.arange(bsz), yb] -= 1.0
            dz /= float(bsz)

            # Backward + Adam update
            adam_step += 1
            lr_t = lr * math.sqrt(1.0 - (beta2 ** adam_step)) / (1.0 - (beta1 ** adam_step))

            for i in range(n_layers - 1, -1, -1):
                dw = acts[i].T @ dz
                db = np.sum(dz, axis=0)
                if i > 0:
                    da = dz @ weights[i].T
                    dz = da * _activate_grad(pre_acts[i - 1], acts[i], activation)

                m_w[i] = beta1 * m_w[i] + (1.0 - beta1) * dw
                v_w[i] = beta2 * v_w[i] + (1.0 - beta2) * (dw * dw)
                weights[i] -= lr_t * m_w[i] / (np.sqrt(v_w[i]) + eps)

                m_b[i] = beta1 * m_b[i] + (1.0 - beta1) * db
                v_b[i] = beta2 * v_b[i] + (1.0 - beta2) * (db * db)
                biases[i] -= lr_t * m_b[i] / (np.sqrt(v_b[i]) + eps)

        # Validation evaluation
        val_probs = _forward_eval(X_val, weights, biases, activation)
        val_metrics = _compute_binary_metrics(y_val, val_probs)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

        # ReduceLROnPlateau on val_loss
        if val_loss < best_lr_loss - 1e-4:
            best_lr_loss = val_loss
            lr_wait = 0
        else:
            lr_wait += 1
            if lr_wait >= 5:
                lr = max(min_lr, lr * 0.5)
                lr_wait = 0

        improved = (val_acc > best_val_acc) or (
            abs(val_acc - best_val_acc) < 1e-6 and val_loss < best_val_loss
        )
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            best_weights = [w.copy() for w in weights]
            best_biases = [b.copy() for b in biases]
            es_wait = 0
        else:
            es_wait += 1

        elapsed = time.time() - train_start
        pct = (epoch + 1) / float(epochs)
        etr = (elapsed / pct) - elapsed if pct > 0 else 0.0
        _notify(
            epoch + 1,
            epochs,
            f"Epoch {epoch + 1}/{epochs}  │  Val Acc: {val_acc * 100:.2f}% (Best: {best_val_acc * 100:.2f}%)  │  Loss: {val_loss:.4f}",
            f"Elapsed: {_fmt_hms(elapsed)} | ETR: {_fmt_hms(etr)} | LR: {lr:.1e}",
        )

        if early_stop_patience > 0 and es_wait >= early_stop_patience:
            break

    # Restore best weights
    if best_weights is not None:
        weights = best_weights
        biases = best_biases

    test_metrics = None
    if X_test is not None and y_test is not None:
        test_probs = _forward_eval(X_test, weights, biases, activation)
        test_metrics = _compute_binary_metrics(y_test, test_probs)

    saved_path = save_encod_model(
        save_model_path,
        arch_name=arch_name,
        activation=activation,
        weights=weights,
        biases=biases,
        scaler_mode=scaler_mode,
        scaler_scale=scale_vec,
        scaler_min=min_vec,
        frag_len=frag_len,
        val_metrics=best_val_metrics,
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
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "total_seconds": time.time() - t_start,
    }


# =====================================================================
# INFERENCE FUNCTIONS (USED BY DETECTION STUDIO)
# =====================================================================
def predict_encod_probs_batch(byte_block: np.ndarray, model: dict) -> np.ndarray:
    """Returns class probabilities (N, 2) where col 0=Unencrypted, col 1=Encrypted."""
    hists = extract_byte_histograms_batch(byte_block)
    x_scaled = _apply_scaler(hists, model["scaler_scale"], model["scaler_min"])
    return _forward_eval(x_scaled, model["weights"], model["biases"], model["activation"])


def predict_encod_batch(byte_block: np.ndarray, model: dict) -> np.ndarray:
    """Vectorized batch prediction: returns uint8 array of 1 (Encrypted) / 0 (Unencrypted)."""
    probs = predict_encod_probs_batch(byte_block, model)
    return np.argmax(probs, axis=1).astype(np.uint8)


def EnCoD_test(byte_data, model: dict) -> int:
    """Single-fragment EnCoD prediction: 1 = Encrypted, 0 = Unencrypted."""
    row = np.asarray(byte_data, dtype=np.uint8)
    if row.size == 0:
        return 0
    return int(predict_encod_batch(row[None, :], model)[0])
