import math
import time
import numpy as np
import pandas as pd

_SELU_ALPHA = 1.6732632423543772848170429916717
_SELU_SCALE = 1.0507009873554804934193349852946


def fmt_hms(seconds: float) -> str:
    """Formats seconds into HH:MM:SS."""
    if seconds is None or seconds < 0:
        return "--:--:--"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# =====================================================================
# CSV HEADER DETECTION & STREAMING UINT8 CHUNK LOADER
# =====================================================================
def detect_header_and_frag_width(file_path: str):
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


def fast_count_lines(file_path: str) -> int:
    """Fast binary buffered newline counter for large CSV files (excludes detected header)."""
    skip_rows, _ = detect_header_and_frag_width(file_path)
    count = 0
    with open(file_path, "rb") as f:
        while True:
            buf = f.read(1048576)
            if not buf:
                break
            count += buf.count(b"\n")
    return max(1, count - skip_rows)


def chunk_df_to_uint8(chunk_df: pd.DataFrame) -> np.ndarray:
    arr = chunk_df.to_numpy()
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, 255).astype(np.uint8, copy=False)
    if np.issubdtype(arr.dtype, np.floating):
        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=0.0)
        return np.rint(np.clip(arr, 0.0, 255.0)).astype(np.uint8)
    num_df = chunk_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return np.rint(np.clip(num_df.to_numpy(dtype=np.float64), 0.0, 255.0)).astype(np.uint8)


def iter_uint8_csv_chunks(file_path: str, chunk_size: int = 2000):
    """Yields (uint8_block, total_rows_estimated) safely across header/float CSVs."""
    skip_rows, data_cols = detect_header_and_frag_width(file_path)
    if data_cols <= 0:
        raise ValueError(f"Empty or invalid CSV file: {file_path}")
    total_rows = fast_count_lines(file_path)
    usecols = list(range(data_cols))

    for chunk_df in pd.read_csv(
        file_path,
        header=None,
        skiprows=skip_rows,
        usecols=usecols,
        chunksize=chunk_size,
        engine="c",
    ):
        if chunk_df.empty:
            continue
        yield chunk_df_to_uint8(chunk_df), total_rows


# =====================================================================
# SCALING, SPLITTING & METRICS
# =====================================================================
def fit_scaler(X_train: np.ndarray, mode: str):
    """
    Fits an affine scaler strictly on X_train: X_scaled = X * scale_vec + min_vec.
      - 'minmax':   [0, 2] per-feature MinMaxScaler
      - 'fixed':    2.0 * X (fixed [0, 1] -> [0, 2])
      - 'standard': (X - mean) / std (zero mean, unit variance)
    """
    n_feats = X_train.shape[1]
    if mode == "minmax":
        data_min = X_train.min(axis=0).astype(np.float32)
        data_max = X_train.max(axis=0).astype(np.float32)
        data_range = data_max - data_min
        data_range[data_range == 0.0] = 1.0
        scale_vec = (2.0 / data_range).astype(np.float32)
        min_vec = (-data_min * scale_vec).astype(np.float32)
    elif mode == "fixed":
        scale_vec = np.full(n_feats, 2.0, dtype=np.float32)
        min_vec = np.zeros(n_feats, dtype=np.float32)
    elif mode == "standard":
        mean_vec = X_train.mean(axis=0, dtype=np.float64)
        std_vec = X_train.std(axis=0, dtype=np.float64)
        std_vec[std_vec == 0.0] = 1.0
        scale_vec = (1.0 / std_vec).astype(np.float32)
        min_vec = (-mean_vec / std_vec).astype(np.float32)
    else:
        raise ValueError(f"Unsupported scaler mode: {mode}")
    return scale_vec, min_vec


def apply_scaler(X: np.ndarray, scale_vec: np.ndarray, min_vec: np.ndarray) -> np.ndarray:
    return (X * scale_vec + min_vec).astype(np.float32, copy=False)


def stratified_split(X: np.ndarray, y: np.ndarray, val_ratio: float, seed: int = 123):
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


def activate(z: np.ndarray, kind: str) -> np.ndarray:
    dt = z.dtype if z.dtype in (np.float32, np.float64) else np.float32
    if kind == "selu":
        scale = dt.type(_SELU_SCALE)
        alpha = dt.type(_SELU_ALPHA)
        z_neg = np.minimum(z, dt.type(0.0))
        return np.where(z > 0.0, scale * z, scale * alpha * (np.exp(z_neg) - dt.type(1.0))).astype(dt, copy=False)
    if kind == "relu":
        return np.maximum(dt.type(0.0), z)
    raise ValueError(f"Unsupported activation: {kind}")


def activate_grad(z: np.ndarray, a: np.ndarray, kind: str) -> np.ndarray:
    dt = z.dtype if z.dtype in (np.float32, np.float64) else np.float32
    if kind == "selu":
        scale = dt.type(_SELU_SCALE)
        sa = dt.type(_SELU_SCALE * _SELU_ALPHA)
        return np.where(z > 0.0, scale, a + sa)
    if kind == "relu":
        return (z > 0.0).astype(dt)
    raise ValueError(f"Unsupported activation: {kind}")


def softmax(logits: np.ndarray) -> np.ndarray:
    dt = logits.dtype if logits.dtype in (np.float32, np.float64) else np.float32
    z_shift = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return (exp_z / np.sum(exp_z, axis=1, keepdims=True)).astype(dt, copy=False)


def compute_binary_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
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
# MODULAR PURE-NUMPY NEURAL NETWORK LAYERS (PyTorch state_dict Parity)
# =====================================================================
class LinearLayer:
    def __init__(self, in_dim: int, out_dim: int, init_kind: str = "he_normal", name: str = "", rng=None):
        self.name = name
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        rng = rng or np.random.default_rng(123)
        if init_kind == "lecun_normal":
            std = math.sqrt(1.0 / float(in_dim))
            self.W = rng.normal(0.0, std, size=(in_dim, out_dim)).astype(np.float32)
        elif init_kind == "he_normal":
            std = math.sqrt(2.0 / float(in_dim))
            self.W = rng.normal(0.0, std, size=(in_dim, out_dim)).astype(np.float32)
        elif init_kind == "glorot_uniform":
            lim = math.sqrt(6.0 / float(in_dim + out_dim))
            self.W = rng.uniform(-lim, lim, size=(in_dim, out_dim)).astype(np.float32)
        else:
            raise ValueError(f"Unknown init_kind: {init_kind}")
        self.b = np.zeros(out_dim, dtype=np.float32)
        self._x = None
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self._x = x
        return x @ self.W + self.b

    def backward(self, dz: np.ndarray) -> np.ndarray:
        self.dW = self._x.T @ dz
        self.db = np.sum(dz, axis=0)
        return dz @ self.W.T

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]

    def export_state(self) -> dict:
        return {
            f"{self.name}.weight": self.W.T.astype(np.float32),
            f"{self.name}.bias": self.b.astype(np.float32),
        }

    def load_state(self, state: dict):
        self.W = np.asarray(state[f"{self.name}.weight"], dtype=np.float32).T.copy()
        self.b = np.asarray(state[f"{self.name}.bias"], dtype=np.float32).copy()


class BatchNorm1dLayer:
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, name: str = ""):
        self.name = name
        self.dim = int(dim)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)
        self.running_mean = np.zeros(dim, dtype=np.float32)
        self.running_var = np.ones(dim, dtype=np.float32)
        self._x_norm = None
        self._inv_std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        dt = x.dtype if x.dtype in (np.float32, np.float64) else np.float32
        if not training:
            inv_std = (1.0 / np.sqrt(self.running_var.astype(dt) + dt.type(self.eps))).astype(dt)
            x_norm = (x - self.running_mean.astype(dt)) * inv_std
            return (self.gamma.astype(dt) * x_norm + self.beta.astype(dt)).astype(dt, copy=False)

        b = x.shape[0]
        mean = np.mean(x, axis=0, dtype=dt)
        x_c = x - mean
        var = np.mean(x_c * x_c, axis=0, dtype=dt)
        inv_std = (1.0 / np.sqrt(var + dt.type(self.eps))).astype(dt)
        x_norm = x_c * inv_std

        self._x_norm = x_norm
        self._inv_std = inv_std

        unbiased_var = var * (float(b) / float(max(1, b - 1)))
        self.running_mean = ((1.0 - self.momentum) * self.running_mean + self.momentum * mean).astype(np.float32)
        self.running_var = ((1.0 - self.momentum) * self.running_var + self.momentum * unbiased_var).astype(np.float32)

        return (self.gamma.astype(dt) * x_norm + self.beta.astype(dt)).astype(dt, copy=False)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dt = dy.dtype if dy.dtype in (np.float32, np.float64) else np.float32
        b = dy.shape[0]
        self.dgamma = np.sum(dy * self._x_norm, axis=0)
        self.dbeta = np.sum(dy, axis=0)
        dx_norm = dy * self.gamma.astype(dt)
        sum_dx = np.sum(dx_norm, axis=0)
        sum_dx_x = np.sum(dx_norm * self._x_norm, axis=0)
        return ((self._inv_std / dt.type(b)) * (dt.type(b) * dx_norm - sum_dx - self._x_norm * sum_dx_x)).astype(dt, copy=False)

    def params_and_grads(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]

    def export_state(self) -> dict:
        return {
            f"{self.name}.weight": self.gamma.astype(np.float32),
            f"{self.name}.bias": self.beta.astype(np.float32),
            f"{self.name}.running_mean": self.running_mean.astype(np.float32),
            f"{self.name}.running_var": self.running_var.astype(np.float32),
        }

    def load_state(self, state: dict):
        self.gamma = np.asarray(state[f"{self.name}.weight"], dtype=np.float32).copy()
        self.beta = np.asarray(state[f"{self.name}.bias"], dtype=np.float32).copy()
        self.running_mean = np.asarray(state[f"{self.name}.running_mean"], dtype=np.float32).copy()
        self.running_var = np.asarray(state[f"{self.name}.running_var"], dtype=np.float32).copy()


class DropoutLayer:
    def __init__(self, p: float = 0.3, rng=None):
        self.p = float(p)
        self.rng = rng or np.random.default_rng(123)
        self._mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        dt = x.dtype if x.dtype in (np.float32, np.float64) else np.float32
        self._mask = ((self.rng.random(x.shape) < keep).astype(dt) / dt.type(keep))
        return x * self._mask

    def backward(self, dy: np.ndarray) -> np.ndarray:
        if self.p <= 0.0 or self._mask is None:
            return dy
        return dy * self._mask


class ActivationLayer:
    def __init__(self, kind: str = "relu"):
        self.kind = kind
        self._z = None
        self._a = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        a = activate(x, self.kind)
        if training:
            self._z = x
            self._a = a
        return a

    def backward(self, da: np.ndarray) -> np.ndarray:
        return da * activate_grad(self._z, self._a, self.kind)


class SequentialMLP:
    def __init__(self, layers):
        self.layers = list(layers)
        self.has_batchnorm = any(isinstance(ly, BatchNorm1dLayer) for ly in self.layers)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        out = x
        for ly in self.layers:
            out = ly.forward(out, training=training)
        return out

    def backward(self, dz: np.ndarray) -> np.ndarray:
        grad = dz
        for ly in reversed(self.layers):
            grad = ly.backward(grad)
        return grad

    def eval_probs(self, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        n = X.shape[0]
        if n == 0:
            return np.zeros((0, 2), dtype=np.float32)
        out_dim = next(ly.out_dim for ly in reversed(self.layers) if isinstance(ly, LinearLayer))
        probs = np.empty((n, out_dim), dtype=np.float32)
        for start in range(0, n, batch_size):
            logits = self.forward(X[start:start + batch_size], training=False)
            probs[start:start + batch_size] = softmax(logits)
        return probs

    def export_state(self) -> dict:
        state = {}
        for ly in self.layers:
            if hasattr(ly, "export_state"):
                state.update(ly.export_state())
        return state

    def load_state(self, state: dict):
        for ly in self.layers:
            if hasattr(ly, "load_state"):
                ly.load_state(state)

    def _clone_state(self) -> dict:
        return {k: v.copy() for k, v in self.export_state().items()}

    def train_loop(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 60,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        early_stop_patience: int = 15,
        seed: int = 123,
        cancel_event=None,
        progress_cb=None,
    ) -> dict:
        param_refs = []
        for ly in self.layers:
            if hasattr(ly, "params_and_grads"):
                for p, _ in ly.params_and_grads():
                    param_refs.append(p)
        m_states = [np.zeros_like(p) for p in param_refs]
        v_states = [np.zeros_like(p) for p in param_refs]

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr = float(learning_rate)
        min_lr = 1e-6
        adam_step = 0

        best_val_acc = -1.0
        best_val_loss = float("inf")
        best_val_metrics = None
        best_state = None
        lr_wait = 0
        best_lr_loss = float("inf")
        es_wait = 0

        n_train = X_train.shape[0]
        rng = np.random.default_rng(seed)
        train_start = time.time()

        for epoch in range(epochs):
            if cancel_event is not None and cancel_event.is_set():
                return {"cancelled": True}

            perm = rng.permutation(n_train)
            for start in range(0, n_train, batch_size):
                if cancel_event is not None and cancel_event.is_set():
                    return {"cancelled": True}

                idx = perm[start:start + batch_size]
                bsz = idx.shape[0]
                if self.has_batchnorm and bsz < 2:
                    continue

                xb = X_train[idx]
                yb = y_train[idx]
                logits = self.forward(xb, training=True)
                probs = softmax(logits)

                dz = probs
                dz[np.arange(bsz), yb] -= 1.0
                dz /= float(bsz)
                self.backward(dz)

                adam_step += 1
                lr_t = lr * math.sqrt(1.0 - (beta2 ** adam_step)) / (1.0 - (beta1 ** adam_step))

                p_idx = 0
                for ly in self.layers:
                    if hasattr(ly, "params_and_grads"):
                        for p, g in ly.params_and_grads():
                            m_states[p_idx] = beta1 * m_states[p_idx] + (1.0 - beta1) * g
                            v_states[p_idx] = beta2 * v_states[p_idx] + (1.0 - beta2) * (g * g)
                            p -= lr_t * m_states[p_idx] / (np.sqrt(v_states[p_idx]) + eps)
                            p_idx += 1

            val_probs = self.eval_probs(X_val)
            val_metrics = compute_binary_metrics(y_val, val_probs)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]

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
                best_state = self._clone_state()
                es_wait = 0
            else:
                es_wait += 1

            elapsed = time.time() - train_start
            pct = (epoch + 1) / float(epochs)
            etr = (elapsed / pct) - elapsed if pct > 0 else 0.0
            if progress_cb is not None:
                progress_cb(
                    epoch + 1,
                    epochs,
                    f"Epoch {epoch + 1}/{epochs}  │  Val Acc: {val_acc * 100:.2f}% (Best: {best_val_acc * 100:.2f}%)  │  Loss: {val_loss:.4f}",
                    f"Elapsed: {fmt_hms(elapsed)} | ETR: {fmt_hms(etr)} | LR: {lr:.1e}",
                )

            if early_stop_patience > 0 and es_wait >= early_stop_patience:
                break

        if best_state is not None:
            self.load_state(best_state)

        return {"cancelled": False, "val_metrics": best_val_metrics}
