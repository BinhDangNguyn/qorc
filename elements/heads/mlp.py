"""A readout whose classical capacity can be scaled: an MLP on the reservoir.

``SoftmaxHead`` is a single linear layer, so a capacity sweep that only grows
the reservoir asks one linear map to exploit ever more features. This head
lets the classical side grow too. ``hidden=()`` is the linear case, trained by
the same optimiser as every MLP so that depth and width are the only things
that change along the classical axis.

Training uses a validation split carved out of TRAIN (never the test rows) for
early stopping and checkpoint selection, with the same class-balanced loss as
``SoftmaxHead``.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import f1_score


ACTIVATIONS = {"gelu": torch.nn.GELU, "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}


def build_mlp(d_in: int, n_classes: int, hidden: tuple[int, ...], dropout: float,
              activation: str = "gelu"):
    layers: list[torch.nn.Module] = []
    width = d_in
    for h in hidden:
        layers += [torch.nn.Linear(width, h), ACTIVATIONS[activation](),
                   torch.nn.Dropout(dropout)]
        width = h
    layers.append(torch.nn.Linear(width, n_classes))
    return torch.nn.Sequential(*layers)


def n_params(d_in: int, n_classes: int, hidden: tuple[int, ...]) -> int:
    """Trainable parameters of ``build_mlp`` without building it."""
    total, width = 0, d_in
    for h in (*hidden, n_classes):
        total += width * h + h
        width = h
    return total


class MLPHead:
    """Class-balanced MLP classifier with early stopping on validation macro-F1.

    Parameters
    ----------
    hidden:
        Hidden-layer widths. ``()`` is multinomial logistic regression.
    dropout:
        Applied after every hidden activation.
    weight_decay:
        AdamW decoupled weight decay -- the L2 analogue.
    patience:
        Epochs without a validation macro-F1 improvement before stopping. The
        weights from the best epoch are restored.
    """

    def __init__(
        self,
        n_classes: int,
        hidden: tuple[int, ...] = (),
        dropout: float = 0.1,
        activation: str = "gelu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        max_epochs: int = 100,
        patience: int = 12,
        class_weight: str | None = "balanced",
        device: torch.device | str = "cpu",
        seed: int = 42,
        gpu_resident_gb: float = 3.5,
    ) -> None:
        self.n_classes = n_classes
        self.hidden = tuple(hidden)
        self.dropout = dropout
        self.activation = activation
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.class_weight = class_weight
        self.device = torch.device(device)
        self.seed = seed
        self.gpu_resident_gb = gpu_resident_gb

    def _batches(self, X: np.ndarray, y: np.ndarray, rng):
        """Yield shuffled minibatches, gathering on the GPU when X fits there.

        Resident X is stored as float16 and every batch is cast back to
        float32, whatever the width -- one storage precision for every config,
        so the reservoir sizes stay comparable. Random-row gathers from host
        RAM were the bottleneck at 10k dims, not the matmuls.
        """
        n = len(X)
        if self._resident is None:
            gb = X.shape[0] * X.shape[1] * 2 / 1e9
            if self.device.type == "cuda" and gb <= self.gpu_resident_gb:
                self._resident = (
                    torch.from_numpy(X).to(self.device, dtype=torch.float16),
                    torch.from_numpy(y).to(self.device),
                )
            else:
                self._resident = False
        perm = rng.permutation(n)
        for i in range(0, n, self.batch_size):
            idx = np.sort(perm[i : i + self.batch_size])  # sorted = faster gather
            if self._resident:
                ti = torch.from_numpy(idx).to(self.device)
                yield self._resident[0][ti].float(), self._resident[1][ti]
            else:
                xb = torch.from_numpy(X[idx].astype(np.float16)).to(self.device)
                yield xb.float(), torch.from_numpy(y[idx]).to(self.device)

    def fit(
        self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> "MLPHead":
        t0 = time.time()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape

        counts = np.bincount(y, minlength=self.n_classes).astype(np.float32)
        if self.class_weight == "balanced":
            w = np.where(counts > 0, n / (self.n_classes * np.maximum(counts, 1)), 0.0)
        else:
            w = np.ones(self.n_classes)
        lossf = torch.nn.CrossEntropyLoss(
            weight=torch.as_tensor(w, dtype=torch.float32, device=self.device)
        )

        self.model = build_mlp(d, self.n_classes, self.hidden, self.dropout, self.activation).to(self.device)
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=max(1, self.patience // 3)
        )

        best, best_state, stale = -1.0, None, 0
        self.history: list[dict] = []
        self._resident = None
        for epoch in range(self.max_epochs):
            self.model.train()
            total = torch.zeros((), device=self.device)
            for xb, yb in self._batches(X, y, rng):
                loss = lossf(self.model(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                total += loss.detach() * len(yb)
            total = float(total)

            val_f1 = float(
                f1_score(y_val, self.predict(X_val), average="macro", zero_division=0)
            )
            sched.step(val_f1)
            self.history.append(
                {"epoch": epoch, "train_loss": total / n, "val_macro_f1": val_f1}
            )
            if val_f1 > best + 1e-4:
                best, stale = val_f1, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break

        self._resident = None  # release the GPU copy before predicting
        self.model.load_state_dict(best_state)
        self.model.eval()
        self.best_val_macro_f1 = best
        self.epochs_run = len(self.history)
        self.n_params = n_params(d, self.n_classes, self.hidden)
        self.fit_time_s = time.time() - t0
        return self

    def _logits(self, X: np.ndarray, batch: int = 8192) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.from_numpy(np.ascontiguousarray(X[i : i + batch])).to(self.device)
                outs.append(self.model(xb).float().cpu())
        self.model.train(was_training)
        return torch.cat(outs, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._logits(X).argmax(1).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return torch.softmax(self._logits(X), dim=1).numpy()
