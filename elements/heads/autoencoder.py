"""A novelty detector whose classical capacity can be scaled: an autoencoder.

``NoveltyDetector`` is linear -- a PCA subspace plus Mahalanobis distance. This
is its nonlinear counterpart for the joint scaling study: learn to reconstruct
NORMAL rows only, then score every row by how badly it reconstructs.

The bottleneck is capped at a third of the input width. On a 12-dim dials
vector a 32-unit bottleneck could simply learn the identity, reconstruct
everything perfectly, and flag nothing.
"""

from __future__ import annotations

import time

import numpy as np
import torch


from elements.heads.mlp import ACTIVATIONS


def build_autoencoder(d: int, hidden: tuple[int, ...], bottleneck: int,
                      activation: str = "gelu"):
    act = ACTIVATIONS[activation]
    enc: list[torch.nn.Module] = []
    width = d
    for h in hidden:
        enc += [torch.nn.Linear(width, h), act()]
        width = h
    enc.append(torch.nn.Linear(width, bottleneck))
    dec: list[torch.nn.Module] = []
    width = bottleneck
    for h in reversed(hidden):
        dec += [torch.nn.Linear(width, h), act()]
        width = h
    dec.append(torch.nn.Linear(width, d))
    return torch.nn.Sequential(*enc, *dec)


class AEDetector:
    """Autoencoder trained on normal data; score = per-row reconstruction MSE.

    Early stopping on the reconstruction loss of a held-out NORMAL split, so
    no attack ever influences training or model selection.
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (512,),
        bottleneck: int = 32,
        activation: str = "gelu",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 512,
        max_epochs: int = 100,
        patience: int = 10,
        device: torch.device | str = "cpu",
        seed: int = 42,
    ) -> None:
        self.hidden = tuple(hidden)
        self.bottleneck = bottleneck
        self.activation = activation
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = torch.device(device)
        self.seed = seed

    def fit(self, X: np.ndarray, X_val: np.ndarray) -> "AEDetector":
        t0 = time.time()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.bottleneck_ = max(2, min(self.bottleneck, d // 3))
        self.model = build_autoencoder(d, self.hidden, self.bottleneck_, self.activation).to(self.device)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=max(1, self.patience // 3)
        )
        Xg = torch.from_numpy(X).to(self.device, dtype=torch.float16)  # normal rows only

        best, best_state, stale, self.epochs_run = np.inf, None, 0, 0
        for _ in range(self.max_epochs):
            self.model.train()
            perm = torch.from_numpy(rng.permutation(n)).to(self.device)
            for i in range(0, n, self.batch_size):
                xb = Xg[perm[i : i + self.batch_size]].float()
                loss = ((self.model(xb) - xb) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
            val = float(self.score(X_val).mean())
            sched.step(val)
            self.epochs_run += 1
            if val < best * (1 - 1e-4):
                best, stale = val, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        del Xg
        self.model.load_state_dict(best_state)
        self.model.eval()
        self.best_val_loss = best
        self.fit_time_s = time.time() - t0
        return self

    def score(self, X: np.ndarray, batch: int = 8192) -> np.ndarray:
        """Higher means more anomalous."""
        was_training = self.model.training
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.from_numpy(np.ascontiguousarray(X[i : i + batch])).to(self.device)
                outs.append(((self.model(xb) - xb) ** 2).mean(1).cpu())
        self.model.train(was_training)
        return torch.cat(outs).numpy()
