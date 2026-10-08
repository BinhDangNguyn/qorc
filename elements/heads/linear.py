"""The supervised readout: multinomial logistic regression on the reservoir.

Equivalent in form to sklearn's ``LogisticRegression(multi_class='multinomial')``
with L2 regularisation, but implemented in torch so a 10,626-dimensional
readout fits in seconds rather than minutes.
"""

from __future__ import annotations

import numpy as np
import torch


def _objective_at(Xt, yt, W, b, weight, l2, wsum, resident, chunk, device) -> float:
    """Evaluate the training objective at the given weights, without grad.

    Needed because ``LBFGS.step()`` returns the loss from the FIRST closure
    call -- i.e. at W=0, which is just log(n_classes) -- not the converged
    value. Recording that would misreport every run.
    """
    lossf = torch.nn.CrossEntropyLoss(weight=weight, reduction="sum")
    if resident:
        total = float(lossf(Xt @ W + b, yt)) / wsum
    else:
        total = 0.0
        for i in range(0, Xt.shape[0], chunk):
            xb = Xt[i : i + chunk].to(device, non_blocking=True)
            total += float(lossf(xb @ W + b, yt[i : i + chunk]))
            del xb
        total /= wsum
    return total + l2 * float((W * W).sum())


class SoftmaxHead:
    """L-BFGS multinomial logistic regression, GPU-capable.

    Parameters
    ----------
    n_classes:
        Number of target classes.
    l2:
        Ridge penalty on the weights.
    max_iter:
        L-BFGS iterations.
    class_weight:
        ``"balanced"`` (default) reweights by inverse class frequency, which
        matters enormously here: ``normal`` and ``neptune`` are most of the
        data and unweighted training simply ignores the rare attacks.
    gpu_resident_gb:
        Keep the design matrix on the GPU only when it is at most this many
        GB. The default of 0.0 means ALWAYS chunk.

        That default is deliberate. The two paths compute the same objective,
        but L-BFGS is path-sensitive -- a 1e-7 gradient difference is enough to
        land on a slightly different point within ``max_iter``. Using one code
        path for every reservoir size keeps the configs exactly comparable,
        which is the entire point of a capacity sweep. Raise it only to trade
        that comparability for speed.
    """

    def __init__(
        self,
        n_classes: int,
        l2: float = 1e-4,
        max_iter: int = 200,
        class_weight: str | None = "balanced",
        device: torch.device | str = "cpu",
        chunk: int = 16384,
        gpu_resident_gb: float = 0.0,
    ) -> None:
        self.n_classes = n_classes
        self.l2 = l2
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.device = torch.device(device)
        self.chunk = chunk
        self.gpu_resident_gb = gpu_resident_gb

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxHead":
        n, d = X.shape
        gb = n * d * 4 / 1e9
        resident = self.device.type == "cuda" and gb <= self.gpu_resident_gb

        yt = torch.as_tensor(y, dtype=torch.long, device=self.device)
        if self.class_weight == "balanced":
            counts = torch.bincount(yt, minlength=self.n_classes).float()
            weight = torch.where(
                counts > 0,
                n / (self.n_classes * counts.clamp(min=1)),
                torch.zeros_like(counts),
            )
        else:
            weight = torch.ones(self.n_classes, device=self.device)

        Xt = (
            torch.as_tensor(X, dtype=torch.float32, device=self.device)
            if resident
            else torch.from_numpy(np.ascontiguousarray(X))
        )
        self.W = torch.zeros(d, self.n_classes, device=self.device, requires_grad=True)
        self.b = torch.zeros(self.n_classes, device=self.device, requires_grad=True)
        lossf = torch.nn.CrossEntropyLoss(weight=weight, reduction="sum")
        opt = torch.optim.LBFGS(
            [self.W, self.b],
            max_iter=self.max_iter,
            history_size=10,
            tolerance_grad=1e-7,
            line_search_fn="strong_wolfe",
        )
        wsum = float(weight[yt].sum())  # normaliser matching reduction="mean"

        def closure():
            opt.zero_grad()
            if resident:
                loss = lossf(Xt @ self.W + self.b, yt) / wsum
                loss = loss + self.l2 * (self.W * self.W).sum()
                loss.backward()
                return loss
            # Chunked: accumulate the gradient over slices so peak memory is
            # governed by `chunk`, not by the full design matrix.
            total = 0.0
            for i in range(0, n, self.chunk):
                xb = Xt[i : i + self.chunk].to(self.device, non_blocking=True)
                part = lossf(xb @ self.W + self.b, yt[i : i + self.chunk]) / wsum
                part.backward()
                total += part.detach().item()
                del xb, part
            reg = self.l2 * (self.W * self.W).sum()
            reg.backward()
            return torch.tensor(total + reg.detach().item(), device=self.device)

        opt.step(closure)
        with torch.no_grad():
            self.final_loss = _objective_at(
                Xt, yt, self.W, self.b, weight, self.l2,
                wsum, resident, self.chunk, self.device,
            )
        self.W = self.W.detach()
        self.b = self.b.detach()
        del Xt, yt
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def _logits(self, X: np.ndarray, batch: int = 16384) -> torch.Tensor:
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.as_tensor(
                    X[i : i + batch], dtype=torch.float32, device=self.device
                )
                outs.append((xb @ self.W + self.b).cpu())
        return torch.cat(outs, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._logits(X).argmax(1).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return torch.softmax(self._logits(X), dim=1).numpy()
