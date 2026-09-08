"""
A small neural network with Monte-Carlo dropout, wrapped for scikit-learn.

Why a network at all on 1,686 rows: not because it is expected to win, but
because the uncertainty it gives is worth having. Dropout left switched on at
inference turns the network into an approximate Bayesian model (Gal and
Ghahramani, 2016), so each authority gets a distribution over its risk rather
than a point estimate. For a service deciding whether to act on a score, "0.42
give or take 0.05" and "0.42 give or take 0.25" are different situations.

The architecture is deliberately small and heavily regularised. With 83
features and roughly 1,100 training rows, anything larger memorises the
training years.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn


class _Net(nn.Module):
    """83 to 64 to 32 to 1, with dropout between every layer."""

    def __init__(self, n_features: int, hidden: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for units in hidden:
            layers += [nn.Linear(width, units), nn.ReLU(), nn.Dropout(dropout)]
            width = units
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MCDropoutMLP(BaseEstimator, ClassifierMixin):
    """Binary classifier with Monte-Carlo dropout uncertainty.

    Implements enough of the scikit-learn estimator interface to sit inside a
    Pipeline alongside the other models and be scored by the same code.

    Parameters mirror the decisions that matter on a dataset this size:
    `hidden` and `dropout` control capacity, `patience` stops training when the
    held-out slice stops improving, and `mc_samples` sets how many stochastic
    forward passes the uncertainty estimate averages over.
    """

    def __init__(self, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.3,
                 lr: float = 3e-3, weight_decay: float = 1e-3, max_epochs: int = 400,
                 patience: int = 40, batch_size: int = 128, mc_samples: int = 100,
                 val_fraction: float = 0.2, random_state: int = 0) -> None:
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.mc_samples = mc_samples
        self.val_fraction = val_fraction
        self.random_state = random_state

    def fit(self, X, y):  # noqa: N803 - scikit-learn's signature
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        x = torch.as_tensor(np.asarray(X, dtype=np.float32))
        target = torch.as_tensor(np.asarray(y, dtype=np.float32))
        self.classes_ = np.array([0, 1])

        # A held-out slice for early stopping, taken at random from the
        # training years only. The test year is never involved.
        n = len(x)
        order = rng.permutation(n)
        cut = max(1, int(n * self.val_fraction))
        val_idx, fit_idx = order[:cut], order[cut:]

        self.model_ = _Net(x.shape[1], tuple(self.hidden), self.dropout)
        optimiser = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        # The positive class is small, so the loss is reweighted rather than
        # the data resampled: resampling would distort the calibration.
        positives = float(target[fit_idx].sum())
        pos_weight = torch.tensor(
            max(1.0, (len(fit_idx) - positives) / max(positives, 1.0)))
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best, best_state, waited = float("inf"), None, 0
        for _ in range(self.max_epochs):
            self.model_.train()
            batches = torch.split(torch.as_tensor(rng.permutation(fit_idx)),
                                  self.batch_size)
            for batch in batches:
                optimiser.zero_grad()
                loss = criterion(self.model_(x[batch]), target[batch])
                loss.backward()
                optimiser.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = criterion(self.model_(x[val_idx]), target[val_idx]).item()
            if val_loss < best - 1e-5:
                best, waited = val_loss, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def _mc_forward(self, X) -> np.ndarray:  # noqa: N803
        """Stochastic forward passes with dropout deliberately left on."""
        x = torch.as_tensor(np.asarray(X, dtype=np.float32))
        self.model_.train()  # keeps dropout active, which is the whole point
        torch.manual_seed(self.random_state + 1)
        with torch.no_grad():
            draws = [torch.sigmoid(self.model_(x)).numpy()
                     for _ in range(self.mc_samples)]
        return np.stack(draws)

    def predict_proba(self, X):  # noqa: N803
        draws = self._mc_forward(X)
        mean = draws.mean(axis=0)
        return np.column_stack([1.0 - mean, mean])

    def predict(self, X):  # noqa: N803
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_uncertainty(self, X):  # noqa: N803
        """Standard deviation across the Monte-Carlo draws, per row.

        This is what makes the network worth including. It separates "the model
        is confident this authority is fine" from "the model has no idea",
        which a ranked list alone cannot express.
        """
        return self._mc_forward(X).std(axis=0)
