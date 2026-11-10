"""Squeeze-Attention Deep Morphological Classifier (synopsis section 4.6).

The fused feature vector F_C is treated as a 1-D signal:
    multi-scale convolutions (kernel 1/3/5) -> concatenate
    -> Squeeze-and-Excitation channel attention -> fully connected -> softmax.
"""
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn


class SADMC(nn.Module):
    def __init__(self, n_features, n_classes, channels=16, reduction=4, dropout=0.3):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(nn.Conv1d(1, channels, k, padding=k // 2),
                          nn.BatchNorm1d(channels), nn.ReLU())
            for k in (1, 3, 5))
        c = channels * 3
        # Excitation: s = sigmoid(W2 * ReLU(W1 * z))
        self.excite = nn.Sequential(nn.Linear(c, c // reduction), nn.ReLU(),
                                    nn.Linear(c // reduction, c), nn.Sigmoid())
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(c * n_features, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, n_classes))

    def attention(self, x):
        f = torch.cat([b(x.unsqueeze(1)) for b in self.branches], dim=1)  # B, C, F
        z = f.mean(dim=2)                                                 # squeeze
        return f, self.excite(z)

    def forward(self, x):
        f, s = self.attention(x)
        return self.head(f * s.unsqueeze(2))                              # recalibrate


class SADMCClassifier(ClassifierMixin, BaseEstimator):
    """scikit-learn wrapper so SADMC shares the cross-validation code path."""

    def __init__(self, epochs=100, lr=1e-3, weight_decay=1e-4, batch_size=16, seed=0):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X, y):
        # The network is tiny; intra-op threading only adds overhead here.
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return self._fit(X, y)
        finally:
            torch.set_num_threads(threads)

    def _fit(self, X, y):
        torch.manual_seed(self.seed)
        self.classes_ = np.unique(y)
        idx = np.searchsorted(self.classes_, y)
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        t = torch.tensor(idx, dtype=torch.long)
        self.model_ = SADMC(X.shape[1], len(self.classes_))
        counts = np.bincount(idx, minlength=len(self.classes_))
        weight = torch.tensor(len(idx) / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        gen = torch.Generator().manual_seed(self.seed)
        self.model_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(X), generator=gen)
            for i in range(0, len(X), self.batch_size):
                b = perm[i:i + self.batch_size]
                if len(b) < 2:  # BatchNorm needs more than one sample
                    continue
                opt.zero_grad()
                loss_fn(self.model_(X[b]), t[b]).backward()
                opt.step()
        self.model_.eval()
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            logits = self.model_(torch.tensor(np.asarray(X), dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
