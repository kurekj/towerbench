"""Popularity floor and EASE (Steck, 2019) closed-form linear autoencoder."""
from __future__ import annotations

import time

import numpy as np
import torch

from .base import Recommender, to_dense_rows


class Popularity(Recommender):
    name = "pop"

    def fit(self, train, train_df, val_hist, val_eval):
        self.pop = torch.as_tensor(np.asarray(train.sum(axis=0)).ravel(), device=self.device)
        return {"epochs": 0}

    def score(self, users, hist):
        return self.pop.unsqueeze(0).expand(len(users), -1).clone()


class EASE(Recommender):
    name = "ease"
    uses_gpu = True

    def fit(self, train, train_df, val_hist, val_eval):
        lam = float(self.cfg.get("lambda", 500.0))
        t0 = time.time()
        X = train.astype(np.float32)
        G = (X.T @ X).toarray()
        n = G.shape[0]
        dev = self.device if n <= int(self.cfg.get("gpu_max_items", 60000)) else torch.device("cpu")
        G = torch.as_tensor(G, device=dev)
        G.diagonal().add_(lam)
        P = torch.linalg.inv(G)
        B = -P / P.diagonal().unsqueeze(0)
        B.diagonal().zero_()
        # Catalogues too large for the GPU keep B on the CPU and score there too (a 105k-item B is
        # 44 GB in fp32); scoring is a dense matmul that MKL handles in minutes on a many-core host.
        self.B, self.device = B, dev
        del G, P
        return {"epochs": 0, "fit_seconds": time.time() - t0, "lambda": lam, "device": str(dev)}

    def score(self, users, hist):
        X = to_dense_rows(hist, users, self.device)
        return X @ self.B
