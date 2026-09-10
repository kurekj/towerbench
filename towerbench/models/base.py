"""Model contract: fit on a history matrix, then rank the full catalog for a batch of users."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


class Recommender:
    name = "base"
    uses_gpu = False

    def __init__(self, n_users: int, n_items: int, cfg: dict, items=None, device="cuda"):
        self.n_users, self.n_items, self.cfg, self.items = n_users, n_items, cfg, items
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def fit(self, train: sp.csr_matrix, train_df, val_hist: sp.csr_matrix, val_eval) -> dict:
        """``val_eval(model) -> float`` gives validation NDCG@10 for early stopping."""
        raise NotImplementedError

    def score(self, users: np.ndarray, hist: sp.csr_matrix) -> torch.Tensor:
        """Return dense scores [len(users), n_items] on self.device."""
        raise NotImplementedError

    @torch.no_grad()
    def rank(self, users: np.ndarray, hist: sp.csr_matrix, k: int = 50, batch: int = 2048,
             mask_history: bool = True) -> np.ndarray:
        out = np.empty((len(users), k), dtype=np.int64)
        for s in range(0, len(users), batch):
            ub = users[s:s + batch]
            sc = self.score(ub, hist).float()
            if mask_history:
                h = hist[ub].tocoo()
                sc[torch.as_tensor(h.row, device=sc.device), torch.as_tensor(h.col, device=sc.device)] = -1e9
            out[s:s + batch] = torch.topk(sc, k, dim=1).indices.cpu().numpy()
        return out


def to_dense_rows(m: sp.csr_matrix, rows: np.ndarray, device) -> torch.Tensor:
    sub = m[rows].tocoo()
    t = torch.zeros((len(rows), m.shape[1]), device=device)
    t[torch.as_tensor(sub.row, device=device), torch.as_tensor(sub.col, device=device)] = 1.0
    return t
