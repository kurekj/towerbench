"""LightGCN (He et al., 2020): linear neighbourhood propagation of user/item ID embeddings on the
normalised user-item bipartite graph, trained with the BPR loss and one uniform negative per event.

The propagation has no parameters, so at ranking time the graph is rebuilt from whatever history
the protocol allows (train for validation, train + validation for the single test read; see
``set_history``), exactly as the other models see validation events as history.  Literature
configuration (He et al., 2020): 64 dimensions, 3 layers, Adam 1e-3, L2 1e-4, batch 2048.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

from .base import Recommender


def _norm_adj(train: sp.csr_matrix, device) -> torch.Tensor:
    """Symmetrically normalised bipartite adjacency D^-1/2 (A) D^-1/2 as a sparse COO tensor."""
    n_u, n_i = train.shape
    R = train.tocoo()
    rows = np.concatenate([R.row, R.col + n_u])
    cols = np.concatenate([R.col + n_u, R.row])
    deg = np.bincount(rows, minlength=n_u + n_i).astype(np.float32)
    dinv = np.where(deg > 0, deg ** -0.5, 0.0).astype(np.float32)
    vals = dinv[rows] * dinv[cols]
    idx = torch.as_tensor(np.stack([rows, cols]), dtype=torch.long)
    A = torch.sparse_coo_tensor(idx, torch.as_tensor(vals), (n_u + n_i, n_u + n_i)).coalesce()
    return A.to(device)


class LightGCN(Recommender):
    name = "lightgcn"
    uses_gpu = True

    def _propagate(self, A: torch.Tensor):
        e = torch.cat([self.user_emb.weight, self.item_emb.weight], 0)
        out = [e]
        for _ in range(self.n_layers):
            e = torch.sparse.mm(A, e)
            out.append(e)
        e = torch.stack(out, 0).mean(0)  # uniform layer combination (alpha_k = 1 / (K + 1))
        return e[: self.n_users], e[self.n_users:]

    def fit(self, train, train_df, val_hist, val_eval):
        c = self.cfg
        seed = int(c.get("seed", 0))
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        d, self.n_layers = int(c.get("dim", 64)), int(c.get("layers", 3))
        self.user_emb = nn.Embedding(self.n_users, d).to(self.device)
        self.item_emb = nn.Embedding(self.n_items, d).to(self.device)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        opt = torch.optim.Adam(list(self.user_emb.parameters()) + list(self.item_emb.parameters()),
                               lr=float(c.get("lr", 1e-3)))
        reg = float(c.get("reg", 1e-4))
        bs, epochs, patience = int(c.get("batch", 2048)), int(c.get("epochs", 100)), int(c.get("patience", 8))
        train = train.tocsr()
        self.train_csr = train
        self.A = _norm_adj(train, self.device)
        coo = train.tocoo()
        pos_u, pos_i = coo.row.astype(np.int64), coo.col.astype(np.int64)
        best, best_state, bad, t0 = -1.0, None, 0, time.time()
        for ep in range(epochs):  # noqa: B007 - used after the loop
            self.training = True
            order = rng.permutation(len(pos_u))
            tot, nb = 0.0, 0
            for s in range(0, len(order), bs):
                b = order[s:s + bs]
                u, i = pos_u[b], pos_i[b]
                j = rng.integers(0, self.n_items, len(b))
                hit = np.asarray(train[u, j]).ravel() > 0  # reject sampled negatives that are positives
                if hit.any():
                    j[hit] = rng.integers(0, self.n_items, int(hit.sum()))
                u_t, i_t, j_t = (torch.as_tensor(x, device=self.device) for x in (u, i, j))
                U, V = self._propagate(self.A)
                eu, ei, ej = U[u_t], V[i_t], V[j_t]
                x = (eu * ei).sum(1) - (eu * ej).sum(1)
                loss = F.softplus(-x).mean()
                e0 = (self.user_emb(u_t).pow(2).sum() + self.item_emb(i_t).pow(2).sum()
                      + self.item_emb(j_t).pow(2).sum()) / len(b)
                loss = loss + reg * e0
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            self.training = False
            self._cache(self.A)
            v = val_eval(self)
            if v > best:
                best, bad = v, 0
                best_state = (self.user_emb.weight.detach().clone(), self.item_emb.weight.detach().clone())
            else:
                bad += 1
                if bad >= patience:
                    break
        with torch.no_grad():
            self.user_emb.weight.copy_(best_state[0])
            self.item_emb.weight.copy_(best_state[1])
        self._cache(self.A)
        return {"epochs": ep + 1, "best_val_ndcg10": best, "fit_seconds": time.time() - t0,
                "last_loss": tot / max(nb, 1), "dim": d, "layers": self.n_layers}

    @torch.no_grad()
    def _cache(self, A):
        self.U, self.V = (t.detach() for t in self._propagate(A))

    def set_history(self, df: pd.DataFrame):
        """Rebuild the propagation graph from the history allowed at ranking time (train + val)."""
        if not hasattr(self, "user_emb"):
            return
        m = sp.csr_matrix((np.ones(len(df), dtype=np.float32), (df.u.values, df.i.values)),
                          shape=(self.n_users, self.n_items))
        m.data[:] = 1.0
        self._cache(_norm_adj(m, self.device))

    @torch.no_grad()
    def score(self, users, hist):
        u = torch.as_tensor(np.asarray(users), device=self.device)
        return self.U[u] @ self.V.T
