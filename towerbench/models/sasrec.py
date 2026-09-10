"""SASRec (Kang & McAuley, 2018): causal self-attention over the user's item sequence, next-item
prediction with one uniformly sampled negative per position (binary cross-entropy), full-catalogue
scoring with the shared item embedding table.  Reuses the sequence helpers of the two-tower model."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .two_tower import TwoTower


class _SASRec(nn.Module):
    def __init__(self, n_items, d, max_len, n_layers, n_heads, dropout):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=d, dropout=dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.max_len, self.n_heads = max_len, n_heads
        # PyTorch's unit-variance embedding init combined with the sqrt(d) input scaling saturates
        # the logits and the model never leaves the popularity solution; use the original small init.
        nn.init.normal_(self.item.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def forward(self, seq):  # seq [B, L] 1-based ids, 0 = pad (left-padded)
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        h = self.drop(self.item(seq) * (self.item.embedding_dim ** 0.5) + self.pos(pos))
        # blocked = future positions or padded keys; self-attention always allowed so no row is fully masked
        causal = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1)
        blocked = (causal.unsqueeze(0) | (seq == 0).unsqueeze(1)) & ~torch.eye(L, device=seq.device, dtype=torch.bool)
        h = self.blocks(h, mask=blocked.repeat_interleave(self.n_heads, dim=0))
        return self.norm(h)


class SASRec(TwoTower):
    name = "sasrec"
    uses_gpu = True

    def fit(self, train, train_df, val_hist, val_eval):
        c = self.cfg
        seed = int(c.get("seed", 0))
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        d, self.max_len = int(c.get("dim", 64)), int(c.get("max_len", 50))
        L = self.max_len
        self.net = _SASRec(self.n_items, d, L, int(c.get("layers", 2)), int(c.get("heads", 1)),
                           float(c.get("dropout", 0.2))).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(c.get("lr", 1e-3)), betas=(0.9, 0.98))
        self.train_seqs = seqs = self._sequences(train_df)
        users = np.array([u for u in range(self.n_users) if len(seqs[u]) >= 2])
        bs, epochs, patience = int(c.get("batch", 128)), int(c.get("epochs", 100)), int(c.get("patience", 6))
        best, best_state, bad, t0 = -1.0, None, 0, time.time()
        for ep in range(epochs):  # noqa: B007 - used after the loop
            self.net.train()
            rng.shuffle(users)
            tot, nb = 0.0, 0
            for s in range(0, len(users), bs):
                ub = users[s:s + bs]
                inp = np.zeros((len(ub), L), dtype=np.int64)
                tgt = np.zeros((len(ub), L), dtype=np.int64)
                for r, u in enumerate(ub):
                    sq = seqs[u]
                    if len(sq) > L + 1:  # random crop of L+1 consecutive events for long users
                        st = rng.integers(0, len(sq) - L)
                        sq = sq[st:st + L + 1]
                    x, y = sq[:-1], sq[1:]
                    inp[r, -len(x):], tgt[r, -len(y):] = x, y
                inp_t, tgt_t = torch.as_tensor(inp, device=self.device), torch.as_tensor(tgt, device=self.device)
                neg_t = torch.randint(1, self.n_items + 1, tgt_t.shape, device=self.device)
                h = self.net(inp_t)
                e_pos, e_neg = self.net.item(tgt_t), self.net.item(neg_t)
                lp, ln = (h * e_pos).sum(-1), (h * e_neg).sum(-1)
                m = (tgt_t > 0).float()
                loss = -((F.logsigmoid(lp) + F.logsigmoid(-ln)) * m).sum() / m.sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            v = val_eval(self)
            if v > best:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        self.net.load_state_dict(best_state)
        self._cache_items()
        return {"epochs": ep + 1, "best_val_ndcg10": best, "fit_seconds": time.time() - t0,
                "last_loss": tot / max(nb, 1)}

    @torch.no_grad()
    def _cache_items(self):
        self.net.eval()
        self.V = self.net.item.weight[1:]

    @torch.no_grad()
    def score(self, users, hist):
        self._cache_items()  # the item table is the output layer, so the "cache" is a free view
        seqs = getattr(self, "eval_seqs", self.train_seqs)
        h = self.net(self._history_tensor(users, seqs))[:, -1]
        return h @ self.V.T
