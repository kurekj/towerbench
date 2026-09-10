"""Two-tower retrieval model with the SOTA training techniques studied in the source project.

Item tower : item-ID embedding (reserved OOV row 0, id-dropout) + optional side features
             (frozen sentence-transformer text embedding, standardized numeric geo/price,
             category embedding) -> MLP -> L2 norm.  Optional DCN-v2 cross layers.
User tower : shared item tower applied to the user's last L history items -> positional
             (recency) embedding -> multi-head learnable-query attention pooling -> MLP -> L2.
Loss       : in-batch sampled softmax with logQ correction (Yi et al., 2019), temperature,
             accidental-hit masking, optional extra uniform negatives (mixed negative sampling),
             optional MixGCF-style positive-mixing hard negatives.
Every technique is a config switch so ablations are single-flag changes.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .base import Recommender


class CrossLayer(nn.Module):  # DCN-v2, full rank
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, d)

    def forward(self, x0, x):
        return x0 * self.w(x) + x


class ItemTower(nn.Module):
    def __init__(self, n_items, d, feat_dim, n_cat, n_cross, p_id, dropout):
        super().__init__()
        self.emb = nn.Embedding(n_items + 1, d, padding_idx=0)  # row 0 = OOV / padding
        self.p_id = p_id
        self.feat = nn.Sequential(nn.Linear(feat_dim, d), nn.ReLU()) if feat_dim else None
        self.cat = nn.Embedding(n_cat + 1, d, padding_idx=0) if n_cat else None
        self.cross = nn.ModuleList(CrossLayer(d) for _ in range(n_cross))
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(d, d))

    def forward(self, ids, feats=None, cats=None):  # ids are 1-based, 0 = pad
        if self.training and self.p_id > 0:
            ids = ids * (torch.rand_like(ids, dtype=torch.float) > self.p_id).long()
        x = self.emb(ids)
        if self.feat is not None and feats is not None:
            x = x + self.feat(feats)
        if self.cat is not None and cats is not None:
            x = x + self.cat(cats)
        x0 = x
        for c in self.cross:
            x = c(x0, x)
        return F.normalize(self.mlp(x), dim=-1)


class UserTower(nn.Module):
    def __init__(self, d, max_len, heads, dropout):
        super().__init__()
        self.pos = nn.Embedding(max_len + 1, d, padding_idx=0)
        self.q = nn.Parameter(torch.randn(heads, d) * 0.02)
        self.proj = nn.Linear(d, d)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(d, d))
        self.heads = heads

    def forward(self, h, mask):  # h [B,L,d], mask [B,L] bool (True = real item)
        L = h.shape[1]
        pos = torch.arange(L, 0, -1, device=h.device).unsqueeze(0) * mask  # recency 1 = last
        k = torch.relu(self.proj(h + self.pos(pos)))
        logits = torch.einsum("bld,hd->bhl", k, self.q)
        logits = logits.masked_fill(~mask.unsqueeze(1), -1e4)
        a = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("bhl,bld->bhd", a, h).mean(1)
        pooled = pooled * mask.any(1, keepdim=True)  # empty history -> zero vector
        return F.normalize(self.out(pooled), dim=-1)


class TwoTower(Recommender):
    name = "two_tower"
    uses_gpu = True

    # ------------------------------------------------------------------ features
    def _build_features(self):
        c, it = self.cfg, self.items
        feats, self.cats = [], None
        if c.get("use_text") and it is not None and "text_emb" in it.attrs:
            feats.append(it.attrs["text_emb"])
        if c.get("use_numeric") and it is not None:
            cols = [k for k in ("lat", "lon", "price") if it[k].notna().mean() > 0.5]
            if cols:
                num = it[cols].copy()
                if "price" in cols:
                    num["price"] = np.log1p(num["price"].clip(lower=0))
                num = (num - num.mean()) / (num.std() + 1e-6)
                feats.append(num.fillna(0).values.astype(np.float32))
        if c.get("use_cat") and it is not None and (it.cat != "").mean() > 0.5:
            codes = pd.Categorical(it.cat).codes.astype(np.int64) + 1
            self.cats = torch.as_tensor(np.concatenate([[0], codes]), device=self.device)
            self.n_cat = int(codes.max())
        else:
            self.n_cat = 0
        if feats:
            f = np.concatenate(feats, axis=1).astype(np.float32)
            self.feats = torch.as_tensor(np.concatenate([np.zeros((1, f.shape[1]), np.float32), f]),
                                         device=self.device)
            return f.shape[1]
        self.feats = None
        return 0

    def _item_batch(self, ids):
        f = self.feats[ids] if self.feats is not None else None
        ca = self.cats[ids] if self.cats is not None else None
        return self.item_tower(ids, f, ca)

    # ------------------------------------------------------------------ data
    def _sequences(self, df):
        df = df.sort_values(["u", "ts"], kind="stable")
        seqs = [np.empty(0, dtype=np.int64)] * self.n_users
        for u, g in df.groupby("u"):
            seqs[u] = g.i.values.astype(np.int64) + 1
        return seqs

    def _history_tensor(self, users, seqs):
        L = self.max_len
        h = np.zeros((len(users), L), dtype=np.int64)
        for r, u in enumerate(users):
            s = seqs[u][-L:]
            if len(s):
                h[r, -len(s):] = s
        return torch.as_tensor(h, device=self.device)

    def _user_batch(self, hist_ids):
        mask = hist_ids > 0
        h = self._item_batch(hist_ids)
        return self.user_tower(h, mask)

    # ------------------------------------------------------------------ training
    def fit(self, train, train_df, val_hist, val_eval):
        c = self.cfg
        seed = int(c.get("seed", 0))
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        d, self.max_len = int(c.get("dim", 128)), int(c.get("max_len", 50))
        feat_dim = self._build_features()
        self.item_tower = ItemTower(self.n_items, d, feat_dim, self.n_cat, int(c.get("cross_layers", 0)),
                                    float(c.get("p_id", 0.0)), float(c.get("dropout", 0.1))).to(self.device)
        self.user_tower = UserTower(d, self.max_len, int(c.get("heads", 1)),
                                    float(c.get("dropout", 0.1))).to(self.device)
        params = list(self.item_tower.parameters()) + list(self.user_tower.parameters())
        opt = torch.optim.AdamW(params, lr=float(c.get("lr", 1e-3)), weight_decay=float(c.get("wd", 1e-6)))
        tau = float(c.get("tau", 0.05))
        logq, n_extra, gamma = bool(c.get("logq", True)), int(c.get("extra_neg", 0)), float(c.get("mixgcf", 0.0))
        pop = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
        q = torch.as_tensor(np.concatenate([[0.0], (pop + 1.0) / (pop + 1.0).sum()]), device=self.device).float()
        self.train_seqs = seqs = self._sequences(train_df)
        users = np.array([u for u in range(self.n_users) if len(seqs[u]) >= 2])
        bs, epochs, patience = int(c.get("batch", 1024)), int(c.get("epochs", 50)), int(c.get("patience", 5))
        spu = int(c.get("samples_per_user", 0))
        # One epoch = every training event with >=1 preceding event serves as target once
        # (samples_per_user == 0); otherwise `spu` random (user, position) draws per user.
        pairs = (np.concatenate([np.stack([np.full(len(seqs[u]) - 1, u), np.arange(1, len(seqs[u]))], 1)
                                 for u in users]) if spu == 0 else None)
        # flat event array + per-user offsets -> vectorized (history, target) batch construction
        lengths = np.array([len(s) for s in seqs], dtype=np.int64)
        offsets = np.zeros(self.n_users + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        flat = np.concatenate(seqs) if offsets[-1] > 0 else np.zeros(1, dtype=np.int64)
        L, ar = self.max_len, np.arange(self.max_len)

        def build(ub, tb):
            base = offsets[ub][:, None] + tb[:, None] - L + ar[None, :]      # seq positions t-L..t-1
            valid = ar[None, :] >= (L - tb)[:, None]                         # position >= 0
            hist = np.where(valid, flat[np.clip(base, 0, len(flat) - 1)], 0)
            return hist, flat[offsets[ub] + tb]

        best, best_state, bad, t0 = -1.0, None, 0, time.time()
        scaler_ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda"
                      else torch.autocast("cpu", enabled=False))
        for ep in range(epochs):  # noqa: B007 - used after the loop
            self.item_tower.train()
            self.user_tower.train()
            if pairs is not None:
                order = pairs[rng.permutation(len(pairs))]
            else:
                ou = np.repeat(users, spu)
                rng.shuffle(ou)
                order = np.stack([ou, np.array([rng.integers(1, len(seqs[u])) for u in ou])], 1)
            tot, nb = 0.0, 0
            for s in range(0, len(order), bs):
                ub, tb = order[s:s + bs, 0], order[s:s + bs, 1]
                hist, tgt = build(ub, tb)
                hist_t = torch.as_tensor(hist, device=self.device)
                tgt_t = torch.as_tensor(tgt, device=self.device)
                with scaler_ctx:
                    ue = self._user_batch(hist_t)
                    ve = self._item_batch(tgt_t)
                    cand, cand_ids = ve, tgt_t
                    if n_extra:
                        extra = torch.randint(1, self.n_items + 1, (n_extra,), device=self.device)
                        cand = torch.cat([ve, self._item_batch(extra)], 0)
                        cand_ids = torch.cat([tgt_t, extra], 0)
                    logits = (ue @ cand.T).float() / tau
                    if logq:
                        logits = logits - torch.log(q[cand_ids]).unsqueeze(0)
                    same = cand_ids.unsqueeze(0) == tgt_t.unsqueeze(1)
                    same.diagonal().fill_(False)
                    logits = logits.masked_fill(same, -1e4)
                    if gamma > 0:  # MixGCF positive mixing: pull hardest in-batch negative toward positive
                        neg_logits = logits.clone().fill_diagonal_(-1e4)
                        hard = cand[neg_logits.argmax(1)]
                        mixed = F.normalize(gamma * ve + (1 - gamma) * hard, dim=-1)
                        logits = torch.cat([logits, ((ue * mixed).sum(1) / tau).unsqueeze(1)], 1)
                    labels = torch.arange(len(ub), device=self.device)
                    loss = F.cross_entropy(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            v = val_eval(self)
            if v > best:
                best, bad = v, 0
                best_state = ({k: t.detach().clone() for k, t in self.item_tower.state_dict().items()},
                              {k: t.detach().clone() for k, t in self.user_tower.state_dict().items()})
            else:
                bad += 1
                if bad >= patience:
                    break
        self.item_tower.load_state_dict(best_state[0])
        self.user_tower.load_state_dict(best_state[1])
        self._cache_items()
        return {"epochs": ep + 1, "best_val_ndcg10": best, "fit_seconds": time.time() - t0,
                "last_loss": tot / max(nb, 1), "feat_dim": feat_dim}

    @torch.no_grad()
    def _cache_items(self):
        self.item_tower.eval()
        ids = torch.arange(1, self.n_items + 1, device=self.device)
        self.V = torch.cat([self._item_batch(ids[s:s + 8192]) for s in range(0, self.n_items, 8192)], 0)

    @torch.no_grad()
    def score(self, users, hist):
        # Item embeddings are cached once per validation pass / after fit (see fit, set_history);
        # a stale cache silently corrupts early stopping, so refresh whenever the towers trained.
        if self.item_tower.training or not hasattr(self, "V"):
            self._cache_items()
        self.user_tower.eval()
        # history sequences from the sparse matrix in temporal order are not available; use the
        # training sequences (train) or train+val order supplied via set_history()
        seqs = getattr(self, "eval_seqs", self.train_seqs)
        ue = self._user_batch(self._history_tensor(users, seqs))
        return ue @ self.V.T

    def set_history(self, df):
        """Give the model the chronological history (train or train+val) used at ranking time."""
        self.eval_seqs = self._sequences(df)
        if hasattr(self, "item_tower"):
            self._cache_items()
