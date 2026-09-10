"""Mult-VAE (Liang et al., 2018): multinomial likelihood variational autoencoder over user rows."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import Recommender, to_dense_rows


class _VAE(nn.Module):
    def __init__(self, n_items, hidden=600, latent=200, dropout=0.5):
        super().__init__()
        self.enc = nn.Linear(n_items, hidden)
        self.mu, self.logvar = nn.Linear(hidden, latent), nn.Linear(hidden, latent)
        self.dec1, self.dec2 = nn.Linear(latent, hidden), nn.Linear(hidden, n_items)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.normalize(x)
        h = self.drop(h)
        h = torch.tanh(self.enc(h))
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.dec2(torch.tanh(self.dec1(z))), mu, logvar


class MultVAE(Recommender):
    name = "multvae"
    uses_gpu = True

    def fit(self, train, train_df, val_hist, val_eval):
        c = self.cfg
        torch.manual_seed(int(c.get("seed", 0)))
        self.net = _VAE(self.n_items, int(c.get("hidden", 600)), int(c.get("latent", 200)),
                        float(c.get("dropout", 0.5))).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(c.get("lr", 1e-3)))
        users = np.arange(self.n_users)
        bs, epochs, patience = int(c.get("batch", 512)), int(c.get("epochs", 100)), int(c.get("patience", 10))
        beta_max, anneal_steps = float(c.get("beta", 0.2)), int(c.get("anneal_steps", 20000))
        rng = np.random.default_rng(int(c.get("seed", 0)))
        best, best_state, bad, step, t0 = -1.0, None, 0, 0, time.time()
        for ep in range(epochs):  # noqa: B007 - used after the loop
            self.net.train()
            rng.shuffle(users)
            for s in range(0, len(users), bs):
                x = to_dense_rows(train, users[s:s + bs], self.device)
                beta = min(beta_max, beta_max * step / anneal_steps)
                logits, mu, logvar = self.net(x)
                nll = -(F.log_softmax(logits, 1) * x).sum(1).mean()
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
                loss = nll + beta * kl
                opt.zero_grad()
                loss.backward()
                opt.step()
                step += 1
            v = val_eval(self)
            if v > best:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        self.net.load_state_dict(best_state)
        return {"epochs": ep + 1, "best_val_ndcg10": best, "fit_seconds": time.time() - t0}

    @torch.no_grad()
    def score(self, users, hist):
        self.net.eval()
        return self.net(to_dense_rows(hist, users, self.device))[0]
