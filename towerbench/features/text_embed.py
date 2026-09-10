"""Frozen sentence-transformer item-text embeddings (GPU), cached per dataset/model."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..paths import PREPARED


def embed_items(dataset: str, model_name: str = "BAAI/bge-base-en-v1.5", batch: int = 256,
                device: str = "cuda", max_len: int = 256) -> str:
    from sentence_transformers import SentenceTransformer

    out = PREPARED / dataset / f"text_emb_{model_name.split('/')[-1]}.npy"
    if out.exists():
        return str(out)
    items = pd.read_parquet(PREPARED / dataset / "items.parquet")
    texts = items.text.fillna("").astype(str).tolist()
    t0 = time.time()
    m = SentenceTransformer(model_name, device=device)
    m.max_seq_length = max_len
    m.half()
    emb = m.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=True,
                   convert_to_numpy=True).astype(np.float32)
    np.save(out, emb)
    (PREPARED / dataset / "text_emb.log").write_text(
        f"{model_name} dim={emb.shape[1]} n={len(texts)} seconds={time.time() - t0:.1f}\n")
    return str(out)
