"""Batched sentence-transformer embeddings (L2-normalised, so inner product = cosine)."""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

from .config import CFG, device, log, path

# Retrieval-tuned models expect an instruction on the query side only.
QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "}


def slug(model_name: str) -> str:
    return model_name.split("/")[-1].replace(".", "_")


@lru_cache(maxsize=4)
def load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device())
    model.max_seq_length = CFG["embedding"]["max_seq_length"]
    import logging

    for noisy in ("httpx", "httpcore", "huggingface_hub"):  # HF re-enables request logging on load
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return model


def encode(texts, model_name: str | None = None, query: bool = False, batch_size: int | None = None,
           show_progress: bool = False) -> np.ndarray:
    model_name = model_name or CFG["embedding"]["serving_model"]
    texts = list(texts)
    if query and model_name in QUERY_PREFIX:
        texts = [QUERY_PREFIX[model_name] + t for t in texts]
    vecs = load_model(model_name).encode(
        texts, batch_size=batch_size or CFG["embedding"]["batch_size"], normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=show_progress)
    return vecs.astype("float32")


def embed_and_save(texts, name: str, model_name: str) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    vecs = encode(texts, model_name, show_progress=False)
    secs = time.perf_counter() - t0
    out = path("models") / f"emb_{name}_{slug(model_name)}.npy"
    np.save(out, vecs)
    log.info("embedded %-22s %6d x %d with %-24s in %5.1fs (%.0f/s)", name, *vecs.shape, slug(model_name),
             secs, len(vecs) / secs)
    return vecs, secs


def load_embeddings(name: str, model_name: str | None = None) -> np.ndarray:
    model_name = model_name or CFG["embedding"]["serving_model"]
    return np.load(path("models") / f"emb_{name}_{slug(model_name)}.npy")
