"""FAISS vector indexes: build, persist, benchmark, and a metadata-aware retriever."""
from __future__ import annotations

import time
from dataclasses import dataclass

import faiss
import numpy as np
import pandas as pd

from .config import CFG, log, path
from .embed import encode, load_embeddings, slug


def build_flat(vecs: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def build_hnsw(vecs: np.ndarray, m: int = 32, ef_search: int = 64) -> faiss.Index:
    index = faiss.IndexHNSWFlat(vecs.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    index.add(vecs)
    index.hnsw.efSearch = ef_search
    return index


def build_ivf(vecs: np.ndarray, nlist: int | None = None, nprobe: int = 16) -> faiss.Index:
    nlist = nlist or int(4 * np.sqrt(len(vecs)))
    quantizer = faiss.IndexFlatIP(vecs.shape[1])
    index = faiss.IndexIVFFlat(quantizer, vecs.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vecs)
    index.add(vecs)
    index.nprobe = nprobe
    return index


BUILDERS = {"Flat (exact)": build_flat, "HNSW (M=32)": build_hnsw, "IVF-Flat": build_ivf}


def benchmark(vecs: np.ndarray, queries: np.ndarray, k: int = 10) -> pd.DataFrame:
    """Build time, query latency and recall@k (vs exact search) for each index type."""
    exact = build_flat(vecs)
    _, truth = exact.search(queries, k)
    rows = []
    for name, builder in BUILDERS.items():
        t0 = time.perf_counter()
        idx = builder(vecs)
        build_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        for q in queries:  # one query at a time, like the live system
            idx.search(q[None, :], k)
        latency_ms = 1000 * (time.perf_counter() - t0) / len(queries)
        _, got = idx.search(queries, k)
        recall = np.mean([len(set(a) & set(b)) / k for a, b in zip(truth, got)])
        rows.append({"index": name, "vectors": len(vecs), "build_s": round(build_s, 3),
                     "latency_ms": round(latency_ms, 3), f"recall@{k}": round(float(recall), 4)})
    return pd.DataFrame(rows)


def save_index(index: faiss.Index, name: str) -> None:
    faiss.write_index(index, str(path("models") / f"faiss_{name}.index"))


def load_index(name: str) -> faiss.Index:
    return faiss.read_index(str(path("models") / f"faiss_{name}.index"))


@dataclass
class Retriever:
    """Top-k resume-chunk retrieval with optional category pre-filtering."""

    strategy: str = CFG["chunking"]["serving_strategy"]
    model_name: str = CFG["embedding"]["serving_model"]

    def __post_init__(self):
        from .chunking import load_chunks

        self.chunks = load_chunks(self.strategy).reset_index(drop=True)
        self.vecs = load_embeddings(f"chunks_{self.strategy}", self.model_name)
        self.index = build_flat(self.vecs)
        self._by_cat = {c: np.where(self.chunks["category"].values == c)[0].astype("int64")
                        for c in self.chunks["category"].unique()}

    @property
    def name(self) -> str:
        return f"{self.strategy}/{slug(self.model_name)}"

    def search_vec(self, qvec: np.ndarray, k: int, category: str | None = None) -> pd.DataFrame:
        params = None
        if category is not None and category in self._by_cat:
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(self._by_cat[category]))
        scores, ids = self.index.search(qvec.reshape(1, -1).astype("float32"), k, params=params)
        keep = ids[0] >= 0
        hits = self.chunks.iloc[ids[0][keep]].copy()
        hits["score"] = scores[0][keep]
        hits["rank"] = np.arange(1, len(hits) + 1)
        return hits

    def search(self, query: str, k: int | None = None, category: str | None = None) -> pd.DataFrame:
        q = encode([query], self.model_name, query=True)[0]
        return self.search_vec(q, k or CFG["retrieval"]["k"], category)
