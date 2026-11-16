"""Evaluation metrics: retrieval relevance, groundedness, skill coverage.

Groundedness is defined the same way for every system so they can be compared:
a generated statement counts as *supported* when its embedding has cosine
similarity >= tau with at least one chunk in the evidence pool retrieved for
that job description. For the RAG system we additionally check that every
cited chunk id was really among the retrieved chunks (citation validity).
"""
from __future__ import annotations

import numpy as np

from .config import CFG


def precision_at_k(relevant: list[bool], k: int) -> float:
    rel = relevant[:k]
    return float(np.mean(rel)) if rel else 0.0


def reciprocal_rank(relevant: list[bool]) -> float:
    for i, r in enumerate(relevant, start=1):
        if r:
            return 1.0 / i
    return 0.0


def statement_support(statements: list[str], evidence_vecs: np.ndarray, encode_fn) -> np.ndarray:
    """Max cosine between each statement and the evidence pool."""
    if not statements or evidence_vecs is None or len(evidence_vecs) == 0:
        return np.zeros(len(statements))
    s_vecs = encode_fn(statements)
    return (s_vecs @ evidence_vecs.T).max(axis=1)


def groundedness(support: np.ndarray, tau: float | None = None) -> float:
    tau = CFG["evaluation"]["groundedness_threshold"] if tau is None else tau
    return float((support >= tau).mean()) if len(support) else 0.0


def citation_validity(cited: list[list[str]], retrieved_ids: set[str]) -> dict:
    n = len(cited)
    if n == 0:
        return {"citation_rate": 0.0, "valid_citation_rate": 0.0}
    with_cite = [c for c in cited if c]
    valid = [c for c in with_cite if all(x in retrieved_ids for x in c)]
    return {"citation_rate": len(with_cite) / n, "valid_citation_rate": len(valid) / n}


def skill_coverage(target_skills: list[str], output_skills: set[str]) -> float:
    if not target_skills:
        return float("nan")
    return float(np.mean([s in output_skills for s in target_skills]))
