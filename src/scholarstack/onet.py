"""Linking job postings to O*NET occupations and summarising labour-market demand.

1. Every O*NET title (occupation title + ~54k alternate titles) is embedded and
   put in a FAISS index.
2. Each of the ~106k posting titles is embedded and matched to its nearest
   O*NET title -> occupation code (kept when cosine >= MIN_TITLE_SIM).
3. Postings are aggregated per occupation: volume, top extracted skills,
   median salary, remote share, experience levels, common titles.

At query time a job description is matched against O*NET occupation *profiles*
(title + description + skills + technologies), which needs no title at all.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .clean import load_onet_profiles
from .config import CFG, log, path
from .embed import encode, load_embeddings
from .index import build_flat

MIN_TITLE_SIM = 0.62


def title_table(onet: pd.DataFrame) -> pd.DataFrame:
    rows = [(r.code, r.title) for r in onet.itertuples()]
    rows += [(r.code, t) for r in onet.itertuples() for t in r.alt_titles]
    df = pd.DataFrame(rows, columns=["code", "title"]).drop_duplicates()
    df["title_l"] = df["title"].str.lower()
    return df.drop_duplicates("title_l").reset_index(drop=True)


def link_postings() -> pd.DataFrame:
    onet = load_onet_profiles()
    titles = title_table(onet)
    log.info("embedding %d O*NET titles", len(titles))
    t_vecs = encode(titles["title"].tolist(), batch_size=512)
    index = build_flat(t_vecs)

    postings = pd.read_parquet(path("processed") / "postings",
                               columns=["job_id", "title", "normalized_salary", "remote_allowed",
                                        "formatted_experience_level", "industries", "category"])
    uniq = postings["title"].str.lower().str.strip().drop_duplicates().reset_index(drop=True)
    log.info("embedding %d unique posting titles (of %d postings)", len(uniq), len(postings))
    p_vecs = encode(uniq.tolist(), batch_size=512)
    scores, ids = index.search(p_vecs, 1)
    title_map = pd.DataFrame({"title_l": uniq, "onet_code": titles["code"].values[ids[:, 0]],
                              "onet_match_title": titles["title"].values[ids[:, 0]], "onet_sim": scores[:, 0]})
    postings["title_l"] = postings["title"].str.lower().str.strip()
    linked = postings.merge(title_map, on="title_l", how="left").drop(columns="title_l")
    linked.loc[linked["onet_sim"] < MIN_TITLE_SIM, "onet_code"] = None
    linked[["job_id", "onet_code", "onet_match_title", "onet_sim"]].to_parquet(
        path("processed") / "posting_onet.parquet", index=False)
    log.info("linked %.1f%% of postings to an O*NET occupation (cosine >= %.2f)",
             100 * linked["onet_code"].notna().mean(), MIN_TITLE_SIM)
    return linked


def market_stats(linked: pd.DataFrame) -> pd.DataFrame:
    from .skills import SkillExtractor

    ex = SkillExtractor()
    skills = pd.read_parquet(path("processed") / "posting_skills", columns=["job_id", "skills"])
    df = linked.dropna(subset=["onet_code"]).merge(skills, on="job_id", how="left")
    out = []
    for code, g in df.groupby("onet_code"):
        exploded = g["skills"].fillna("").str.split("|").explode()
        exploded = exploded[exploded != ""]
        top = exploded.value_counts().head(20)
        salary = g["normalized_salary"].dropna()
        salary = salary[(salary > 15_000) & (salary < 1_000_000)]
        out.append({
            "code": code,
            "n_postings": int(len(g)),
            "top_skills": [{"id": s, "name": ex.by_id[s]["name"], "share": round(c / len(g), 3)}
                           for s, c in top.items() if s in ex.by_id],
            "median_salary": float(salary.median()) if len(salary) >= 5 else None,
            "remote_share": round(float(g["remote_allowed"].mean()), 3),
            "experience_levels": g["formatted_experience_level"].fillna("Not stated").value_counts().head(4).to_dict(),
            "top_titles": g["title"].value_counts().head(5).index.tolist(),
        })
    stats = pd.DataFrame(out)
    store = stats.copy()
    for c in ("top_skills", "experience_levels", "top_titles"):
        store[c] = store[c].map(json.dumps)
    store.to_parquet(path("processed") / "onet_market_stats.parquet", index=False)
    log.info("market stats for %d occupations (median %d postings each)", len(stats), stats["n_postings"].median())
    return stats


def load_market_stats() -> pd.DataFrame:
    df = pd.read_parquet(path("processed") / "onet_market_stats.parquet")
    for c in ("top_skills", "experience_levels", "top_titles"):
        df[c] = df[c].map(json.loads)
    return df


class OnetMatcher:
    """Job description -> most similar O*NET occupation profiles."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or CFG["embedding"]["serving_model"]
        self.onet = load_onet_profiles()
        vecs = load_embeddings("onet_profiles", self.model_name)
        keep = self.onet["has_profile"].values
        self.onet = self.onet[keep].reset_index(drop=True)
        self.index = build_flat(np.ascontiguousarray(vecs[keep]))

    def match_vec(self, qvec: np.ndarray, n: int | None = None) -> pd.DataFrame:
        n = n or CFG["retrieval"]["onet_top_n"]
        scores, ids = self.index.search(qvec.reshape(1, -1).astype("float32"), n)
        hits = self.onet.iloc[ids[0]].copy()
        hits["score"] = scores[0]
        return hits

    def match(self, text: str, n: int | None = None) -> pd.DataFrame:
        return self.match_vec(encode([text], self.model_name, query=True)[0], n)
