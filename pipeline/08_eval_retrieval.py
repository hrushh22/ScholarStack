"""Step 8 — Retrieval experiments (no LLM needed).

Compares, on the held-out set of 240 job descriptions:
  * query construction (raw JD vs. title+skills vs. hybrid)
  * chunking strategy (section vs. fixed-length)
  * embedding model (MiniLM vs. BGE-small vs. GTE-small) vs. the TF-IDF baseline
  * k (1..20)
Relevance proxies (no gold labels exist):
  category  - retrieved chunk comes from a resume in the JD's category
  skill     - retrieved chunk shares >= 1 hard skill with the JD
Also: per-category precision, JD -> O*NET occupation linking accuracy, latency.

Output: reports/results/retrieval_*.csv, onet_linking_eval.json
"""
import json
import time

import numpy as np
import pandas as pd

from scholarstack.baseline import TfidfRetriever
from scholarstack.chunking import load_chunks
from scholarstack.config import CFG, log, path, seed_everything
from scholarstack.embed import encode, load_embeddings, slug
from scholarstack.evalset import load_eval_set
from scholarstack.index import build_flat
from scholarstack.metrics import precision_at_k, reciprocal_rank
from scholarstack.onet import OnetMatcher
from scholarstack.query import QUERY_VARIANTS, build_query, jd_skills
from scholarstack.skills import SkillExtractor

KS = CFG["retrieval"]["k_sweep"]
KMAX = max(KS)


def score_hits(hit_rows: list[pd.DataFrame], ev: pd.DataFrame, chunk_skills: dict, jd_sk: list[set]) -> pd.DataFrame:
    rows = []
    for (i, jd), hits in zip(ev.iterrows(), hit_rows):
        cat_rel = (hits["category"].values == jd["category"]).tolist()
        skill_rel = [len(chunk_skills.get(cid, set()) & jd_sk[i]) > 0 for cid in hits["chunk_id"]]
        row = {"job_id": jd["job_id"], "category": jd["category"], "mrr_category": reciprocal_rank(cat_rel)}
        for k in KS:
            row[f"p@{k}_category"] = precision_at_k(cat_rel, k)
            row[f"p@{k}_skill"] = precision_at_k(skill_rel, k)
            row[f"resumes@{k}"] = hits["resume_id"].iloc[:k].nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def dense_hits(chunks: pd.DataFrame, vecs: np.ndarray, queries: list[str], model: str):
    index = build_flat(vecs)
    t0 = time.perf_counter()
    q = encode(queries, model, query=True)
    embed_ms = 1000 * (time.perf_counter() - t0) / len(queries)
    t0 = time.perf_counter()
    _, ids = index.search(q, KMAX)
    search_ms = 1000 * (time.perf_counter() - t0) / len(queries)
    return [chunks.iloc[row] for row in ids], embed_ms, search_ms


def summarise(per_jd: pd.DataFrame, **labels) -> list[dict]:
    out = []
    for k in KS:
        out.append({**labels, "k": k,
                    "precision_category": per_jd[f"p@{k}_category"].mean(),
                    "precision_skill": per_jd[f"p@{k}_skill"].mean(),
                    "distinct_resumes": per_jd[f"resumes@{k}"].mean(),
                    "mrr_category": per_jd["mrr_category"].mean()})
    return out


if __name__ == "__main__":
    seed_everything()
    ex = SkillExtractor()
    ev = load_eval_set().reset_index(drop=True)
    jd_sk = [set(jd_skills(jd, ex)) for jd in ev["jd"]]
    serving_model, serving_strategy = CFG["embedding"]["serving_model"], CFG["chunking"]["serving_strategy"]

    chunks = {s: load_chunks(s).reset_index(drop=True) for s in ("section", "fixed")}
    chunk_skills = {s: {cid: {x for x in ex.extract(t) if ex.by_id[x]["kind"] != "soft"}
                        for cid, t in zip(df["chunk_id"], df["text"])} for s, df in chunks.items()}

    results, per_cat = [], None

    # 1) Query construction (serving model + serving chunking).
    for variant in QUERY_VARIANTS:
        queries = [build_query(jd, ex, variant) for jd in ev["jd"]]
        hits, emb_ms, s_ms = dense_hits(chunks[serving_strategy],
                                        load_embeddings(f"chunks_{serving_strategy}", serving_model), queries, serving_model)
        pj = score_hits(hits, ev, chunk_skills[serving_strategy], jd_sk)
        results += summarise(pj, experiment="query", system=f"dense/{variant}", chunking=serving_strategy,
                             model=slug(serving_model), query=variant, embed_ms=emb_ms, search_ms=s_ms)
        log.info("query=%-6s P@8 category %.3f skill %.3f", variant, pj["p@8_category"].mean(), pj["p@8_skill"].mean())

    # Pick the best query variant by P@8 (category) for the remaining experiments.
    qdf = pd.DataFrame(results)
    best_variant = qdf[qdf.k == 8].sort_values("precision_category", ascending=False)["query"].iloc[0]
    queries = [build_query(jd, ex, best_variant) for jd in ev["jd"]]
    log.info("best query construction: %s", best_variant)

    # 2) Chunking x embedding model, plus TF-IDF baseline.
    for strategy, df in chunks.items():
        for model in CFG["embedding"]["candidates"]:
            hits, emb_ms, s_ms = dense_hits(df, load_embeddings(f"chunks_{strategy}", model), queries, model)
            pj = score_hits(hits, ev, chunk_skills[strategy], jd_sk)
            results += summarise(pj, experiment="model_chunking", system="dense", chunking=strategy,
                                 model=slug(model), query=best_variant, embed_ms=emb_ms, search_ms=s_ms)
            log.info("%-7s %-20s P@8 category %.3f skill %.3f", strategy, slug(model),
                     pj["p@8_category"].mean(), pj["p@8_skill"].mean())
            if strategy == serving_strategy and model == serving_model:
                per_cat = pj
        tf = TfidfRetriever(df)
        t0 = time.perf_counter()
        hits = [tf.search(q, KMAX) for q in queries]
        s_ms = 1000 * (time.perf_counter() - t0) / len(queries)
        pj = score_hits(hits, ev, chunk_skills[strategy], jd_sk)
        results += summarise(pj, experiment="model_chunking", system="tfidf", chunking=strategy, model="tf-idf",
                             query=best_variant, embed_ms=0.0, search_ms=s_ms)
        log.info("%-7s %-20s P@8 category %.3f skill %.3f", strategy, "TF-IDF baseline",
                 pj["p@8_category"].mean(), pj["p@8_skill"].mean())

    res = pd.DataFrame(results).round(4)
    res.to_csv(path("results") / "retrieval_eval.csv", index=False)

    # 3) Per-category results for the serving configuration.
    by_cat = per_cat.groupby("category")[["p@8_category", "p@8_skill", "mrr_category"]].mean().round(3)
    by_cat.to_csv(path("results") / "retrieval_by_category.csv")

    # 4) JD -> O*NET linking accuracy, against the occupation linked from the posting title.
    links = pd.read_parquet(path("processed") / "posting_onet.parquet")
    ev_l = ev.merge(links, on="job_id", how="left")
    matcher = OnetMatcher()
    top1 = top3 = n = 0
    for jd, code in zip(ev_l["jd"], ev_l["onet_code"]):
        if not isinstance(code, str):
            continue
        hits = matcher.match(build_query(jd, ex, best_variant), n=3)["code"].tolist()
        n += 1
        top1 += hits[0] == code
        top3 += code in hits
    # Same SOC major group (first 2 digits) is a softer, still meaningful agreement.
    onet_eval = {"n": n, "top1_exact": round(top1 / n, 3), "top3_exact": round(top3 / n, 3)}
    major = sum(matcher.match(build_query(jd, ex, best_variant), n=1)["code"].iloc[0][:2] == code[:2]
                for jd, code in zip(ev_l["jd"], ev_l["onet_code"]) if isinstance(code, str))
    onet_eval["top1_same_major_group"] = round(major / n, 3)
    (path("results") / "onet_linking_eval.json").write_text(json.dumps(onet_eval, indent=2))
    log.info("O*NET linking: %s", onet_eval)

    print(res[res.k == 8][["experiment", "system", "chunking", "model", "query", "precision_category",
                           "precision_skill", "mrr_category", "embed_ms", "search_ms"]].to_string(index=False))
