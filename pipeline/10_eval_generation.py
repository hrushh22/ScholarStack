"""Step 10 - Generation experiments (needs GROQ_API_KEY).

Systems, on the held-out generation subset (one JD per category):
  baseline           TF-IDF keyword advice (no retrieval, no generation)
  llm_only           large LLM prompted with the job description alone
  rag                full system: retrieved resume evidence + O*NET + large LLM (k = 8)
  rag_small          same with the small LLM                 (larger vs smaller model)
  rag_small_no_onet  small LLM, evidence without O*NET        (does O*NET grounding help?)  } ablations on a
  rag_small_k3/k15   small LLM with 3 or 15 excerpts          (number of retrieved docs)   } subset of JDs
Ablations use the small model and are compared with rag_small, so each comparison changes
exactly one thing (this also keeps the large model's free daily token budget for the main
comparison and the live demo).

Groundedness (claim-level verification, as in RAGAS "faithfulness")
  Every generated statement is checked by an independent verifier LLM (a different model
  family) against its closest evidence lines from the SAME pool of real-resume excerpts
  retrieved for that job (top-20), so every system is scored against the same evidence.
  yes = supported; partial = related but adds specifics; no = unsupported.
  Citation correctness: the supporting line comes from a chunk the statement cited.
  Negative control: some outputs are re-verified against evidence for an unrelated job.
  The live site's fast embedding check is calibrated to agree with these verdicts.
Also: JD / market skill coverage, pairwise LLM judge (randomised A/B order), latency, tokens.

All LLM responses are cached: stop and re-run freely; finished calls are not repeated.
"""
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from scholarstack.baseline import fit_advice_model, keyword_advice
from scholarstack.config import CFG, env, log, path, seed_everything
from scholarstack.embed import encode
from scholarstack.evalset import load_eval_set
from scholarstack.generate import PROMPTS, Generator, fill, output_text, statements, truncate_words
from scholarstack.llm import chat_model
from scholarstack.metrics import evidence_units, skill_coverage
from scholarstack.query import jd_skills

POOL_K = 20
N_CANDIDATES = 4
SYSTEMS = {  # name: (generator, mode, k, use_onet, ablation_only)
    "llm_only": ("large", "llm_only", 8, True, False),
    "rag": ("large", "rag", 8, True, False),
    "rag_small": ("small", "rag", 8, True, False),
    "rag_small_no_onet": ("small", "rag", 8, False, True),
    "rag_small_k3": ("small", "rag", 3, True, True),
    "rag_small_k15": ("small", "rag", 15, True, True),
}


def render(output: dict) -> str:
    lines = [output.get("headline", "")]
    lines += [s["text"] for s in output.get("summary") or []]
    for g in output.get("skills") or []:
        lines.append(f"{g.get('group', 'Skills')}: " + ", ".join(i.get("name", "") for i in g.get("items") or []))
    for r in output.get("experience") or []:
        lines.append(f"{r.get('title', '')} - {r.get('context', '')}")
        lines += [f"  - {b.get('text', '')}" for b in r.get("bullets") or []]
    lines += ["Education: " + e.get("text", "") for e in output.get("education") or []]
    lines += ["Certification: " + c.get("text", "") for c in output.get("certifications") or []]
    return "\n".join(l for l in lines if l.strip())


def render_baseline(advice: dict) -> str:
    return ("Resume keyword advice (TF-IDF): make sure your resume includes these terms from the job "
            "description: " + ", ".join(advice["keywords"]))


def embed_units(chunk_texts):
    units, owner = evidence_units(chunk_texts)
    return {"vecs": encode(units), "owner": np.array(owner), "units": units}


def verify(verifier, items: list[dict], depth: int = 0) -> list[dict]:
    """One verifier call for all statements of one output (split in half if the JSON fails)."""
    try:
        return _verify(verifier, items)
    except Exception:
        if depth >= 1 or len(items) < 4:
            raise
        mid = len(items) // 2
        return verify(verifier, items[:mid], depth + 1) + verify(verifier, items[mid:], depth + 1)


def _verify(verifier, items: list[dict]) -> list[dict]:
    blocks = []
    for i, it in enumerate(items, start=1):
        ev = "\n".join(f"   E{j}: {truncate_words(u, 45)}" for j, u in enumerate(it["candidates"], start=1))
        blocks.append(f"{i}. STATEMENT: {it['statement']}\n{ev}")
    msg = verifier.invoke([SystemMessage(PROMPTS["verify_system"]),
                           HumanMessage(fill(PROMPTS["verify_user"], items="\n\n".join(blocks)))])
    verdicts = {v.get("id"): v for v in json.loads(msg.content).get("verdicts", [])}
    return [verdicts.get(i, {}) for i in range(1, len(items) + 1)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=["baseline", *SYSTEMS])
    ap.add_argument("--limit", type=int, default=None, help="number of JDs (default: all 24)")
    ap.add_argument("--ablation-limit", type=int, default=8)
    ap.add_argument("--negatives", type=int, default=12,
                    help="JDs whose rag and llm_only outputs are re-verified against an unrelated job's evidence")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    seed_everything()
    ev = load_eval_set()
    ev = ev[ev["gen_subset"]].sort_values("category").reset_index(drop=True)
    if args.limit:
        ev = ev.head(args.limit)
    large = Generator(with_fallback=False)
    small = Generator(retriever=large.retriever, model=env("LLM_FALLBACK_MODEL", "openai/gpt-oss-20b"), with_fallback=False)
    gens = {"large": large, "small": small}
    models = {"large": env("LLM_MODEL"), "small": env("LLM_FALLBACK_MODEL")}
    ex = large.ex
    advice_model = fit_advice_model()
    judge_model = env("JUDGE_MODEL", "qwen/qwen3.8-27b")

    # ---- phase 0: retrieval, once ---------------------------------------------------------
    ctx, pools = {}, {}
    for _, r in ev.iterrows():
        for k in sorted({s[2] for s in SYSTEMS.values()}):
            ctx[(r.job_id, k)] = large.retrieve({"jd": r.jd, "k": k})
        pool = large.retriever.search(ctx[(r.job_id, 8)]["query"], k=POOL_K)
        pools[r.job_id] = {**embed_units(pool["text"].tolist()), "chunk_ids": pool["chunk_id"].tolist()}
    n = len(ev)
    negative_of = {ev.job_id[i]: ev.job_id[(i + n // 2) % n] for i in range(n)}  # different category
    log.info("retrieval done for %d JDs", n)

    # ---- phase A: generation, one thread per model ---------------------------------------
    tasks = {"large": [], "small": []}
    for i, r in ev.iterrows():
        for s in args.systems:
            if s == "baseline":
                continue
            g, mode, k, use_onet, abl = SYSTEMS[s]
            if not (abl and i >= args.ablation_limit):
                tasks[g].append((r.job_id, s, mode, k, use_onet))
    results, errors = {}, {}

    def run_tasks(gkey):
        for j, (job_id, s, mode, k, use_onet) in enumerate(tasks[gkey], start=1):
            try:
                results[(job_id, s)] = gens[gkey].complete(ctx[(job_id, k)], mode, use_onet)
            except Exception as err:
                errors[(job_id, s)] = str(err)[:300]
                log.warning("%s %s failed: %s", s, job_id, str(err)[:160])
            if j % 8 == 0:
                log.info("%s model: %d/%d generations", gkey, j, len(tasks[gkey]))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool_ex:
        list(pool_ex.map(run_tasks, [g for g in tasks if tasks[g]]))
    log.info("generation: %d outputs, %d errors in %.0fs", len(results), len(errors), time.time() - t0)

    # ---- phase B: statements, candidates, embedding scores --------------------------------
    stmt_rows, verify_jobs, rows, rendered = [], [], [], {}
    for _, r in ev.iterrows():
        hard = jd_skills(r.jd, ex)
        occ = ctx[(r.job_id, 8)]["occupation"]
        market = [s["id"] for s in (large.market.get(occ["code"]) or {}).get("top_skills", [])[:10]]
        P, N = pools[r.job_id], pools[negative_of[r.job_id]]
        for s in args.systems:
            rec = {"job_id": r.job_id, "category": r.category, "title": r.title, "system": s}
            if s == "baseline":
                adv = keyword_advice(advice_model, r.jd)
                text = " ; ".join(adv["keywords"])
                rendered[(r.job_id, s)] = render_baseline(adv)
                rec["n_statements"] = 0
            else:
                res = results.get((r.job_id, s))
                if res is None:
                    if (r.job_id, s) in errors:
                        rows.append({**rec, "error": errors[(r.job_id, s)]})
                    continue
                out = res["output"]
                sts = statements(out)
                text = output_text(out)
                rendered[(r.job_id, s)] = render(out)
                items = []
                if sts:
                    sv = encode([x["text"] for x in sts])
                    sims, nsims = sv @ P["vecs"].T, sv @ N["vecs"].T
                    for i_s, st in enumerate(sts):
                        order = np.argsort(-sims[i_s])
                        cand = [int(u) for u in order[:N_CANDIDATES]]
                        cited_chunks = {P["chunk_ids"].index(c) for c in st["sources"] if c in P["chunk_ids"]}
                        cited_units = [int(u) for u in order if P["owner"][u] in cited_chunks]
                        if cited_units and cited_units[0] not in cand:
                            cand = cand[:N_CANDIDATES - 1] + [cited_units[0]]
                        neg_order = np.argsort(-nsims[i_s])[:N_CANDIDATES]
                        row = {"job_id": r.job_id, "category": r.category, "system": s, "part": st["part"],
                               "statement": st["text"], "n_sources": len(st["sources"]),
                               "emb_support": float(sims[i_s].max()), "emb_support_negative": float(nsims[i_s].max()),
                               "candidates": [P["units"][u] for u in cand],
                               "candidate_cited": [bool(P["owner"][u] in cited_chunks) for u in cand],
                               "negative_candidates": [N["units"][u] for u in neg_order]}
                        stmt_rows.append(row)
                        items.append(len(stmt_rows) - 1)
                    verify_jobs.append(((r.job_id, s), items))
                rec.update(n_statements=len(sts), latency_s=res.get("llm_time_s"), model=res.get("model"),
                           invalid_citations_removed=res.get("invalid_citations", 0),
                           input_tokens=(res.get("usage") or {}).get("input_tokens"),
                           output_tokens=(res.get("usage") or {}).get("output_tokens"))
            out_sk = set(ex.extract(text))
            rec["jd_skill_coverage"] = skill_coverage(hard, out_sk)
            rec["market_skill_coverage"] = skill_coverage(market, out_sk)
            rows.append(rec)

    # ---- phase C: claim verification by an independent LLM --------------------------------
    verifier = chat_model(judge_model, json_mode=True, max_tokens=900, with_fallback=False, seconds_per_call=18,
                          reasoning_effort="none")
    t0 = time.time()
    for j, ((key, idxs)) in enumerate(verify_jobs, start=1):
        try:
            vs = verify(verifier, [{"statement": stmt_rows[i]["statement"], "candidates": stmt_rows[i]["candidates"]}
                                   for i in idxs])
        except Exception as err:
            log.warning("verify %s failed: %s", key, str(err)[:160])
            continue
        for i, v in zip(idxs, vs):
            if not isinstance(v.get("supported"), bool):
                continue
            stmt_rows[i]["supported"] = v["supported"]
            stmt_rows[i]["extra"] = bool(v.get("extra"))
            stmt_rows[i]["verdict"] = ("unsupported" if not v["supported"] else
                                       "supported_extra" if v.get("extra") else "supported")
            e = v.get("evidence") if v["supported"] else None
            k = int(e[1:]) - 1 if isinstance(e, str) and e[1:].isdigit() else None
            stmt_rows[i]["citation_correct"] = (bool(stmt_rows[i]["candidate_cited"][k])
                                                if k is not None and k < len(stmt_rows[i]["candidates"]) and
                                                stmt_rows[i]["n_sources"] else None)
        if j % 10 == 0:
            log.info("verified %d/%d outputs (%.0fs)", j, len(verify_jobs), time.time() - t0)
    # Negative control ("chance level"): the same statements checked against the closest lines of
    # evidence retrieved for a job in another category. Generic statements pass this too, so
    # grounding is reported as the lift of real-evidence support over this level.
    neg_ids = set(ev.job_id[: args.negatives])
    neg_jobs = [vj for vj in verify_jobs if vj[0][1] in ("rag", "llm_only") and vj[0][0] in neg_ids]
    neg_rows = []
    for key, idxs in neg_jobs:
        try:
            vs = verify(verifier, [{"statement": stmt_rows[i]["statement"], "candidates": stmt_rows[i]["negative_candidates"]}
                                   for i in idxs])
            neg_rows += [{"system": key[1], "job_id": key[0], "supported": v.get("supported")} for v in vs
                         if isinstance(v.get("supported"), bool)]
        except Exception as err:
            log.warning("negative control %s failed: %s", key, str(err)[:160])

    st = pd.DataFrame(stmt_rows)
    for col in ("verdict", "citation_correct", "supported", "extra"):
        if col not in st:
            st[col] = None
    st["supported"] = st["supported"].astype("boolean")
    st["extra"] = st["extra"].astype("boolean")
    st["supported_strict"] = (st["supported"] & ~st["extra"]).astype("boolean")

    # Calibrate the live embedding check against the verifier (maximise agreement).
    lab = st.dropna(subset=["supported"])
    y = lab["supported"].astype(bool).values
    grid = np.round(np.arange(0.30, 0.86, 0.01), 2)
    # balanced accuracy, so the threshold does not collapse onto the majority class
    acc = [0.5 * ((lab["emb_support"].values >= t)[y].mean() + (lab["emb_support"].values < t)[~y].mean())
           if y.any() and (~y).any() else 0 for t in grid]
    tau = float(grid[int(np.argmax(acc))])
    pos, negv = lab[y]["emb_support"].values, lab[~y]["emb_support"].values
    auc = float((pos[:, None] > negv[None, :]).mean()) if len(pos) and len(negv) else float("nan")
    neg_df = pd.DataFrame(neg_rows)
    calib = {
        "verifier": judge_model, "n_statements_verified": int(len(lab)),
        "positive_rate": round(float(y.mean()), 3) if len(y) else None,
        "negative_control_supported": round(float(neg_df["supported"].astype(bool).mean()), 3) if len(neg_df) else None,
        "negative_control_by_system": (neg_df.groupby("system")["supported"].mean().round(3).to_dict()
                                       if len(neg_df) else {}),
        "negative_control_n": int(len(neg_df)),
        "embedding_tau": tau, "embedding_balanced_accuracy": round(float(max(acc)), 3),
        "embedding_agreement_with_verifier": round(float(((lab["emb_support"] >= tau).values == y).mean()), 3) if len(y) else None,
        "embedding_auc_vs_verifier": round(auc, 3),
    }
    log.info("verification/calibration: %s", calib)

    agg = st.groupby(["job_id", "system"]).agg(
        grounded=("supported", "mean"), grounded_strict=("supported_strict", "mean"), added_specifics=("extra", "mean"),
        citation_rate=("n_sources", lambda x: (x > 0).mean()),
        citation_correct=("citation_correct", lambda x: x.dropna().astype(float).mean() if x.notna().any() else np.nan),
        embedding_grounded=("emb_support", lambda x: (x >= tau).mean())).reset_index()
    df = pd.DataFrame(rows).merge(agg, on=["job_id", "system"], how="left")
    df.to_csv(path("results") / "generation_eval.csv", index=False)
    st.drop(columns=["candidates", "negative_candidates", "candidate_cited"]).to_csv(
        path("results") / "generation_statements.csv", index=False)
    (path("results") / "groundedness_calibration.json").write_text(json.dumps(calib, indent=2))

    df["latency_s"] = df.get("latency_s")
    cols = [c for c in ["grounded", "grounded_strict", "added_specifics", "citation_rate", "citation_correct", "embedding_grounded",
                        "jd_skill_coverage", "market_skill_coverage", "n_statements", "invalid_citations_removed",
                        "latency_s", "input_tokens", "output_tokens"] if c in df]
    summary = df.groupby("system")[cols].mean().round(3)
    summary["n"] = df.dropna(subset=["jd_skill_coverage"]).groupby("system").size()
    summary["errors"] = df.groupby("system")["error"].count() if "error" in df else 0
    summary["model"] = summary.index.map(lambda s: "tf-idf" if s == "baseline" else models[SYSTEMS[s][0]])
    for s_name, rate in calib["negative_control_by_system"].items():
        if s_name in summary.index:
            summary.loc[s_name, "chance_level"] = rate
            summary.loc[s_name, "grounding_lift"] = round(summary.loc[s_name, "grounded"] - rate, 3)
    summary.to_csv(path("results") / "generation_summary.csv")
    curve = st.groupby("system").apply(lambda g: pd.Series({f"grounded@{t:.2f}": (g["emb_support"] >= t).mean()
                                                            for t in np.round(np.arange(0.40, 0.81, 0.05), 2)}),
                                       include_groups=False).round(3)
    curve.to_csv(path("results") / "groundedness_threshold_curve.csv")
    verdict_dist = (st.dropna(subset=["verdict"]).groupby("system")["verdict"].value_counts(normalize=True)
                    .unstack(fill_value=0).round(3))
    verdict_dist.to_csv(path("results") / "verifier_verdicts.csv")
    print(summary.to_string())
    print(verdict_dist.to_string())

    # audit sheet for manual review by the team
    pool_st = st[st.system.isin(["rag", "llm_only"])]
    audit = pool_st.sample(min(60, len(pool_st)), random_state=CFG["seed"])
    audit_cols = audit.copy()
    audit_cols["evidence_shown_to_verifier"] = [" || ".join(stmt_rows[i]["candidates"]) for i in audit.index]
    audit_cols[["system", "category", "part", "statement", "evidence_shown_to_verifier", "supported", "extra", "emb_support"]] \
        .assign(manual_verdict="").to_csv(path("results") / "groundedness_audit.csv", index=False)

    examples = []
    for _, r in ev.iterrows():
        res = results.get((r.job_id, "rag"))
        if res and len(examples) < 6:
            examples.append({"title": r.title, "category": r.category, "jd": truncate_words(r.jd, 200),
                             "occupation": {"code": res["occupation"]["code"], "title": res["occupation"]["title"]},
                             "rag": res["output"], "llm_only": (results.get((r.job_id, "llm_only")) or {}).get("output"),
                             "evidence": res["chunks"][["chunk_id", "category", "resume_title", "section", "score"]]
                             .round(3).to_dict("records")})  # ids only: raw resume text is not republished
    (path("results") / "generation_examples.json").write_text(json.dumps(examples, indent=1, default=str))

    if args.no_judge:
        raise SystemExit

    # ---- phase D: pairwise LLM judge (different model family, randomised order) -----------
    judge = chat_model(judge_model, json_mode=True, max_tokens=700, with_fallback=False, seconds_per_call=15,
                       reasoning_effort="none")
    rng = random.Random(CFG["seed"])
    jrows = []
    for _, r in ev.iterrows():
        for a_sys, b_sys in (("rag", "baseline"), ("rag", "llm_only")):
            ga, gb = rendered.get((r.job_id, a_sys)), rendered.get((r.job_id, b_sys))
            if not ga or not gb:
                continue
            flip = rng.random() < 0.5
            a, b = (gb, ga) if flip else (ga, gb)
            try:
                msg = judge.invoke([SystemMessage(PROMPTS["judge_system"]),
                                    HumanMessage(fill(PROMPTS["judge_user"], jd=truncate_words(r.jd, 500), a=a, b=b))])
                v = json.loads(msg.content)
            except Exception as err:
                log.warning("judge failed: %s", str(err)[:160])
                continue
            sa, sb = (v["b"], v["a"]) if flip else (v["a"], v["b"])
            pref = v.get("preferred", "tie")
            winner = "tie" if pref == "tie" else {("a", False): a_sys, ("b", False): b_sys,
                                                    ("a", True): b_sys, ("b", True): a_sys}.get((pref, flip), "tie")
            for crit in ("specificity", "actionability", "alignment", "realism"):
                jrows.append({"job_id": r.job_id, "category": r.category, "pair": f"{a_sys} vs {b_sys}",
                              "criterion": crit, a_sys: sa.get(crit), b_sys: sb.get(crit)})
            jrows.append({"job_id": r.job_id, "category": r.category, "pair": f"{a_sys} vs {b_sys}",
                          "criterion": "preferred", "winner": winner, "reason": v.get("reason", ""), "flipped": flip})
    jdf = pd.DataFrame(jrows)
    jdf.to_csv(path("results") / "judge_eval.csv", index=False)
    if len(jdf):
        print(jdf[jdf.criterion == "preferred"].groupby("pair")["winner"].value_counts(normalize=True).round(3).to_string())
