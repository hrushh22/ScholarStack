"""Step 10 — Generation experiments (needs GROQ_API_KEY).

Systems compared on the held-out generation subset (one JD per category):
  baseline      TF-IDF keyword advice (no retrieval, no generation)
  llm_only      LLM prompted with the job description alone
  rag           full system: FAISS evidence + O*NET + LLM (k = 8)
  rag_no_onet   ablation: evidence without the O*NET profile
  rag_small     same as rag with the smaller fallback model
  rag_k3/k15    number of retrieved chunks
Metrics: groundedness (statements attested in real resumes), citation validity,
JD skill coverage, market (O*NET-linked) skill coverage, latency/tokens, and a
pairwise LLM-judge rating with randomised A/B order.

Every LLM response is cached, so the script can be stopped and re-run freely.
    python pipeline/10_eval_generation.py                 # all systems
    python pipeline/10_eval_generation.py --systems rag llm_only --limit 6
"""
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from scholarstack.baseline import fit_advice_model, keyword_advice
from scholarstack.config import CFG, env, log, path, seed_everything
from scholarstack.embed import encode
from scholarstack.evalset import load_eval_set
from scholarstack.generate import PROMPTS, Generator, fill, format_evidence, output_text, statements, truncate_words
from scholarstack.llm import chat_model
from scholarstack.metrics import citation_validity, groundedness, skill_coverage, statement_support
from scholarstack.query import jd_skills

TAUS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
POOL_K = 20  # shared evidence pool used to score every system the same way


def render(output: dict) -> str:
    """Plain-text rendering of a generated reference resume (for the judge and examples)."""
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*",
                    default=["baseline", "llm_only", "rag", "rag_no_onet", "rag_small", "rag_k3", "rag_k15"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    seed_everything()
    ev = load_eval_set()
    ev = ev[ev["gen_subset"]].reset_index(drop=True)
    if args.limit:
        ev = ev.head(args.limit)
    gen = Generator()
    small = Generator(retriever=gen.retriever, model=env("LLM_FALLBACK_MODEL", "llama-3.1-8b-instant"))
    ex = gen.ex
    advice_model = fit_advice_model()

    def enc(texts):
        return encode(texts, query=False)

    rows, examples, outputs = [], [], {}
    for i, jd_row in ev.iterrows():
        jd = jd_row["jd"]
        hard = jd_skills(jd, ex)
        pool = gen.retriever.search(gen.retrieve({"jd": jd})["query"], k=POOL_K)
        pool_vecs = enc(pool["text"].tolist())
        for system in args.systems:
            t0 = time.perf_counter()
            rec = {"job_id": jd_row["job_id"], "category": jd_row["category"], "title": jd_row["title"],
                   "system": system, "jd_hard_skills": len(hard)}
            try:
                if system == "baseline":
                    adv = keyword_advice(advice_model, jd)
                    text = " ; ".join(adv["keywords"])
                    rec.update({"n_statements": 0, "latency_s": time.perf_counter() - t0})
                    outputs[(jd_row["job_id"], system)] = render_baseline(adv)
                else:
                    if system == "llm_only":
                        res = gen.generate(jd, mode="llm_only")
                    elif system == "rag_small":
                        res = small.generate(jd, mode="rag")
                    elif system in ("rag_k3", "rag_k15"):
                        res = gen.generate(jd, mode="rag", k=int(system.split("k")[-1]))
                    elif system == "rag_no_onet":
                        ctx = gen.retrieve({"jd": jd})
                        common = dict(title=ctx["title"] or "(not given)", jd=truncate_words(jd, 900),
                                      jd_skills=", ".join(ctx["jd_skill_names"]), applicant="",
                                      schema=PROMPTS["schema"], onet="(not provided)",
                                      evidence=format_evidence(ctx["chunks"]))
                        msg = gen.llm.invoke([SystemMessage(PROMPTS["system_rag"]),
                                              HumanMessage(fill(PROMPTS["user_rag"], **common))])
                        out = gen.parser.invoke(msg)
                        from scholarstack.generate import validate_citations

                        validate_citations(out, set(ctx["chunks"]["chunk_id"]))
                        res = {**ctx, "output": out, "latency_s": time.perf_counter() - t0,
                               "usage": getattr(msg, "usage_metadata", None) or {}, "mode": "rag"}
                    else:
                        res = gen.generate(jd, mode="rag")
                    out = res["output"]
                    sts = statements(out)
                    text = output_text(out)
                    support = statement_support([s["text"] for s in sts], pool_vecs, enc)
                    for tau in TAUS:
                        rec[f"grounded@{tau}"] = groundedness(support, tau)
                    rec["grounded"] = groundedness(support)
                    rec["mean_support"] = float(support.mean()) if len(support) else 0.0
                    rec["n_statements"] = len(sts)
                    if res.get("mode") == "rag":
                        own = enc(res["chunks"]["text"].tolist())
                        rec["grounded_own_evidence"] = groundedness(statement_support([s["text"] for s in sts], own, enc))
                        rec.update(citation_validity([s["sources"] for s in sts], set(res["chunks"]["chunk_id"])))
                    rec["latency_s"] = res.get("latency_s", time.perf_counter() - t0)
                    usage = res.get("usage") or {}
                    rec["input_tokens"] = usage.get("input_tokens")
                    rec["output_tokens"] = usage.get("output_tokens")
                    outputs[(jd_row["job_id"], system)] = render(out)
                    if system == "rag" and len(examples) < 6:
                        examples.append({"title": jd_row["title"], "category": jd_row["category"],
                                         "jd": truncate_words(jd, 250), "rag": out,
                                         # ids/metadata only: raw resume text is not republished
                                         "evidence": res["chunks"][["chunk_id", "category", "resume_title",
                                                                    "section", "score"]].to_dict("records"),
                                         "occupation": {"code": res["occupation"]["code"],
                                                        "title": res["occupation"]["title"]}})
                out_sk = set(ex.extract(text))
                rec["jd_skill_coverage"] = skill_coverage(hard, out_sk)
                occ = gen.onet.match(gen.retrieve({"jd": jd})["query"], n=1).iloc[0]
                market = [s["id"] for s in (gen.market.get(occ["code"]) or {}).get("top_skills", [])[:10]]
                rec["market_skill_coverage"] = skill_coverage(market, out_sk)
            except Exception as err:  # keep going; failures are reported, not hidden
                log.warning("%s / %s failed: %s", jd_row["title"][:40], system, err)
                rec["error"] = str(err)[:300]
            rows.append(rec)
        log.info("[%d/%d] %s done", i + 1, len(ev), jd_row["title"][:60])

    df = pd.DataFrame(rows)
    df.to_csv(path("results") / "generation_eval.csv", index=False)
    metric_cols = [c for c in ["grounded", "grounded_own_evidence", "valid_citation_rate", "jd_skill_coverage",
                               "market_skill_coverage", "n_statements", "latency_s", "input_tokens",
                               "output_tokens"] if c in df]
    summary = df.groupby("system")[metric_cols].mean().round(3)
    summary["n"] = df.groupby("system").size()
    summary["errors"] = df.groupby("system")["error"].count() if "error" in df else 0
    summary.to_csv(path("results") / "generation_summary.csv")
    tau_curve = df.groupby("system")[[f"grounded@{t}" for t in TAUS if f"grounded@{t}" in df]].mean().round(3)
    tau_curve.to_csv(path("results") / "groundedness_threshold_curve.csv")
    (path("results") / "generation_examples.json").write_text(json.dumps(examples, indent=1, default=str))
    print(summary.to_string())

    # ---------------------------------------------------------------- LLM judge
    if args.no_judge:
        raise SystemExit
    judge = chat_model(json_mode=True, max_tokens=500)
    rng = random.Random(CFG["seed"])
    jrows = []
    pairs = [("rag", "baseline"), ("rag", "llm_only")]
    for _, jd_row in ev.iterrows():
        for sys_a, sys_b in pairs:
            ga, gb = outputs.get((jd_row["job_id"], sys_a)), outputs.get((jd_row["job_id"], sys_b))
            if not ga or not gb:
                continue
            flip = rng.random() < 0.5
            a, b = (gb, ga) if flip else (ga, gb)
            try:
                msg = judge.invoke([SystemMessage(PROMPTS["judge_system"]),
                                    HumanMessage(fill(PROMPTS["judge_user"], jd=truncate_words(jd_row["jd"], 600),
                                                      a=a, b=b))])
                verdict = json.loads(msg.content)
            except Exception as err:
                log.warning("judge failed: %s", err)
                continue
            sa, sb = (verdict["b"], verdict["a"]) if flip else (verdict["a"], verdict["b"])
            pref = verdict.get("preferred", "tie")
            winner = "tie" if pref == "tie" else ((sys_b if pref == "a" else sys_a) if flip else (sys_a if pref == "a" else sys_b))
            for crit in ("specificity", "actionability", "alignment", "realism"):
                jrows.append({"job_id": jd_row["job_id"], "category": jd_row["category"], "pair": f"{sys_a} vs {sys_b}",
                              "criterion": crit, sys_a: sa.get(crit), sys_b: sb.get(crit)})
            jrows.append({"job_id": jd_row["job_id"], "category": jd_row["category"], "pair": f"{sys_a} vs {sys_b}",
                          "criterion": "preferred", "winner": winner, "reason": verdict.get("reason", "")})
    jdf = pd.DataFrame(jrows)
    jdf.to_csv(path("results") / "judge_eval.csv", index=False)
    if len(jdf):
        pref = jdf[jdf.criterion == "preferred"].groupby("pair")["winner"].value_counts(normalize=True).round(3)
        print(pref.to_string())
