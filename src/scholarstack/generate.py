"""Reference-resume generation: RAG (retrieval-grounded) and LLM-only variants.

The pipeline is a LangChain runnable:
    {jd, title, resume?}
      -> retrieve (FAISS resume chunks + O*NET occupation + market demand)
      -> build messages (shared prompt file, identical to the live Edge Function)
      -> chat model (JSON mode, retries, fallback, cache)
      -> parse + validate citations
"""
from __future__ import annotations

import json
import re
import time

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

from .categories import pretty
from .config import CFG, ROOT
from .index import Retriever
from .llm import chat_model
from .onet import OnetMatcher, load_market_stats
from .query import build_query, jd_skills, split_title
from .skills import SkillExtractor

PROMPTS = json.loads((ROOT / "supabase" / "functions" / "_shared" / "prompts.json").read_text())


def fill(template: str, **values: str) -> str:
    """{{name}} substitution — the same rule the TypeScript Edge Function uses."""
    return re.sub(r"\{\{(\w+)\}\}", lambda m: str(values.get(m.group(1), "")), template)


def truncate_words(text: str, n: int) -> str:
    words = (text or "").split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


def format_evidence(chunks: pd.DataFrame) -> str:
    return "\n".join(
        f"[{r.chunk_id}] ({pretty(r.category)} | {r.resume_title} | {r.section}) {truncate_words(r.text, 110)}"
        for r in chunks.itertuples())


def format_onet(occ: pd.Series, market: dict | None) -> str:
    skills = sorted(occ["essential_skills"] + occ["transferable_skills"], key=lambda s: -s["importance"])[:8]
    tech = [t["name"] for t in occ["technologies"]][:12]
    zone = occ["job_zone"] if isinstance(occ["job_zone"], str) and occ["job_zone"] else "n/a"
    lines = [
        f"{occ['title']} (O*NET-SOC {occ['code']}, job zone {zone})",
        occ["description"],
        "Key skills: " + ", ".join(s["name"] for s in skills),
        "Knowledge: " + ", ".join(k["name"] for k in occ["knowledge"][:5]),
        "Technologies: " + ", ".join(tech),
        "Typical tasks: " + " | ".join(occ["tasks"][:4]),
    ]
    if market and market.get("top_skills"):
        top = ", ".join(f"{s['name']} ({round(100 * s['share'])}%)" for s in market["top_skills"][:10])
        lines.append(f"Demand across {market['n_postings']} LinkedIn postings: {top}")
    return "\n".join(lines)


def applicant_block(resume: str | None) -> str:
    return fill(PROMPTS["applicant_block"], resume=truncate_words(resume, 900)) if resume else ""


def statements(output: dict) -> list[dict]:
    """Every generated factual statement with its citations (used for groundedness)."""
    out = []
    for s in output.get("summary") or []:
        out.append({"part": "summary", "text": s.get("text", ""), "sources": s.get("sources") or []})
    for role in output.get("experience") or []:
        for b in role.get("bullets") or []:
            out.append({"part": "experience", "text": b.get("text", ""), "sources": b.get("sources") or []})
    for key in ("education", "certifications"):
        for s in output.get(key) or []:
            out.append({"part": key, "text": s.get("text", ""), "sources": s.get("sources") or []})
    return [s for s in out if s["text"].strip()]


def output_text(output: dict) -> str:
    parts = [output.get("headline", "")]
    parts += [s["text"] for s in statements(output)]
    for g in output.get("skills") or []:
        parts += [i.get("name", "") for i in g.get("items") or []]
    parts += [r.get("title", "") for r in output.get("experience") or []]
    return "\n".join(p for p in parts if p)


def validate_citations(output: dict, retrieved_ids: set[str]) -> int:
    """Drop citation ids that were not retrieved; return how many were dropped."""
    dropped = 0
    for st_list in [output.get("summary") or [], output.get("education") or [], output.get("certifications") or []] + \
                   [r.get("bullets") or [] for r in output.get("experience") or []]:
        for s in st_list:
            src = s.get("sources") or []
            keep = [x for x in src if x in retrieved_ids]
            dropped += len(src) - len(keep)
            s["sources"] = keep
    return dropped


class Generator:
    def __init__(self, retriever: Retriever | None = None, model: str | None = None, with_fallback: bool = True):
        self.ex = SkillExtractor()
        self.retriever = retriever or Retriever()
        self.onet = OnetMatcher()
        self.market = load_market_stats().set_index("code").to_dict("index")
        self.llm = chat_model(model, with_fallback=with_fallback)
        self.parser = JsonOutputParser()
        self.query_variant = CFG["retrieval"]["query_variant"]

    # -- retrieval ---------------------------------------------------------
    def retrieve(self, inp: dict) -> dict:
        title, _ = split_title(inp["jd"])
        title = inp.get("title") or title
        query = build_query(inp["jd"], self.ex, self.query_variant, title=title)
        k = inp.get("k", CFG["retrieval"]["k"])
        chunks = self.retriever.search(query, k=k, category=inp.get("category"))
        occ = self.onet.match(query, n=1).iloc[0]
        skills = self.ex.names(jd_skills(inp["jd"], self.ex))
        return {**inp, "title": title, "query": query, "chunks": chunks, "occupation": occ,
                "market": self.market.get(occ["code"]), "jd_skill_names": skills}

    # -- prompting ---------------------------------------------------------
    def messages(self, ctx: dict, mode: str, use_onet: bool = True):
        common = dict(title=ctx["title"] or "(not given)", jd=truncate_words(ctx["jd"], 900),
                      jd_skills=", ".join(ctx["jd_skill_names"]) or "(none detected)",
                      applicant=applicant_block(ctx.get("resume")), schema=PROMPTS["schema"])
        if mode == "rag":
            onet = format_onet(ctx["occupation"], ctx["market"]) if use_onet else "(not provided)"
            user = fill(PROMPTS["user_rag"], onet=onet, evidence=format_evidence(ctx["chunks"]), **common)
            return [SystemMessage(PROMPTS["system_rag"]), HumanMessage(user)]
        user = fill(PROMPTS["user_llm_only"], **common)
        return [SystemMessage(PROMPTS["system_llm_only"]), HumanMessage(user)]

    def complete(self, ctx: dict, mode: str = "rag", use_onet: bool = True) -> dict:
        """LLM step only (retrieval already done) -> new dict with output and bookkeeping."""
        t0 = time.perf_counter()
        msg = self.llm.invoke(self.messages(ctx, mode, use_onet))
        out = self.parser.invoke(msg)
        res = dict(ctx)
        res["latency_s"] = time.perf_counter() - t0
        res["usage"] = getattr(msg, "usage_metadata", None) or {}
        meta = getattr(msg, "response_metadata", None) or {}
        res["model"] = meta.get("model_name", "")
        # Groq reports its own processing time; wall-clock latency would include our rate-limiter waits.
        res["llm_time_s"] = (meta.get("token_usage") or {}).get("total_time")
        res["invalid_citations"] = validate_citations(out, set(ctx["chunks"]["chunk_id"])) if mode == "rag" else 0
        res["output"] = out
        res["mode"] = mode
        return res

    def chain(self, mode: str = "rag"):
        """retrieve -> prompt -> LLM -> parse/validate, as one LangChain runnable."""
        return RunnableLambda(self.retrieve) | RunnableLambda(lambda ctx: self.complete(ctx, mode))

    def generate(self, jd: str, resume: str | None = None, mode: str = "rag", **kw) -> dict:
        return self.chain(mode).invoke({"jd": jd, "resume": resume, **kw})
