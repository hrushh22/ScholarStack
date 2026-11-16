"""Step 11 — Load the serving database (Supabase Postgres + pgvector).

Applies supabase/migrations/*.sql, then bulk-loads (COPY) the resume chunks with
their embeddings and the O*NET occupation profiles with LinkedIn demand stats.
Re-running replaces the data. Needs SUPABASE_DB_URL in .env.
"""
import json

import numpy as np
import pandas as pd
import psycopg

from scholarstack.chunking import load_chunks
from scholarstack.clean import load_onet_profiles
from scholarstack.config import CFG, ROOT, env, log
from scholarstack.embed import load_embeddings
from scholarstack.onet import load_market_stats
from scholarstack.skills import SkillExtractor


def normalise_dsn(dsn: str) -> str:
    """Percent-encode the password if it was pasted raw (special characters break the URI)."""
    import re
    import urllib.parse

    m = re.match(r"^(postgres(?:ql)?://)([^:/]+):(.*)@([^@]+)$", dsn.strip())
    if not m:
        return dsn.strip()
    scheme, user, password, rest = m.groups()
    if re.search(r"%[0-9A-Fa-f]{2}", password):  # already encoded
        return dsn.strip()
    return f"{scheme}{user}:{urllib.parse.quote(password, safe='')}@{rest}"


def vec_literal(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


if __name__ == "__main__":
    dsn = normalise_dsn(env("SUPABASE_DB_URL", required=True))
    strategy, model = CFG["chunking"]["serving_strategy"], CFG["embedding"]["serving_model"]
    chunks = load_chunks(strategy)
    vecs = load_embeddings(f"chunks_{strategy}", model)
    onet = load_onet_profiles()
    onet_vecs = load_embeddings("onet_profiles", model)
    market = load_market_stats().set_index("code")
    ex = SkillExtractor()

    with psycopg.connect(dsn, autocommit=False, prepare_threshold=None) as conn, conn.cursor() as cur:
        for sql_file in sorted((ROOT / "supabase" / "migrations").glob("*.sql")):
            log.info("applying %s", sql_file.name)
            cur.execute(sql_file.read_text())
        conn.commit()

        cur.execute("truncate public.resume_chunks")
        with cur.copy("copy public.resume_chunks (chunk_id, resume_id, category, resume_title, section, "
                      "content, skills, embedding) from stdin") as cp:
            for r, v in zip(chunks.itertuples(), vecs):
                cp.write_row((r.chunk_id, r.resume_id, r.category, r.resume_title, r.section, r.text,
                              ex.names(ex.extract(r.text))[:15], vec_literal(v)))
        log.info("loaded %d resume chunks", len(chunks))

        cur.execute("truncate public.onet_occupations")
        with cur.copy("copy public.onet_occupations (code, title, description, job_zone, essential_skills, "
                      "transferable_skills, knowledge, tasks, technologies, market, embedding) from stdin") as cp:
            for r, v in zip(onet.itertuples(), onet_vecs):
                if not r.has_profile:
                    continue
                m = market.loc[r.code].to_dict() if r.code in market.index else None
                if m:
                    m = {"n_postings": int(m["n_postings"]), "top_skills": m["top_skills"][:12],
                         "median_salary": None if pd.isna(m["median_salary"]) else m["median_salary"],
                         "remote_share": m["remote_share"], "experience_levels": m["experience_levels"],
                         "top_titles": m["top_titles"]}
                cp.write_row((r.code, r.title, r.description, int(r.job_zone) if isinstance(r.job_zone, str) else None,
                              json.dumps(r.essential_skills), json.dumps(r.transferable_skills),
                              json.dumps(r.knowledge), json.dumps(r.tasks), json.dumps(r.technologies),
                              json.dumps(m) if m else None, vec_literal(v)))
        log.info("loaded %d O*NET occupations", int(onet["has_profile"].sum()))
        conn.commit()

        cur.execute("analyze public.resume_chunks; analyze public.onet_occupations;")
        conn.commit()

        # Parity check: pgvector must return the same top-k as the offline FAISS index.
        from scholarstack.index import build_flat

        index = build_flat(vecs)
        rng = np.random.default_rng(CFG["seed"])
        agree = []
        for qi in rng.choice(len(vecs), 20, replace=False):
            _, ids = index.search(vecs[qi:qi + 1], 8)
            cur.execute("select chunk_id from public.match_resume_chunks(%s::extensions.vector, 8, null)",
                        (vec_literal(vecs[qi]),))
            got = {row[0] for row in cur.fetchall()}
            agree.append(len(got & set(chunks["chunk_id"].iloc[ids[0]])) / 8)
        log.info("pgvector (HNSW) vs FAISS (exact) top-8 overlap: %.1f%%", 100 * np.mean(agree))
