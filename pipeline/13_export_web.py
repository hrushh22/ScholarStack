"""Step 13 — Export everything the static web app needs into web/data/.

  skills_lexicon.json   lexicon for live skill extraction (same rules as Python)
  tfidf.json            vocabulary + IDF for the live TF-IDF baseline
  samples.json          demo job descriptions + fictional applicant resumes
  results.json          every evaluation number shown on the dashboard
  web/figures/*.png     EDA and results figures
  web/js/config.js      public Supabase settings (URL + publishable key only)
"""
import json
import re
import shutil

import pandas as pd
import yaml

from scholarstack.baseline import export_for_browser, fit_advice_model
from scholarstack.categories import CATEGORY_RULES, pretty
from scholarstack.config import CFG, ROOT, env, log, path
from scholarstack.skills import SkillExtractor, load_lexicon

DEMO_ROLES = [
    ("Registered Nurse", r"^registered nurse\b", "HEALTHCARE"),
    ("Staff Accountant", r"^staff accountant$", "ACCOUNTANT"),
    ("Data Analyst", r"^data analyst$", "INFORMATION-TECHNOLOGY"),
    ("Software Engineer", r"^software engineer$", "ENGINEERING"),
    ("IT Support Specialist", r"^(?:it support|help desk|desktop support)", "INFORMATION-TECHNOLOGY"),
    ("HR Generalist", r"^(?:hr|human resources) generalist$", "HR"),
    ("Graphic Designer", r"^graphic designer$", "DESIGNER"),
    ("Executive Chef", r"^(?:executive chef|sous chef)$", "CHEF"),
    ("Elementary Teacher", r"^(?:elementary|kindergarten|preschool).{0,20}teacher", "TEACHER"),
    ("Sales Representative", r"^(?:outside |inside )?sales representative$", "SALES"),
    ("Business Development Manager", r"^business development manager$", "BUSINESS-DEVELOPMENT"),
    ("Financial Analyst", r"^(?:senior )?financial analyst$", "FINANCE"),
    ("Digital Marketing Specialist", r"^digital marketing (?:specialist|coordinator|manager)$", "DIGITAL-MEDIA"),
    ("Public Relations Specialist", r"^(?:public relations|pr|communications) (?:specialist|coordinator|manager)$", "PUBLIC-RELATIONS"),
    ("Construction Project Manager", r"^construction (?:project )?manager$", "CONSTRUCTION"),
    ("Aircraft Maintenance Technician", r"^(?:aircraft|a&p|aviation maintenance).{0,25}(?:technician|mechanic)", "AVIATION"),
    ("Bank Teller", r"^(?:bank )?teller\b", "BANKING"),
    ("Customer Service Representative", r"^customer service representative$", "BPO"),
    ("Personal Trainer", r"^(?:certified )?personal trainer$", "FITNESS"),
    ("Automotive Technician", r"^(?:automotive|auto) (?:service )?(?:technician|mechanic)$", "AUTOMOBILE"),
]


def pick_demo_jds(ex: SkillExtractor) -> list[dict]:
    p = pd.read_parquet(path("processed") / "postings",
                        columns=["job_id", "title", "company_name", "location", "description", "n_words",
                                 "formatted_experience_level"])
    held_out = set(pd.read_parquet(path("processed") / "eval_jds.parquet")["job_id"])
    p = p[~p["job_id"].isin(held_out) & p["n_words"].between(220, 650)]
    demos = []
    for label, rx, cat in DEMO_ROLES:
        cand = p[p["title"].str.strip().str.lower().str.contains(rx, regex=True)].copy()
        if cand.empty:
            log.warning("no demo JD for %s", label)
            continue
        cand["n_skills"] = cand["description"].map(lambda d: len([s for s in ex.extract(d)
                                                                  if ex.by_id[s]["kind"] != "soft"]))
        cand["kw"] = cand["description"].map(lambda d: len(re.findall(CATEGORY_RULES[cat], d, flags=re.I)))
        best = cand.sort_values(["n_skills", "job_id"], ascending=[False, True]).head(5).iloc[0]
        clean = lambda v: None if pd.isna(v) else str(v)  # noqa: E731
        demos.append({"label": label, "category": cat, "title": best["title"].strip(),
                      "company": clean(best["company_name"]), "location": clean(best["location"]),
                      "experience": clean(best["formatted_experience_level"]), "job_id": int(best["job_id"]),
                      "description": best["description"]})
    return demos


def csv_records(name: str) -> list[dict]:
    f = path("results") / name
    return json.loads(pd.read_csv(f).to_json(orient="records")) if f.exists() else []


def json_file(name: str):
    f = path("results") / name
    return json.loads(f.read_text()) if f.exists() else None


def live_threshold() -> float:
    """Embedding threshold calibrated against the LLM verifier (step 10), else the config default."""
    calib = json_file("groundedness_calibration.json") or {}
    return calib.get("embedding_tau", CFG["evaluation"]["groundedness_threshold"])


if __name__ == "__main__":
    web = path("web_data")
    lexicon = load_lexicon()
    (web / "skills_lexicon.json").write_text(json.dumps(
        [{k: e[k] for k in ("id", "name", "group", "kind", "aliases", "cs")} for e in lexicon],
        separators=(",", ":")))
    export_for_browser(fit_advice_model(), web / "tfidf.json")

    ex = SkillExtractor()
    samples = {"jds": pick_demo_jds(ex),
               "resumes": yaml.safe_load((ROOT / "config" / "sample_resumes.yaml").read_text())}
    (web / "samples.json").write_text(json.dumps(samples, indent=1))
    log.info("demo job descriptions: %s", [d["label"] for d in samples["jds"]])

    results = {
        "eda": json_file("eda_summary.json"),
        "profile": {k: {"rows": v["rows"]} for k, v in (json_file("data_profile.json") or {}).items()
                    if isinstance(v, dict) and "rows" in v},
        "retrieval": csv_records("retrieval_eval.csv"),
        "retrieval_by_category": csv_records("retrieval_by_category.csv"),
        "faiss": csv_records("faiss_benchmark.csv"),
        "embedding_timing": csv_records("embedding_timing.csv"),
        "onet_linking": json_file("onet_linking_eval.json"),
        "generation": csv_records("generation_summary.csv"),
        "generation_by_category": csv_records("generation_eval.csv"),
        "groundedness_curve": csv_records("groundedness_threshold_curve.csv"),
        "judge": csv_records("judge_eval.csv"),
        "verifier": csv_records("verifier_verdicts.csv"),
        "calibration": json_file("groundedness_calibration.json"),
        "examples": json_file("generation_examples.json"),
        "config": {"embedding_model": CFG["embedding"]["serving_model"], "chunking": CFG["chunking"]["serving_strategy"],
                   "k": CFG["retrieval"]["k"], "query_variant": CFG["retrieval"].get("query_variant", "hybrid"),
                   "groundedness_threshold": live_threshold(),
                   "llm_model": env("LLM_MODEL", "openai/gpt-oss-120b"),
                   "llm_fallback_model": env("LLM_FALLBACK_MODEL", "openai/gpt-oss-20b"),
                   "judge_model": env("JUDGE_MODEL", "qwen/qwen3.8-27b")},
        "categories": {c: pretty(c) for c in sorted(CATEGORY_RULES)},
    }
    (web / "results.json").write_text(json.dumps(results, separators=(",", ":"), default=str))

    figs = ROOT / "web" / "figures"
    figs.mkdir(exist_ok=True)
    for f in path("figures").glob("*.png"):
        shutil.copy(f, figs / f.name)

    cfg = {
        "supabaseUrl": env("SUPABASE_URL", ""),
        "supabaseKey": env("SUPABASE_PUBLISHABLE_KEY", ""),
        "embeddingModel": CFG["embedding"]["browser_model"],
        "embeddingDtype": "fp32",
        "queryVariant": CFG["retrieval"].get("query_variant", "hybrid"),
        "k": CFG["retrieval"]["k"],
        "groundednessThreshold": live_threshold(),
    }
    (ROOT / "web" / "js" / "config.js").write_text(
        "// Generated by pipeline/13_export_web.py. Only public values: the Supabase URL and the\n"
        "// publishable (anon) key are designed to be used in browsers; data access is limited by\n"
        "// row-level security and the RPC functions in supabase/migrations.\n"
        f"export const CONFIG = {json.dumps(cfg, indent=2)};\n")
    log.info("web assets written to %s", web)
