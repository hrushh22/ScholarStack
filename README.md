# ScholarStack

**Grounding resume advice in evidence: a retrieval-augmented generation system for job-specific reference resumes.**
BDA 696 — Processing and Modeling Big Data · San Diego State University

**Live demo:** https://hrushh22.github.io/ScholarStack/

Paste a job description and ScholarStack:

1. extracts the required skills (rule-based, aligned to O*NET),
2. embeds the job with a sentence-transformer and retrieves excerpts from resumes of people in similar roles,
3. matches the job to its O*NET occupation and to labour-market demand mined from 106k LinkedIn postings,
4. asks an LLM to write a **reference resume** in which every line must cite the excerpts it is based on,
5. verifies each statement against the evidence, and — if you paste your own resume — produces a **gap analysis**.

It is compared against a **TF-IDF keyword-advice baseline** and an **LLM with no retrieval**.

---

## Results at a glance

Held-out evaluation set: 240 LinkedIn job descriptions (10 per resume category), never used for tuning.

| Retrieval (k = 8) | Category precision | Skill-overlap precision |
|---|---|---|
| **MiniLM, section chunks, title + skills query (served)** | **61.6%** | 60.3% |
| GTE-small, section chunks | 61.8% | 59.2% |
| BGE-small, section chunks | 57.8% | 58.9% |
| MiniLM, fixed 120-word windows | 60.0% | 68.1% |
| TF-IDF baseline | 32.3% | 77.0% |
| MiniLM, *raw job description* as query | 56.5% | 47.6% |

* Dense retrieval almost **doubles** category precision over TF-IDF. TF-IDF wins the skill-overlap proxy because that
  proxy is lexical by construction (it rewards exact term matches).
* **Query construction mattered most**: embedding *title + extracted skills* beats embedding the raw ad
  (+5 points), because sentence-transformers truncate at 256 tokens and many ads open with company boilerplate.
* MiniLM matches GTE-small at ~5× the speed, so it is served (it also runs in the browser).
* FAISS on 106,291 posting vectors: exact search 1.96 ms/query; **HNSW 0.25 ms at 99.1% recall@10**.
* Generation results (groundedness, skill coverage, LLM-judge preference) are in
  [`reports/results/generation_summary.csv`](reports/results) and on the site's *Evaluation* tab.

All numbers are reproduced by the pipeline below and shown interactively on the site's *Evaluation* tab.

---

## Architecture

```
                OFFLINE (Python, this repo)                                  ONLINE (free hosting)
 ┌──────────────────────────────────────────────────────────┐   ┌───────────────────────────────────────────────┐
 │ Kaggle resumes ─┐                                        │   │ GitHub Pages (static site, web/)               │
 │ LinkedIn posts ─┼─ acquire → profile → clean (Dask)      │   │   • skill extraction   (same rules as Python)  │
 │ O*NET 31.0 ─────┘        → skills (O*NET-aligned)        │   │   • query embedding    (MiniLM, transformers.js)│
 │      → chunk (section | fixed) → embed (3 models)        │   │   • TF-IDF baseline, grounding check            │
 │      → FAISS indexes + benchmark                         │   │                 │                               │
 │      → postings ↔ O*NET linking, market demand           │   │ Supabase Postgres + pgvector (HNSW)             │
 │      → experiments: retrieval / generation / LLM judge   │──▶│   resume_chunks, onet_occupations, runs,        │
 │      → figures, web assets, database load                │   │   feedback  (row-level security)                │
 └──────────────────────────────────────────────────────────┘   │ Supabase Edge Function `generate`               │
                                                                │   prompt (shared prompts.json) → Groq LLM        │
                                                                └───────────────────────────────────────────────┘
```

* The **same prompts** (`supabase/functions/_shared/prompts.json`) drive the offline evaluation (LangChain) and the
  live Edge Function, so what is evaluated is what is served.
* The browser ports of skill extraction, query construction and the TF-IDF baseline were checked against
  Python on 100 held-out job descriptions: identical skills and queries; embeddings identical (cosine 1.0000).
* API keys stay server-side (Edge Function secrets). Raw resume text is never committed to the repository.

## Repository layout

```
config/          config.yaml (every model id, path, seed, k …), curated skills, sample resumes
src/scholarstack/   library: acquire, clean (Dask), skills, chunking, embed, index (FAISS), onet,
                    baseline (TF-IDF), llm + generate (LangChain), metrics, evalset, viz
pipeline/        numbered steps 01–14 — run in order (see below)
supabase/        migrations/0001_schema.sql (pgvector schema + RPCs), functions/generate (Edge Function)
web/             the static website (vanilla JS modules, no build step)
reports/         figures/ (EDA + results) and results/ (CSV/JSON of every experiment)
docs/            presentation guide and report notes
data/            created by the pipeline, git-ignored (see data/README.md)
```

## Reproduce the pipeline

Requirements: Python 3.12, ~2 GB disk, ~20 min on a laptop (Apple-silicon GPU or CUDA used if available).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env            # fill in keys (only needed for steps 10–12)

./run_pipeline.sh               # steps 01–09, 13, 14 (no API keys needed)
./run_pipeline.sh --with-llm    # also step 10, the generation experiments (needs GROQ_API_KEY)
```

| Step | Script | What it does |
|---|---|---|
| 01 | `01_acquire.py` | Scripted download of both Kaggle datasets and O*NET 31.0; SHA-256 manifest. `--upload` mirrors raw files to S3-compatible storage via fsspec |
| 02 | `02_profile.py` | Stages the 517 MB postings CSV to Parquet; profiles schema, nulls, duplicates, length outliers |
| 03 | `03_clean.py` | Parses resume HTML into sections; cleans/deduplicates 123,849 → 106,291 postings with a Dask cluster; builds O*NET occupation profiles |
| 04 | `04_skills.py` | Builds a 1,457-skill lexicon (O*NET software skills + curated domain skills) and extracts skills from every document |
| 05 | `05_chunk_embed.py` | Section-based vs fixed-window chunks; embeddings with MiniLM-L6, BGE-small, GTE-small |
| 06 | `06_index.py` | FAISS indexes over resume chunks and all postings; Flat vs HNSW vs IVF benchmark |
| 07 | `07_onet_link.py` | Links postings to O*NET by title (FAISS over 54k O*NET titles); demand per occupation; builds the held-out evaluation set |
| 08 | `08_eval_retrieval.py` | Query construction × chunking × embedding model × k, vs TF-IDF; JD→O*NET linking accuracy |
| 09 | `09_eda.py` | EDA figures (categories, lengths, missingness, industries, skills, t-SNE embedding space) |
| 10 | `10_eval_generation.py` | RAG vs LLM-only vs TF-IDF; ablations (no O*NET, 8B model, k = 3/15); groundedness, skill coverage, LLM judge |
| 11 | `11_load_database.py` | Applies the schema and bulk-loads chunks + O*NET into Supabase; checks pgvector top-k = FAISS top-k |
| 12 | `12_deploy_supabase.py` | Sets Edge Function secrets, deploys `generate`, writes the browser config |
| 13 | `13_export_web.py` | Exports lexicon, TF-IDF model, samples, results and figures for the website |
| 14 | `14_report_figures.py` | Results figures for the report and slides |

## Deploy (all free tiers)

1. **Groq** — create an API key at https://console.groq.com → `GROQ_API_KEY` in `.env`.
2. **Supabase** — create a project at https://supabase.com, then fill `.env`:
   `SUPABASE_URL` (Project Settings → Data API), `SUPABASE_DB_URL` (Connect → Session pooler URI, with your DB password),
   `SUPABASE_ACCESS_TOKEN` (Account → Access tokens).
3. Load and deploy:
   ```bash
   python pipeline/11_load_database.py
   python pipeline/12_deploy_supabase.py
   git add web/js/config.js && git commit -m "Configure live backend" && git push
   ```
4. **GitHub Pages** — repository Settings → Pages → Source: **GitHub Actions**. The workflow in
   `.github/workflows/pages.yml` publishes `web/` on every push.

To preview the site locally: `cd web && python -m http.server 8000` and open http://localhost:8000.

## Evaluation design

| Measure | How it is computed |
|---|---|
| Retrieval relevance (precision@k) | No gold labels exist, so two proxies: the excerpt's resume is in the job's category; the excerpt shares ≥ 1 hard skill with the job |
| Groundedness | Share of generated statements whose embedding has cosine ≥ 0.55 with a real resume excerpt retrieved for that job (same evidence pool for every system); threshold sensitivity reported |
| Citation validity | Share of statements whose cited ids were actually retrieved (invalid ids are removed live) |
| Skill coverage | Share of the job's hard skills (and of the matched occupation's top-10 in-demand skills) present in the output |
| Output quality | Pairwise LLM judge (specificity, actionability, alignment, realism) with randomised A/B order; plus live 1–5 human ratings stored in Supabase |
| Time & scale | Embedding throughput, FAISS build/query time, end-to-end latency |

## Data and licences

| Dataset | Source | Use |
|---|---|---|
| Resume Dataset (2,484 LiveCareer resumes, 24 categories) | [Kaggle: snehaanbhawal/resume-dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) | Evidence corpus |
| LinkedIn Job Postings 2023–24 (123,849 postings) | [Kaggle: arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | Job descriptions, skill demand, evaluation set |
| O*NET 31.0 Database | [National Center for O*NET Development](https://www.onetcenter.org/database.html) | Occupation skills, knowledge, tasks, technologies |

O*NET® is a trademark of the U.S. Department of Labor, Employment and Training Administration (USDOL/ETA), used under the
CC BY 4.0 licence. Raw resume text is not redistributed in this repository.

## Limitations and responsible use

* Resumes are example documents from one website, not verified employees; categories are keyword-based and broad.
* Relevance labels are proxies; groundedness uses embedding similarity and can accept paraphrases that add small details.
* Postings are a 2023–24 snapshot; O*NET describes occupations rather than individual jobs.
* A reference resume is an example to learn from. Applicants should only claim experience they actually have.

## Team

Sonika Srinivas (group leader) · Hrushik Nimesh Mehta · Dhrumil Vinay Panchal · Parjanya Hebbale Krishna

## References

Lewis et al. (2020), *Retrieval-augmented generation for knowledge-intensive NLP tasks*, NeurIPS ·
Reimers & Gurevych (2019), *Sentence-BERT*, EMNLP · Johnson, Douze & Jégou (2021), *Billion-scale similarity search with GPUs*,
IEEE TBD · Spärck Jones (1972), *A statistical interpretation of term specificity* · Dask Development Team (2016) ·
National Center for O*NET Development, *O*NET 31.0 Database*.
