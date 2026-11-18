# Final paper: where everything lives

A map from each section of the *BDA696 Final Project Paper Outline* to the code, figures and numbers that fill it.
All figures are in `reports/figures/`, all tables in `reports/results/`. Re-running the pipeline regenerates them.

| Outline section | What to write | Source in this repo |
|---|---|---|
| **4. Data sources & acquisition** | Three sources, sizes, scripted download, SHA-256 manifest, fsspec storage | `pipeline/01_acquire.py`, `data/raw/MANIFEST.json`, `src/scholarstack/storage.py` |
| **5. Data processing** | Resume HTML to sections; postings CSV to Parquet to Dask clean/dedup (123,849 down to 106,291); O*NET profiles; 1,457-skill lexicon; chunking (27,207 section vs 21,888 fixed chunks) | `src/scholarstack/clean.py`, `skills.py`, `chunking.py`; `reports/results/data_profile.json` |
| **5. Missing data / quality** | Salary missing for 71% of postings; 2 duplicate + 2 very short resumes removed; 17,558 postings removed (short, non-English, duplicate) | `reports/results/data_profile.json`, `eda_summary.json`; figure `eda_postings_missingness.png` |
| **6. EDA** | Category imbalance (5.5×), length distributions, industries/work types, skills in resumes vs postings, embedding space (10-NN purity 0.65) | figures `eda_*.png`; `reports/results/top_skills_by_category.csv`, `eda_summary.json` |
| **7. Methodology & architecture** | Offline pipeline + live system; query = title + extracted skills; pgvector serving; shared prompts; citation validation | `README.md` (architecture), `supabase/functions/_shared/prompts.json`, `src/scholarstack/generate.py`, `supabase/functions/generate/index.ts`, site to *How it works* |
| **8. Baseline & comparisons** | TF-IDF keyword advice; LLM-only; ablations (O*NET, k, model size, chunking, embedding model, query construction) | `src/scholarstack/baseline.py`, `pipeline/08_eval_retrieval.py`, `pipeline/10_eval_generation.py` |
| **9. Evaluation methodology** | Proxy relevance labels (category, skill overlap); held-out set (240 JDs, 24 for generation); claim-level verification with an independent LLM; negative control; calibration of the live embedding check; LLM judge with randomized order; human ratings on the site | `src/scholarstack/evalset.py`, `metrics.py`, `pipeline/10_eval_generation.py`, `reports/results/groundedness_calibration.json` |
| **10. Results, retrieval** | Precision@k by configuration; query construction; model × chunking; per-category; FAISS benchmark; O*NET linking | `retrieval_eval.csv`, `retrieval_by_category.csv`, `faiss_benchmark.csv`, `onet_linking_eval.json`; figures `results_retrieval_k_sweep.png`, `results_query_construction.png`, `results_model_chunking.png`, `results_retrieval_by_category.png`, `results_faiss_benchmark.png` |
| **10. Results, generation** | Verified groundedness, citation correctness, skill coverage, judge preference, latency/tokens | `generation_summary.csv`, `verifier_verdicts.csv`, `judge_eval.csv`, `generation_eval.csv`; figures `results_generation.png`, `results_judge.png` |
| **10. Example outputs** | Side-by-side RAG vs LLM-only for selected jobs (incl. failure cases) | `reports/results/generation_examples.json`, live site to *Compare with baselines* |
| **11. Limitations** | Proxy labels; resumes ≠ verified employees; broad keyword categories (BPO 0% P@8); verifier LLM bias; 2023-24 snapshot | site to *How it works*; `retrieval_by_category.csv` |
| **15. Appendix / reproducibility** | Pinned requirements, config file, seeds, step-by-step commands, prompts | `requirements.txt`, `config/config.yaml`, `run_pipeline.sh`, `README.md` |

## Numbers worth quoting (retrieval, held-out 240 JDs)

* Dense retrieval (MiniLM, section chunks, title + skills query): **category P@8 = 61.6%** vs **32.3% for TF-IDF**.
* Query construction: title + extracted skills **61.6%** vs raw job ad **56.5%** (skill-overlap 60.3% vs 47.6%); embedding it is also 13× faster (0.8 ms vs 11.3 ms).
* Embedding models at k = 8: MiniLM 61.6%, GTE-small 61.8%, BGE-small 57.8%, MiniLM served for speed (6× faster per query, 0.8 ms vs 5.0 ms) and browser support.
* Chunking: section 61.6% vs fixed windows 60.0% (fixed windows score higher on skill overlap because they are longer: 115 vs 70 words).
* Easiest categories: Chef, Fitness (95%), Construction (88%); hardest: BPO (0%), Automobile (16%), Arts (29%), categories defined by
  broad keywords whose resumes span many occupations.
* FAISS on 106,291 posting vectors: exact 1.96 ms/query; HNSW 0.25 ms at 99.1% recall@10; IVF 0.10 ms at 94.0%.
* JD to O*NET occupation: top-3 exact 49%, same SOC major group 53% (vs. title-based linking).
* 87.3% of postings linked to an O*NET occupation; demand statistics for 978 occupations.

## Honest caveats to mention

* Relevance labels are proxies; the skill-overlap proxy rewards lexical matching, which is why TF-IDF scores high on it.
* The generation subset is small (24 JDs, one per category); report it as indicative, with per-job results in `generation_eval.csv`.
* The verifier and judge are LLMs (a different family from the generator to limit self-preference); the audit sheet
  `groundedness_audit.csv` has an empty `manual_verdict` column for the team to check a sample by hand.
