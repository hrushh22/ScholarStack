# data/

Nothing in this folder is committed (see `.gitignore`); every file is reproduced by the pipeline.

| Folder | Created by | Contents |
|---|---|---|
| `raw/` | `pipeline/01_acquire.py` | Kaggle zips + extracted CSVs, O*NET 31.0 text files, `MANIFEST.json` with SHA-256 hashes |
| `interim/` | `pipeline/02_profile.py` | `postings_raw/`, the 517 MB postings CSV staged as Parquet parts for Dask |
| `processed/` | steps 03-07 | cleaned `resumes.parquet`, `postings/` (Dask Parquet), `onet_occupations.parquet`, skills, chunks, O*NET links, market stats, `eval_jds.parquet` |
| `models/` | steps 05-06 | embeddings (`emb_*.npy`) and FAISS indexes (`faiss_*.index`) |

Approximate disk use after a full run: 1.6 GB.

Licences: the Kaggle datasets are used under their Kaggle terms for research/education; O*NET is
CC BY 4.0 (U.S. Department of Labor, Employment & Training Administration). Raw resume text is
not redistributed.
