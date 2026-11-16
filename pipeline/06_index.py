"""Step 6 - Store & index: FAISS indexes over resume chunks and all job postings,
plus a benchmark of exact vs. approximate index types (build time, latency, recall).

Output: data/models/faiss_*.index, data/models/emb_postings_*.npy,
        reports/results/faiss_benchmark.csv
"""
import time

import numpy as np
import pandas as pd

from scholarstack.config import CFG, log, path, seed_everything
from scholarstack.embed import encode, load_embeddings, slug
from scholarstack.index import benchmark, build_flat, build_hnsw, save_index

if __name__ == "__main__":
    seed_everything()
    rng = np.random.default_rng(CFG["seed"])
    model = CFG["embedding"]["serving_model"]

    # Serving index: section chunks embedded with the serving model.
    chunks = load_embeddings(f"chunks_{CFG['chunking']['serving_strategy']}", model)
    save_index(build_flat(chunks), "chunks_section")

    # Embed every cleaned posting (title + description, truncated to the model's max length).
    postings = pd.read_parquet(path("processed") / "postings", columns=["job_id", "title", "description"])
    out = path("models") / f"emb_postings_{slug(model)}.npy"
    if out.exists():
        p_vecs = np.load(out)
    else:
        t0 = time.perf_counter()
        p_vecs = encode((postings["title"] + ". " + postings["description"].str[:1500]).tolist(), model,
                        batch_size=256, show_progress=True)
        np.save(out, p_vecs)
        log.info("embedded %d postings in %.0fs", len(p_vecs), time.perf_counter() - t0)
    postings[["job_id"]].to_parquet(path("models") / "postings_index_ids.parquet", index=False)
    save_index(build_hnsw(p_vecs), "postings_hnsw")

    # Benchmark on both corpora with 500 held-out style queries (random postings).
    queries = p_vecs[rng.choice(len(p_vecs), 500, replace=False)]
    results = pd.concat([
        benchmark(chunks, queries).assign(corpus="resume chunks"),
        benchmark(p_vecs, queries).assign(corpus="job postings"),
    ])
    results.to_csv(path("results") / "faiss_benchmark.csv", index=False)
    print(results.to_string(index=False))
