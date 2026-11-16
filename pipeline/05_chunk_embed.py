"""Step 5 — Chunk & embed: build section-based and fixed-length chunks, then embed
them with every candidate sentence-transformer (batched, GPU/MPS when available).

Output: data/processed/chunks_{section,fixed}.parquet, data/models/emb_*.npy,
        reports/results/embedding_timing.csv
"""
import pandas as pd

from scholarstack.chunking import build_chunks
from scholarstack.clean import load_onet_profiles
from scholarstack.config import CFG, path, seed_everything
from scholarstack.embed import embed_and_save, slug

if __name__ == "__main__":
    seed_everything()
    chunks = build_chunks()
    onet = load_onet_profiles()
    timing = []
    for model in CFG["embedding"]["candidates"]:
        for strategy, df in chunks.items():
            _, secs = embed_and_save(df["embed_text"].tolist(), f"chunks_{strategy}", model)
            timing.append({"what": f"resume chunks ({strategy})", "model": slug(model), "n": len(df), "seconds": secs})
        _, secs = embed_and_save(onet["profile_text"].tolist(), "onet_profiles", model)
        timing.append({"what": "O*NET profiles", "model": slug(model), "n": len(onet), "seconds": secs})
    pd.DataFrame(timing).assign(per_sec=lambda d: d.n / d.seconds).round(2).to_csv(
        path("results") / "embedding_timing.csv", index=False)
