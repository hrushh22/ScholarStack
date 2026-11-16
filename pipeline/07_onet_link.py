"""Step 7 — Integrate sources: link postings to O*NET occupations (FAISS title
matching), aggregate labour-market demand per occupation, and build the
held-out evaluation set of job descriptions.

Output: data/processed/{posting_onet.parquet, onet_market_stats.parquet, eval_jds.parquet}
"""
from scholarstack.evalset import build_eval_set
from scholarstack.onet import link_postings, market_stats

if __name__ == "__main__":
    linked = link_postings()
    market_stats(linked)
    build_eval_set()
