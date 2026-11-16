"""Step 1 — Acquire & store: download resumes, LinkedIn postings and O*NET.

    python pipeline/01_acquire.py            # download to data/raw
    python pipeline/01_acquire.py --upload   # also mirror to RAW_STORAGE_URL via fsspec
"""
import sys

from scholarstack.acquire import acquire_all
from scholarstack.config import path
from scholarstack.storage import upload_raw

if __name__ == "__main__":
    acquire_all()
    if "--upload" in sys.argv:
        raw = path("raw")
        upload_raw([p for p in raw.rglob("*") if p.is_file() and p.suffix in {".csv", ".txt", ".json"}])
