"""Central configuration: reads config/config.yaml and the git-ignored .env file."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

with open(ROOT / "config" / "config.yaml") as fh:
    CFG: dict = yaml.safe_load(fh)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("scholarstack")
for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "urllib3", "distributed"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def path(key: str) -> Path:
    """Absolute path for a configured directory (created if missing)."""
    p = ROOT / CFG["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def env(name: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.getenv(name) or default
    if required and not value:
        raise SystemExit(f"Missing {name}. Add it to {ROOT / '.env'} (see .env.example).")
    return value


def seed_everything(seed: int | None = None) -> int:
    seed = CFG["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    return seed


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
