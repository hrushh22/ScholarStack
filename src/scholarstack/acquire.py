"""Scripted data acquisition (no manual downloads).

* Kaggle datasets are fetched through Kaggle's public download endpoint, which
  works without an account for public datasets.
* O*NET is fetched as the official tab-delimited text bundle.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .config import CFG, log, path

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/{slug}"


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("already downloaded: %s", dest.name)
        return dest
    log.info("downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
    tmp.rename(dest)
    return dest


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def acquire_all() -> dict:
    raw = path("raw")
    src = CFG["sources"]
    manifest = {"downloaded_at": datetime.now(timezone.utc).isoformat(), "files": {}}

    # 1) Resume dataset — only the CSV is needed (the zip also holds ~2.4k PDFs).
    z = _download(KAGGLE_URL.format(slug=src["resumes"]["kaggle"]), raw / "resume-dataset.zip")
    with zipfile.ZipFile(z) as zf:
        zf.extract("Resume/Resume.csv", raw / "resumes")
    manifest["files"]["resume-dataset.zip"] = _sha256(z)

    # 2) LinkedIn job postings.
    z = _download(KAGGLE_URL.format(slug=src["postings"]["kaggle"]), raw / "linkedin-job-postings.zip")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(raw / "linkedin")
    manifest["files"]["linkedin-job-postings.zip"] = _sha256(z)

    # 3) O*NET database (text bundle).
    version = src["onet"]["version"]
    z = _download(src["onet"]["url"], raw / f"onet_{version.replace('.', '_')}_text.zip")
    with zipfile.ZipFile(z) as zf:
        top = zf.namelist()[0].split("/")[0]
        zf.extractall(raw)
    target = raw / "onet"
    if target.exists():
        shutil.rmtree(target)
    (raw / top).rename(target)
    manifest["files"][z.name] = _sha256(z)
    manifest["onet_version"] = version

    (raw / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    log.info("raw data ready in %s", raw)
    return manifest
