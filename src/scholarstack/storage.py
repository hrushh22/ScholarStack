"""fsspec-backed storage so raw data can live locally or in S3-compatible cloud storage.

By default everything is read from `data/raw`. Setting RAW_STORAGE_URL (e.g. an
S3 bucket on Supabase Storage) makes `raw_url()` point there instead, and
`upload_raw()` mirrors the local raw files to it.
"""
from __future__ import annotations

from pathlib import Path

import fsspec

from .config import env, log, path


def storage_options() -> dict:
    if not env("RAW_STORAGE_URL"):
        return {}
    opts = {"key": env("RAW_STORAGE_KEY"), "secret": env("RAW_STORAGE_SECRET")}
    if env("RAW_STORAGE_ENDPOINT"):
        opts["client_kwargs"] = {"endpoint_url": env("RAW_STORAGE_ENDPOINT")}
    return opts


def raw_url(relative: str) -> str:
    """URL of a raw file, local (file://) or remote (s3://...)."""
    base = env("RAW_STORAGE_URL")
    if base:
        return f"{base.rstrip('/')}/{relative}"
    return str(path("raw") / relative)


def open_raw(relative: str, mode: str = "rb"):
    return fsspec.open(raw_url(relative), mode, **storage_options()).open()


def upload_raw(files: list[Path]) -> None:
    base = env("RAW_STORAGE_URL")
    if not base:
        log.info("RAW_STORAGE_URL not set - keeping raw data local only.")
        return
    fs, _ = fsspec.core.url_to_fs(base, **storage_options())
    root = path("raw")
    for f in files:
        dest = f"{base.rstrip('/')}/{f.relative_to(root).as_posix()}"
        log.info("uploading %s -> %s", f.name, dest)
        fs.put_file(str(f), dest)
