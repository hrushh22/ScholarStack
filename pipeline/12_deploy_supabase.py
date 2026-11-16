"""Step 12 - Deploy the live backend to Supabase.

  1. reads the project's publishable (browser) key and stores it in .env
  2. stores the LLM settings as Edge Function secrets (the key never reaches the browser)
  3. deploys supabase/functions/generate with the Supabase CLI (downloaded to .tools/ if missing)
  4. checks the function responds, then regenerates web/js/config.js

Needs SUPABASE_URL, SUPABASE_ACCESS_TOKEN and GROQ_API_KEY in .env.
    python pipeline/12_deploy_supabase.py          # the CLI asks for confirmation before deploying
    python pipeline/12_deploy_supabase.py --yes    # non-interactive
"""
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.parse

import httpx

from scholarstack.config import ROOT, env, log

API = "https://api.supabase.com/v1"


def project_ref() -> str:
    host = urllib.parse.urlparse(env("SUPABASE_URL", required=True)).hostname or ""
    return host.split(".")[0]


def set_env_var(name: str, value: str) -> None:
    f = ROOT / ".env"
    text = f.read_text() if f.exists() else ""
    if re.search(rf"^{name}=.*$", text, flags=re.M):
        text = re.sub(rf"^{name}=.*$", f"{name}={value}", text, flags=re.M)
    else:
        text += f"\n{name}={value}\n"
    f.write_text(text)
    os.environ[name] = value


def supabase_cli() -> str:
    found = shutil.which("supabase")
    if found:
        return found
    tools = ROOT / ".tools"
    exe = tools / "supabase"
    if exe.exists():
        return str(exe)
    system = {"Darwin": "darwin", "Linux": "linux"}[platform.system()]
    arch = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "amd64"}[platform.machine()]
    url = f"https://github.com/supabase/cli/releases/latest/download/supabase_{system}_{arch}.tar.gz"
    log.info("downloading Supabase CLI from %s", url)
    tools.mkdir(exist_ok=True)
    archive = tools / "supabase.tar.gz"
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        with open(archive, "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
    with tarfile.open(archive) as tf:
        tf.extract("supabase", tools, filter="data")
    archive.unlink()
    exe.chmod(0o755)
    return str(exe)


if __name__ == "__main__":
    ref = project_ref()
    token = env("SUPABASE_ACCESS_TOKEN", required=True)
    headers = {"Authorization": f"Bearer {token}"}

    # 1. browser key (publishable if the project has one, legacy anon otherwise)
    keys = httpx.get(f"{API}/projects/{ref}/api-keys", headers=headers, timeout=30).raise_for_status().json()
    browser_key = next((k["api_key"] for k in keys if k.get("type") == "publishable" and k.get("api_key")), None) or \
        next(k["api_key"] for k in keys if k.get("name") == "anon")
    set_env_var("SUPABASE_PUBLISHABLE_KEY", browser_key)
    log.info("browser key stored in .env (%s…)", browser_key[:16])

    # 2. Edge Function secrets
    secrets = [{"name": n, "value": env(n, d)} for n, d in (
        ("GROQ_API_KEY", None), ("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        ("LLM_MODEL", "openai/gpt-oss-120b"), ("LLM_FALLBACK_MODEL", "openai/gpt-oss-20b"),
        ("LLM_REASONING_EFFORT", "low"))]
    if not secrets[0]["value"]:
        raise SystemExit("GROQ_API_KEY missing in .env")
    httpx.post(f"{API}/projects/{ref}/secrets", headers=headers, json=secrets, timeout=30).raise_for_status()
    log.info("Edge Function secrets set: %s", ", ".join(s["name"] for s in secrets))

    # 3. deploy the function (no Docker needed with --use-api)
    cli = supabase_cli()
    cmd = [cli, "functions", "deploy", "generate", "--project-ref", ref, "--no-verify-jwt", "--use-api"]
    if "--yes" in sys.argv:  # skip the CLI's confirmation prompt (non-interactive runs)
        cmd.append("--yes")
    subprocess.run(cmd, cwd=ROOT, check=True, env={**os.environ, "SUPABASE_ACCESS_TOKEN": token})

    # 4. health check + browser config
    url = f"{env('SUPABASE_URL').rstrip('/')}/functions/v1/generate"
    r = httpx.get(url, headers={"apikey": browser_key}, timeout=60)
    log.info("health check %s -> %s %s", url, r.status_code, r.text[:200])
    subprocess.run([str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "pipeline" / "13_export_web.py")], check=True)
    log.info("done - commit web/js/config.js and push to publish the site")
