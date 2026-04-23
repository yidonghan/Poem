from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


POEM_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = POEM_ROOT.parent
VENDOR_ROOT = POEM_ROOT / "vendor"

COGVIEW_ROOT = WORKSPACE_ROOT / "CogView3-main"
COGVIEW_DIFFUSERS_SRC = VENDOR_ROOT / "cogview_diffusers"
SD360_ROOT = VENDOR_ROOT / "sd360"
DREAMSCENE_ROOT = VENDOR_ROOT / "dreamscene"
DREAMSCENE_ORIG_ROOT = WORKSPACE_ROOT / "DreamScene360"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(text: str, max_length: int = 48) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if not text:
        text = "run"
    return text[:max_length].rstrip("-")


def default_run_name(prompt: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{slugify(prompt)}"


def run_command(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    printable = " ".join(cmd)
    print(f"$ {printable}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=merged_env, check=True)


def python_cmd(script: Path, *args: str) -> list[str]:
    return [sys.executable, str(script), *args]
