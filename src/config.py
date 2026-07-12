"""Environment + OpenAI-compatible client setup."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
DATASET_DIR = DATA_DIR / "dataset"
EVAL_DIR = ROOT / "eval"

for d in (GENERATED_DIR, DATASET_DIR, EVAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

TEACHER_MODEL = os.getenv("TEACHER_MODEL", "gpt-4o-mini")


def get_client():
    """Return an OpenAI-compatible client. Honors OPENAI_BASE_URL so you can
    point at the lower-cost teacher you're provided without code changes."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    base_url = os.getenv("OPENAI_BASE_URL") or None
    # The OpenAI SDK also reads OPENAI_BASE_URL from the environment on its own,
    # and treats an empty string as a real (invalid) base URL rather than falling
    # back to the default endpoint. Scrub the blank value so the default is used.
    if not base_url:
        os.environ.pop("OPENAI_BASE_URL", None)
    return OpenAI(api_key=api_key, base_url=base_url)
