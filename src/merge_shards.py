"""Merge per-worker shards into raw.jsonl (idempotent, corruption-safe).

The reviewed gold records already at the top of raw.jsonl are preserved and
stay FIRST. Every shard record is appended exactly once, keyed by its `id`, so
re-running this (e.g. after a crashed/duplicated shard) never double-writes and
never corrupts raw.jsonl. Writes go to a temp file then atomically replace.

Run:
    python merge_shards.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from config import GENERATED_DIR

RAW = GENERATED_DIR / "raw.jsonl"
SHARD_DIR = GENERATED_DIR / "shards"


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # tolerate a single half-written trailing line from a crash
                continue
    return out


def main() -> None:
    existing = read_jsonl(RAW) if RAW.exists() else []
    seen = {r["id"] for r in existing}
    print(f"raw.jsonl currently has {len(existing)} records ({len(seen)} unique ids)")

    added = 0
    merged = list(existing)
    for shard in sorted(SHARD_DIR.glob("w*.jsonl")):
        recs = read_jsonl(shard)
        new = [r for r in recs if r.get("id") and r["id"] not in seen]
        for r in new:
            seen.add(r["id"])
            merged.append(r)
        added += len(new)
        print(f"  {shard.name}: {len(recs)} records, {len(new)} new")

    tmp = RAW.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in merged:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, RAW)

    print(f"\nmerged: added {added} new records. raw.jsonl now has {len(merged)} records.")


if __name__ == "__main__":
    main()
