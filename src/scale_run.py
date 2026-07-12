"""Parallel generation orchestrator (driver only -- does NOT modify the pipeline).

Spawns N independent worker processes, each running the existing
`generate.py` with a distinct RNG seed and its OWN shard output file. Because
every worker writes to a private shard (and `generate.py` already appends with
flush after each gated record), there is no shared-file contention and a crash
loses at most one in-flight record per worker. Shards are merged later by
`merge_shards.py`, followed by a single global gate + dedup pass in
`build_dataset.py`.

Run:
    python scale_run.py --workers 16 --per-worker 230
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARD_DIR = HERE.parent / "data" / "generated" / "shards"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--per-worker", type=int, default=230)
    ap.add_argument("--seed-base", type=int, default=1000)
    args = ap.parse_args()

    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    procs = []
    for i in range(args.workers):
        seed = args.seed_base + i
        out = SHARD_DIR / f"w{seed:06d}.jsonl"
        logf = open(SHARD_DIR / f"w{seed:06d}.log", "w", encoding="utf-8")
        p = subprocess.Popen(
            [
                sys.executable,
                str(HERE / "generate.py"),
                "-n", str(args.per_worker),
                "--seed", str(seed),
                "--out", str(out),
            ],
            cwd=str(HERE),
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
        procs.append((seed, p, logf, out))
        print(f"launched worker seed={seed} -> {out.name}", flush=True)

    start = time.time()
    while True:
        alive = [s for s, p, _, _ in procs if p.poll() is None]
        done = args.workers - len(alive)
        total_kept = 0
        for _, _, _, out in procs:
            if out.exists():
                total_kept += sum(1 for _ in open(out, encoding="utf-8"))
        elapsed = int(time.time() - start)
        print(
            f"[heartbeat] elapsed={elapsed}s done={done}/{args.workers} "
            f"kept_so_far={total_kept}",
            flush=True,
        )
        if not alive:
            break
        time.sleep(20)

    for _, p, logf, _ in procs:
        p.wait()
        logf.close()

    total_kept = 0
    for _, _, _, out in procs:
        if out.exists():
            total_kept += sum(1 for _ in open(out, encoding="utf-8"))
    print(f"\nALL WORKERS DONE. total kept across shards = {total_kept}", flush=True)


if __name__ == "__main__":
    main()
