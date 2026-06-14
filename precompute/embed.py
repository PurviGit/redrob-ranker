#!/usr/bin/env python3
"""
precompute/embed.py  —  Pre-compute candidate embeddings offline.

Run ONCE before rank.py. Saves scores to precomputed/embeddings.npy.
Subsequent rank.py runs load in ~2 s instead of rebuilding the index.

Usage:
    python precompute/embed.py --candidates candidates.jsonl
    python precompute/embed.py --candidates candidates.jsonl.gz
"""
import argparse, gzip, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ranker.semantic import SemanticScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="./candidates.jsonl")
    ap.add_argument("--out",        default="./precomputed/embeddings.npy")
    ap.add_argument("--model",      default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading {args.candidates} …")
    t0     = time.time()
    opener = gzip.open if args.candidates.endswith(".gz") else open
    cands  = []
    with opener(args.candidates, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cands.append(json.loads(line))
    print(f"      {len(cands):,} loaded  ({time.time()-t0:.1f}s)")

    print(f"[2/3] Building index with {args.model} …")
    scorer = SemanticScorer(use_neural=True, precomputed_path=args.out)
    scorer.fit(cands)

    print(f"[3/3] Saved to {args.out}")
    print("      Run rank.py normally — loads precomputed automatically.")


if __name__ == "__main__":
    main()
