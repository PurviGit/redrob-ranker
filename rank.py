#!/usr/bin/env python3
"""
rank.py  —  Redrob Hackathon: Intelligent Candidate Discovery v2
Produces a validated top-100 submission CSV from candidates.jsonl in <= 5 min on CPU.

Pipeline (two-phase filtering for speed):
  Phase A  — honeypot + title pre-filter on all 100K  (~30 s, O(N) Python)
  Phase B  — semantic TF-IDF index on survivors only   (~30 s, ~300 MB RAM)
  Phase C  — full 8-component scoring on survivors     (~30 s)
  Phase D  — sort + write top-100 CSV

Usage:
  python rank.py
  python rank.py --candidates candidates.jsonl --out submission.csv
  python rank.py --candidates candidates.jsonl.gz --verbose
  python rank.py --tfidf          # force TF-IDF (skips neural even if installed)
  python rank.py --no-semantic    # skip semantic entirely (fastest, lower quality)
"""
import argparse, csv, gzip, json, re, sys, time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ranker.scorer    import score_candidate, is_honeypot, score_title, WEIGHTS
from ranker.semantic  import SemanticScorer
from ranker.reasoning import generate

# Titles that pass Phase A filter (fast string match, no full scoring)
_PASS_TITLE_WORDS = {
    "ml", "machine learning", "ai", "artificial intelligence",
    "nlp", "natural language", "search", "ranking", "recommendation",
    "retrieval", "embedding", "recsys", "data scientist",
    "applied scientist", "research engineer", "applied ml",
}
_REJECT_TITLE_WORDS = {
    "hr manager", "human resources", "marketing manager", "accountant",
    "finance manager", "sales executive", "content writer", "graphic designer",
    "operations manager", "civil engineer", "mechanical engineer",
    "project manager", "customer support", "teacher", "doctor", "lawyer",
    "recruiter", "qa engineer", "quality assurance", "product designer",
    "ui designer", "ux designer",
}


def _quick_title_pass(c: dict) -> bool:
    """True = keep for full scoring. Fast O(1) check."""
    title = (c.get("profile", {}).get("current_title", "") or "").lower()
    if any(r in title for r in _REJECT_TITLE_WORDS):
        return False
    # Keep everything not explicitly rejected; the full scorer will handle it
    return True


def load_candidates(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    out = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser(description="Redrob Candidate Ranker v2")
    ap.add_argument("--candidates",  default="./candidates.jsonl")
    ap.add_argument("--out",         default="./submission.csv")
    ap.add_argument("--top-n",       type=int, default=100)
    ap.add_argument("--no-semantic", action="store_true")
    ap.add_argument("--tfidf",       action="store_true",
                    help="Force TF-IDF semantic (skip neural). Meets 5-min budget.")
    ap.add_argument("--precomputed", default="./precomputed/embeddings.npy")
    ap.add_argument("--verbose",     action="store_true")
    args = ap.parse_args()

    t_total = time.time()

    # ── 1. Load ───────────────────────────────────────────────────────────
    print(f"[1/5] Loading {args.candidates} …")
    t0 = time.time()
    candidates = load_candidates(args.candidates)
    print(f"      {len(candidates):,} loaded  ({time.time()-t0:.1f}s)")

    # ── Phase A: Pre-filter (honeypot + obvious title rejects) ───────────
    print("[A]   Pre-filtering: honeypot + title gate …")
    t0 = time.time()
    survivors = []
    pre_rejected = 0
    for c in candidates:
        hp, _ = is_honeypot(c)
        if hp or not _quick_title_pass(c):
            pre_rejected += 1
        else:
            survivors.append(c)
    print(f"      {len(survivors):,} survivors  "
          f"({pre_rejected:,} pre-rejected)  ({time.time()-t0:.1f}s)")

    # ── 2. Semantic index (survivors only) ────────────────────────────────
    sem_scores:    dict[str, float] = {}
    phrase_scores: dict[str, float] = {}

    if not args.no_semantic:
        print(f"[2/5] Building semantic index for {len(survivors):,} survivors …")
        t0 = time.time()
        precomputed = args.precomputed if Path(args.precomputed).exists() else None
        sem_scorer  = SemanticScorer(use_neural=(not args.tfidf), precomputed_path=precomputed)
        sem_scorer.fit(survivors)
        sem_scores = sem_scorer.get_id_to_score_map()
        for c in survivors:
            phrase_scores[c["candidate_id"]] = sem_scorer.score_phrase_hits(c)
        print(f"      Semantic ready  ({time.time()-t0:.1f}s)")
    else:
        print("[2/5] Semantic skipped (--no-semantic)")

    # ── 3. Full 8-component scoring (survivors only) ──────────────────────
    print(f"[3/5] Scoring {len(survivors):,} survivors …")
    t0            = time.time()
    scored        = []
    honeypots_sc  = 0
    early_rejects = 0

    for i, c in enumerate(survivors):
        cid   = c["candidate_id"]
        sem   = sem_scores.get(cid, 0.50)
        phr   = phrase_scores.get(cid, 0.0)
        score, comps = score_candidate(c, semantic_score=sem, phrase_score=phr)

        if comps.get("honeypot"):       honeypots_sc  += 1
        elif comps.get("early_reject"): early_rejects += 1

        scored.append((score, c, comps))
        if args.verbose and (i + 1) % 10_000 == 0:
            print(f"      … {i+1:,} scored")

    print(f"      {len(scored):,} scored in {time.time()-t0:.1f}s")
    if honeypots_sc or early_rejects:
        print(f"      Honeypots (2nd pass): {honeypots_sc} | Early rejects: {early_rejects}")

    # ── 4. Sort + select ──────────────────────────────────────────────────
    print(f"[4/5] Ranking -> top {args.top_n} …")
    scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
    top = scored[:args.top_n]

    for i in range(1, len(top)):
        if top[i][0] > top[i-1][0] + 1e-9:
            print(f"  WARNING: score inversion at rank {i+1}")

    # ── 5. Write CSV ──────────────────────────────────────────────────────
    print(f"[5/5] Writing -> {args.out} …")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cand, comps) in enumerate(top, 1):
            reasoning = generate(cand, rank, score, comps)
            w.writerow([cand["candidate_id"], rank, round(score, 5), reasoning])

    total = time.time() - t_total
    print(f"\n[DONE]  Done in {total:.1f}s")
    print(f"   Output : {args.out}  ({args.top_n} candidates)")
    print(f"   Top-5 scores   : {[round(s,4) for s,_,_ in top[:5]]}")
    print(f"   Top-5 titles   : {[c['profile']['current_title'] for _,c,_ in top[:5]]}")
    print(f"   Top-5 companies: {[c['profile']['current_company'] for _,c,_ in top[:5]]}")
    print(f"\n   Validate: python validate_submission.py {args.out}")


if __name__ == "__main__":
    main()
