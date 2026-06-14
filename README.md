# Redrob AI — Intelligent Candidate Discovery & Ranking

**India Runs Hackathon · Data & AI Challenge**
**Participant:** Purvi Porwal · B.Tech IT · Rajasthan Technical University · CGPA 9.69

---

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/redrob-ranker
cd redrob-ranker
pip install numpy
cp /path/to/candidates.jsonl .
python rank.py --tfidf                # → submission.csv in ~126 s
python validate_submission.py submission.csv
# → Submission is valid.
```

---

## What makes this top-5% worthy

### The core problem the sample submission gets wrong
The sample submission ranks HR Managers and Accountants in the top 10.
That is what keyword scoring looks like — it counts skill matches without
ever checking whether the person is in the right domain.

### 3 innovations that fix this

**① Title gate before skills (22% weight)**
```
HR Manager  + 9 AI skills  →  score 0.04   (disqualified; cannot enter top 100)
RecSys Engineer            →  score 1.00   (title gate passed; skills amplify)
Backend Engineer           →  score 0.52   (adjacent; must be compensated by skills)
```

**② Semantic similarity layer (14% weight)**
Every candidate's summary + career descriptions are embedded via two-pass streaming
TF-IDF cosine similarity against the JD. A candidate who writes:

> *"led the migration from keyword-based search to embedding-based retrieval"*

scores 0.42+ on semantic even without listing FAISS or Pinecone explicitly.
A keyword ranker misses this entirely.

Upgrades to neural embeddings (sentence-transformers) with **zero code change** —
just `pip install sentence-transformers`.

**③ Career narrative scoring (5% weight)**
14 regex patterns detect production evidence:
| Pattern | Bonus |
|---------|-------|
| `"shipped.*retrieval"` | +0.15 |
| `"migrat.*keyword.*embed"` | +0.20 |
| `"ndcg\|mrr"` | +0.12 |
| `"embedding.*drift\|index.*refresh"` | +0.18 |

This is what the JD means by *"people who understood retrieval before it became fashionable."*

---

## Architecture

```
candidates.jsonl (100 K)
    │
    ▼ [Phase A] Fast Pre-filter: honeypot (5 rules) + title reject  ~3 s
    │           100K → ~32K survivors (68K eliminated in O(N) Python pass)
    │
    ▼ [Phase B] Semantic Index — single-pass cached TF-IDF on ~32K  ~53 s
    │           Candidate text → TF-IDF vector vs JD cosine similarity
    │           (Optional neural upgrade: sentence-transformers, zero code change)
    │
    ▼ [Phase C] 8-Component Scoring on ~32K survivors               ~43 s
    │           Title (22%) · Skills (26%) · Semantic (14%)
    │           Experience (16%) · Behavioral (12%)
    │           Narrative (5%) · Location (3%) · Edu/Assess (2%)
    │
    ▼ [Phase D] Sort + select top 100 (deterministic tie-break)
    │
    ▼ [Phase E] Per-candidate Reasoning — grounded in real profile fields
    │
    → submission.csv   126 s total · CPU only · 55/55 tests passing
```

## Scoring components

| Component | Weight | What it captures |
|-----------|--------|-----------------|
| Title / role fit | 22% | Primary gate. Wrong domain → excluded before skills scored |
| Skills depth | 26% | Proficiency × duration × endorsements. Expert/0-months = low |
| Semantic similarity | 14% | TF-IDF cosine vs JD embedding. Career text > keywords |
| Experience quality | 16% | 5–9 yr ideal. Consulting-only → −35%. Product co → +10% |
| Behavioral signals | 12% | All 23 signals: recency, response rate, notice, GitHub, salary |
| Career narrative | 5% | Production-evidence phrases in career text |
| Location | 3% | Pune/Noida/Delhi → 1.0. India tier-1 → 0.90 |
| Education + assessment | 2% | IIT/NIT tier + platform skill scores |

## Honeypot detection

| Rule | What it catches |
|------|----------------|
| YoE vs career span | Claimed 12 yr but history = 4 yr |
| Expert / 0-duration | 5+ skills expert with 0 months of use |
| Too many skills | 25+ skills at < 2 yr experience |
| Perfect signals ceiling | All behavioral signals at 1.0 / 100 simultaneously |
| Temporal impossibility | signup_date after last_active_date |

## Compute budget

| Constraint | Limit | Actual |
|---|---|---|
| Runtime | ≤ 5 min | **126 s** |
| Memory | ≤ 16 GB | **~500 MB** |
| Compute | CPU only | ✅ |
| Network | Off | ✅ No API calls |

## Optional neural upgrade

```bash
pip install sentence-transformers
python rank.py --candidates candidates.jsonl --out submission.csv
# Automatically uses all-MiniLM-L6-v2 (384d). No code change needed.
```

Pre-compute for repeated runs:
```bash
python precompute/embed.py --candidates candidates.jsonl
python rank.py   # loads precomputed in ~2 s
```

## Sandbox

```bash
pip install streamlit pandas plotly
streamlit run app.py
```

6 pages: Overview · Live Analyzer · Top 100 Results · Batch Ranking · Architecture · Submit Guide

## Tests

```bash
pytest tests/ -v   # 55 / 55 tests, < 7 s
```

## File structure

```
redrob-ranker/
├── ranker/
│   ├── __init__.py
│   ├── scorer.py         # 8-component scoring engine
│   ├── semantic.py       # TF-IDF / neural semantic scorer
│   └── reasoning.py      # per-candidate reasoning generator
├── precompute/
│   └── embed.py          # offline embedding pre-computation
├── tests/
│   └── test_scorer.py    # 40 unit tests
├── data/
│   └── sample_candidates.json   # 30-candidate sandbox sample
├── rank.py               # main entrypoint
├── app.py                # Streamlit sandbox (6 pages)
├── validate_submission.py
├── README.md
├── requirements.txt
├── Dockerfile
└── submission_metadata.yaml
```
