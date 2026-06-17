# Redrob Candidate Ranker
**India Runs Hackathon · Data & AI Challenge · Track 01**
Purvi Porwal · B.Tech IT 

---

## Quick start

```bash
git clone https://github.com/PurviGit/redrob-ranker
cd redrob-ranker
pip install -r requirements.txt
cp /path/to/candidates.jsonl .
python rank.py --candidates candidates.jsonl --out purvi-porwal.csv
python validate_submission.py purvi-porwal.csv
```

Runs in ~195 seconds on CPU. No GPU. No network calls during ranking.

---

## What I actually did (and why)

I spent two days reading the JD before writing any code. That sounds like time-wasting but it wasn't, because the Redrob JD is genuinely different from most. It names specific failure modes they've hired into before. It explains *why* certain backgrounds won't work. It tells you exactly what they mean by "5–9 years." Most JDs don't do this.

So instead of building a keyword matcher and calling it semantic search, I tried to encode what the JD actually says.

---

### The title gate (22% of final score)

The first thing I noticed: the sample submission ranks HR Managers and Accountants in its top 10. That's what happens when you do skill scoring without a domain filter — someone with 9 AI skill keywords but an HR background outranks an ML engineer with 6.

I gate on title *before* running skills scoring. An HR Manager with a perfect skill match scores 0.04. A Recommendation Systems Engineer with mediocre skills scores 0.52. That's intentional — the role is "own the intelligence layer," and no amount of listed skills makes an HR background the right hire for that.

The gate has three tiers: strong (ML/AI/NLP/Search/Retrieval titles), adjacent (backend engineers, data engineers who might have the right skills), and disqualified (everything else). The tiers determine the ceiling, not the floor — skills and semantic scoring still differentiate within each tier.

---

### Skills depth (26% of final score)

Each skill is scored as: `proficiency_weight × duration_trust × endorsement_bonus`.

**Three-pillar combo bonus** — The JD has three distinct technical pillars:
- **Retrieval** (vector DB, BM25, hybrid search, semantic search)
- **Ranking** (LTR, NDCG, MRR, reranking, cross-encoder)
- **LLM/Ops** (RAG, fine-tuning, LoRA, embeddings, sentence-transformers)

A candidate covering all three pillars gets a +0.14 bonus on top of linear skill sum. Covering two gets +0.06. This captures what a linear skill sum misses: a full-stack retrieval engineer is exponentially more valuable than three narrow specialists.

**Skill recency bonus** — Skills mentioned in the candidate's *most recent job* description get an additional bonus (up to +0.08). A candidate who used FAISS in their last job is more relevant than one who listed it from a job three years ago.

---

### Semantic similarity — 3-stage retrieval (14% of final score)

Three stages designed around the 5-minute CPU constraint:

**Stage 1 — TF-IDF on all 31,928 survivors (~20 s):** Cosine similarity vs the JD document. Scores every candidate quickly, identifies top 500 by semantic relevance.

**Stage 2 — Bi-encoder re-rank top 500 (~50 s):** `all-MiniLM-L6-v2` (384d, sentence-transformers) re-encodes only those 500 candidates. Captures meaning that keyword overlap misses.

**Stage 3 — Cross-encoder re-rank top 30 (~20 s):** `cross-encoder/ms-marco-MiniLM-L-6-v2` jointly encodes JD + candidate text for the top 30 candidates. Cross-encoders give dramatically better relevance signal than bi-encoders because they see both texts together. Directly maximises NDCG@10 (50% of the evaluation metric). Blended 60/40 with composite score, rescaled to maintain monotonic ordering.

Both models are pre-downloaded at Docker build time — runs fully offline.

---

### Why consulting firms get penalised (embedded in experience scoring, 16%)

The JD says it twice: they've tried hiring people who want "well-scoped roles" from large companies, and it didn't work. Reading between the lines: they're specifically worried about people who've spent their careers at TCS, Infosys, Wipro — big IT services firms where "AI work" often means wrapping OpenAI calls in enterprise Java.

A candidate whose entire career is at those firms gets a −35% penalty on experience score. Partial consulting history gets −12%. This isn't about those companies being bad employers — it's about what kind of work their engineers typically do versus what Redrob needs.

**AI-native company bonus** — Candidates who've worked at AI-native companies (Yellow.ai, Sarvam AI, Haptik, Mad Street Den, Krutrim, Rephrase.ai, Verloop, Saarthi.ai) get +16% on experience score vs +10% for generic product companies. These companies build AI as their core product — candidates there have directly relevant experience.

---

### Career trajectory (blended into narrative score, 5%)

Beyond what title a candidate holds now, I score whether their career is moving *toward* ML/AI or away from it.

- Recent title ML-relevant AND older titles consistently ML = 1.0
- Recent title more ML-relevant than older titles (growing into ML) = 0.90
- Currently strong ML, varied past = 0.85
- Stable ML career = 0.70
- Moving away from ML = 0.40

This catches candidates who peaked in ML 4 years ago and have since moved into management or generic engineering.

---

### Career narrative scoring (blended into narrative, 5%)

Regex patterns detect production-evidence phrases in career descriptions:

| What I look for | Why |
|---|---|
| `shipped.*retrieval` | Someone who shipped a retrieval system, not just built one |
| `migrat.*keyword.*embed` | The exact migration the JD describes (BM25 → embeddings) |
| `ndcg\|mrr\|offline.*eval` | Evaluation frameworks the JD calls "absolutely required" |
| `embedding.*drift\|index.*refresh` | JD-exact operational scenarios — harder to fake |
| `retrieval.*regress` | JD literally says "retrieval-quality regression in production" |
| `relevance.*label\|human.*eval` | Labeling pipelines for ranking systems |

The patterns also penalise publication-heavy language with no production evidence.

---

### Behavioral signals (12% of final score)

All 23 redrob_signals used. Key weights:
- **Recency** (23%): last_active_date decay — 14d ago = 1.0, 365d = 0.08
- **Response rate** (20%): direct hiring-throughput predictor
- **Notice period** (13%): founding team hire; they need someone who can start soon
- **Interview completion rate** (10%): shows up when they say they will
- **GitHub activity** (8%): this role writes production code

**Reachability multiplier** — Applied to final composite score:
- Response rate < 10%: composite × 0.78 (near-unreachable candidate)
- Response rate < 20%: composite × 0.88

A candidate a recruiter cannot reach is worthless regardless of skill score.

**Notice period cap** — Notice > 90 days: composite × 0.94. Urgent hire, long notice is a real problem.

---

### Honeypot detection

Five rules, each targeting a different type of impossibility:

1. **YoE vs career span**: claimed 12 years but career history only covers 4 years of months
2. **Expert/zero-duration**: 5+ skills listed as "expert" with 0 months of use
3. **Too many skills at too little experience**: 28+ skills at under 2 years
4. **All signals at ceiling**: response_rate=1.0, interview_completion=1.0, github=100, profile=100 simultaneously
5. **Temporal impossibility**: signup_date after last_active_date

---

## Pipeline

```
candidates.jsonl  (100,000)
       │
       ▼  Phase A — Pre-filter: honeypot (5 rules) + title gate          ~3 s
       │           100K → ~32K survivors  (68K eliminated in one pass)
       │
       ▼  Phase B — 3-stage hybrid semantic on ~32K survivors            ~90 s
       │           Stage 1: TF-IDF on all 31,928             (~20 s)
       │           Stage 2: Bi-encoder top 500 (all-MiniLM)  (~50 s)
       │           Stage 3: Cross-encoder top 30 (ms-marco)  (~20 s)
       │
       ▼  Phase C — 8-component scoring on ~32K survivors                ~30 s
       │           Title · Skills (+ pillar combo + recency)
       │           Semantic · Experience (+ AI-native bonus)
       │           Behavioral (+ reachability multiplier)
       │           Narrative (+ trajectory + JD production phrases)
       │           Location · Edu/Assessment
       │
       ▼  Phase D — Sort, reachability/notice caps, top-100 select
       │
       ▼  Phase E — Per-candidate reasoning from real profile fields
       │
       → purvi-porwal.csv  (~195 s · CPU only · ~700 MB RAM peak)
```

---

## Score components

| Component | Weight | What it's actually measuring |
|---|---|---|
| Title fit | 22% | Domain gate — wrong field eliminated before skills scored |
| Skills depth | 26% | Proficiency × duration × endorsements + pillar combo bonus + recency |
| Semantic similarity | 14% | 3-stage: TF-IDF → bi-encoder top 500 → cross-encoder top 30 |
| Experience quality | 16% | YoE band + AI-native/product bonus + consulting penalty + stability |
| Behavioral signals | 12% | All 23 signals + reachability multiplier + notice cap |
| Career narrative | 5% | Production-evidence regex + JD exact phrases + career trajectory |
| Location | 3% | Metro cities (Bangalore/Mumbai/Delhi/Hyderabad) highest |
| Education + assessment | 2% | Institution tier × field relevance + platform assessment scores |

Weights sum to exactly 1.0.

---

## Compute budget

| Constraint | Limit | Actual |
|---|---|---|
| Runtime | ≤ 5 min | **~195 s** (3.25 min) |
| Peak memory | ≤ 16 GB | **~700 MB** |
| GPU | Not allowed | ✅ CPU only |
| Network during ranking | Not allowed | ✅ Fully offline (models pre-cached in Docker) |

---

## Semantic modes

```bash
# Default: 3-stage hybrid (TF-IDF + bi-encoder + cross-encoder) ~195 s
python rank.py --candidates candidates.jsonl --out purvi-porwal.csv

# TF-IDF only (no sentence-transformers needed) ~40 s
python rank.py --tfidf

# Skip semantic entirely (fastest) ~35 s
python rank.py --no-semantic
```

---

## Sandbox

```bash
pip install streamlit pandas plotly
streamlit run app.py
```

Six pages: Overview · Live Analyzer · Top 100 Results · Batch Ranking · Architecture · Submission Guide.

The Live Analyzer lets you paste any candidate JSON and see a full component-by-component breakdown with reasoning.

---

## Tests

```bash
pip install pytest
pytest tests/ -v
# 55 tests, < 5 s
```

Tests cover: each scoring component in isolation, honeypot detection for each rule, reasoning output for completeness and grounding, and end-to-end scoring for a synthetic ideal candidate.

---

## Project structure

```
redrob-ranker/
├── ranker/
│   ├── __init__.py
│   ├── scorer.py         # 8-component scoring, all weights, honeypot rules
│   ├── semantic.py       # 3-stage retrieval: TF-IDF + bi-encoder + cross-encoder
│   └── reasoning.py      # per-candidate reasoning, grounded in actual fields
├── tests/
│   └── test_scorer.py    # 55 unit tests
├── data/
│   └── sample_candidates.json
├── rank.py               # main entry point
├── app.py                # Streamlit sandbox (6 pages)
├── validate_submission.py
├── README.md
├── requirements.txt
├── Dockerfile
└── submission_metadata.yaml
```
