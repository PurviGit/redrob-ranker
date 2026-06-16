# Redrob Candidate Ranker
**India Runs Hackathon · Data & AI Challenge · Track 01**
Purvi Porwal · B.Tech IT · Rajasthan Technical University · CGPA 9.69

---

## Quick start

```bash
git clone https://github.com/purviporwal/redrob-ranker
cd redrob-ranker
pip install numpy
cp /path/to/candidates.jsonl .
python rank.py --tfidf
python validate_submission.py submission.csv
```

Runs in ~126 seconds on CPU. No GPU. No network calls during ranking.

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

### Semantic similarity via TF-IDF cosine (14% of final score)

I wrote a two-pass streaming TF-IDF implementation because I didn't want to require sentence-transformers as a hard dependency. It runs on all survivors (~32K after pre-filter) in about 40 seconds using ~300 MB of RAM — no NxV matrix stored, just chunk-by-chunk dot products.

The JD "document" I embed against is not just a dump of skill keywords. It includes the reasoning behind the role: phrases like "migration from keyword-based search to embedding-based retrieval," "embedding drift and index refresh in production," "retrieval quality regression." A candidate who describes exactly this in their career history will score 0.42+ on semantic even without listing FAISS or Pinecone by name.

The code is written so that `pip install sentence-transformers` automatically upgrades to neural embeddings with zero code changes. That's a deliberate design decision — if this system were running in production, you'd want to be able to swap in a better encoder without rewiring anything.

---

### Why consulting firms get penalised (embedded in experience scoring, 16%)

The JD says it twice: they've tried hiring people who want "well-scoped roles" from large companies, and it didn't work. Reading between the lines: they're specifically worried about people who've spent their careers at TCS, Infosys, Wipro — big IT services firms where "AI work" often means wrapping OpenAI calls in enterprise Java.

I have a list of Indian IT consulting firms. A candidate whose entire career is at those firms gets a 35% penalty on experience score. This isn't about those companies being bad employers — it's about what kind of work their engineers typically do versus what Redrob needs.

---

### Career narrative scoring (5% of final score)

Fourteen regex patterns detect production-evidence phrases in career descriptions:

| What I look for | Why |
|---|---|
| `shipped.*retrieval` | Someone who shipped a retrieval system, not just built one |
| `migrat.*keyword.*embed` | The exact migration the JD describes (BM25 → embeddings) |
| `ndcg\|mrr\|offline.*eval` | Evaluation frameworks the JD calls "absolutely required" |
| `embedding.*drift\|index.*refresh` | Operational knowledge — harder to fake than skill keywords |
| `hybrid.*search\|bm25.*embed` | The specific architecture they're running at Redrob |

The patterns also penalise publication-heavy language with no production evidence — "publication", "arxiv", "theorem", "proof" with no shipping language — because the JD explicitly says pure researchers won't move forward.

---

### Behavioral signals (12% of final score)

The redrob_signals doc says these often matter more than static profile data. I believe it, and here's why: a candidate who hasn't logged in for six months and responds to 5% of recruiter messages is not practically available, regardless of how good their skills are.

My behavioral scoring uses all 23 signals. The highest-weighted:
- **Recency** (23% of behavioral): last_active_date decay — 14 days ago scores 1.0, 365 days ago scores 0.08
- **Response rate** (20%): direct hiring-throughput predictor
- **Notice period** (13%): founding team hire; they need someone who can start soon
- **Interview completion rate** (10%): shows up when they say they will
- **GitHub activity** (8%): this specific role writes production code — GitHub score matters

I also check salary range against market (25–65 LPA band for a senior AI engineer in India), trust signals (verified email/phone/LinkedIn), and recruiter-demand signals (saves and profile views in 30 days).

---

### Honeypot detection

Five rules, each targeting a different type of impossibility:

1. **YoE vs career span**: claimed 12 years but career history only covers 4 years of months
2. **Expert/zero-duration**: 5+ skills listed as "expert" with 0 months of use (spec example: "10 skills, 0 years")
3. **Too many skills at too little experience**: 28+ skills at under 2 years
4. **All signals at ceiling**: response_rate=1.0, interview_completion=1.0, github=100, profile=100 simultaneously — synthetic artifact
5. **Temporal impossibility**: signup_date after last_active_date

These eliminate ~19 clear honeypots. The remaining suspicious profiles (keyword stuffers, profile inflators) get naturally penalised through the duration trust multiplier in skills scoring — expert with 0 months of duration gets 0.40×0.00 = 0.0 on that dimension.

---

## Pipeline

```
candidates.jsonl  (100,000)
       │
       ▼  Phase A — Pre-filter: honeypot (5 rules) + title gate       ~3 s
       │           100K → ~32K survivors  (68K eliminated in one pass)
       │
       ▼  Phase B — TF-IDF semantic index on ~32K survivors            ~53 s
       │           Single vocab build, chunk-by-chunk cosine scores
       │           Neural upgrade: pip install sentence-transformers (auto)
       │
       ▼  Phase C — 8-component scoring on ~32K survivors              ~43 s
       │           Title · Skills · Semantic · Experience
       │           Behavioral · Narrative · Location · Edu/Assess
       │
       ▼  Phase D — Sort, select top 100, enforce non-increasing scores
       │
       ▼  Phase E — Per-candidate reasoning from real profile fields
       │
       → submission.csv  (126 s · CPU only · ~500 MB RAM peak)
```

---

## Score components

| Component | Weight | What it's actually measuring |
|---|---|---|
| Title fit | 22% | Domain gate — wrong field is eliminated before skills are scored |
| Skills depth | 26% | Proficiency × duration × endorsements for JD-aligned skills specifically |
| Semantic similarity | 14% | TF-IDF cosine of career text vs JD document |
| Experience quality | 16% | YoE band + company type + stability + consulting/product penalty |
| Behavioral signals | 12% | All 23 signals: recency, response rate, notice, GitHub, salary, trust |
| Career narrative | 5% | 14 production-evidence regex patterns against career descriptions |
| Location | 3% | Pune/Noida highest; India Tier-1 very close; relocation factored in |
| Education + assessment | 2% | Institution tier × field relevance + platform assessment scores |

Weights sum to exactly 1.0.

---

## Compute budget

| Constraint | Limit | Actual |
|---|---|---|
| Runtime | ≤ 5 min | **126 s** |
| Peak memory | ≤ 16 GB | **~500 MB** |
| GPU | Not allowed | ✅ CPU only |
| Network during ranking | Not allowed | ✅ Fully offline |

---

## Optional neural upgrade

```bash
pip install sentence-transformers
python rank.py --candidates candidates.jsonl --out submission.csv
# Automatically uses all-MiniLM-L6-v2 (384d). No code change.
```

To pre-compute embeddings for faster reruns:
```bash
python precompute/embed.py --candidates candidates.jsonl
python rank.py   # loads in ~2 s
```

---

## Sandbox

```bash
pip install streamlit pandas plotly
streamlit run app.py
```

Six pages: Overview · Live Analyzer · Top 100 Results · Batch Ranking · Architecture · Submission Guide.

The Live Analyzer lets you paste any candidate JSON and see a full component-by-component breakdown with reasoning — useful for debugging why a specific candidate ranked where they did.

---

## Tests

```bash
pip install pytest numpy
pytest tests/ -v
# 55 tests, < 7 s
```

Tests cover: each scoring component in isolation, honeypot detection for each rule, reasoning output for completeness and grounding, and end-to-end scoring for a synthetic ideal candidate.

---

## Project structure

```
redrob-ranker/
├── ranker/
│   ├── __init__.py
│   ├── scorer.py         # 8-component scoring, all weights, honeypot rules
│   ├── semantic.py       # TF-IDF + optional neural semantic scorer
│   └── reasoning.py      # per-candidate reasoning, grounded in actual fields
├── precompute/
│   └── embed.py          # offline pre-computation for neural embeddings
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
