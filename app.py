"""
app.py  —  Redrob AI Candidate Ranker: Interactive Sandbox
Deploy : streamlit run app.py

Pages:
  🏠 Overview       — system design, score-weight pie, key innovations
  🔍 Live Analyzer  — paste any candidate JSON → full 8-component breakdown
  📊 Top 100        — browse / filter / download the submission CSV
  🧪 Batch Ranking  — upload your own JSON batch, see ranked table + heatmap
  📐 Architecture   — pipeline diagram + design-decision explainers
  📋 Submit Guide   — step-by-step portal submission checklist
"""
import json, sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from ranker.scorer   import score_candidate, full_text, skill_map, CORE_SKILLS, WEIGHTS
from ranker.reasoning import generate
from ranker.semantic  import SemanticScorer, JD_SEMANTIC_DOCUMENT, build_candidate_semantic_text

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob AI Ranker",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ─────────────────────────────────────────────────────────────────
SAMPLE_FILE     = Path("data/sample_candidates.json")
SUBMISSION_FILE = Path("submission.csv")

COMP_COLORS = {
    "title":      "#1E2761",
    "skills":     "#4F86F7",
    "semantic":   "#7C3AED",
    "experience": "#0D9488",
    "behavioral": "#F59E0B",
    "narrative":  "#10B981",
    "location":   "#EC4899",
    "edu_asm":    "#6366F1",
}
COMP_LABELS = {
    "title":      "Role Fit (22%)",
    "skills":     "Skills (26%)",
    "semantic":   "Semantic (14%)",
    "experience": "Experience (16%)",
    "behavioral": "Availability (12%)",
    "narrative":  "Career Story (5%)",
    "location":   "Location (3%)",
    "edu_asm":    "Edu/Assess (2%)",
}

# ── session state ─────────────────────────────────────────────────────────────
if "sem_scorer"    not in st.session_state: st.session_state.sem_scorer    = None
if "batch_results" not in st.session_state: st.session_state.batch_results = None


# ── cached loaders ────────────────────────────────────────────────────────────
@st.cache_data
def load_samples():
    if SAMPLE_FILE.exists():
        with open(SAMPLE_FILE) as f:
            return json.load(f)
    return []

@st.cache_data
def load_submission():
    if SUBMISSION_FILE.exists():
        return pd.read_csv(SUBMISSION_FILE)
    return pd.DataFrame()

def get_sem_scorer(candidates):
    if st.session_state.sem_scorer is None:
        with st.spinner("Building semantic index (first run only) …"):
            sc = SemanticScorer(use_neural=False)
            sc.fit(candidates)
            st.session_state.sem_scorer = sc
    return st.session_state.sem_scorer

def score_and_explain(candidate, sem_scorer=None):
    sem, phr = 0.50, 0.0
    if sem_scorer:
        sem = sem_scorer.score_single(candidate)
        phr = sem_scorer.score_phrase_hits(candidate)
    score, comps = score_candidate(candidate, semantic_score=sem, phrase_score=phr)
    reasoning    = generate(candidate, 1, score, comps)
    matched      = comps.get("skills_matched", [])
    smap_keys    = set(skill_map(candidate).keys())
    ft           = full_text(candidate)
    missing      = [s for s in list(CORE_SKILLS)[:20] if s not in smap_keys and s not in ft][:8]
    return score, comps, reasoning, matched, missing

def pct(v): return f"{v*100:.1f}%"

def verdict_badge(score):
    if score >= 0.80: return "🟢 Strong Fit"
    if score >= 0.65: return "🟡 Good Fit"
    if score >= 0.45: return "🟠 Weak Fit"
    return "🔴 Not a Fit"


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Redrob AI Ranker")
    st.markdown("**India Runs Hackathon** · Data & AI Challenge")
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Overview",
        "🔍 Live Analyzer",
        "📊 Top 100 Results",
        "🧪 Batch Ranking",
        "📐 Architecture",
        "📋 Submission Guide",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("""
**System stats**
- 100 K candidates · 8 signals
- Semantic: TF-IDF cosine (neural upgrade ready)
- Runtime: ~90 s CPU · No GPU
- ✅ Validator passed · 40/40 tests
""")
    st.caption("Purvi Porwal · RTU B.Tech IT · CGPA 9.69")

samples = load_samples()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Intelligent Candidate Discovery & Ranking")
    st.markdown("### *Ranks candidates the way a great recruiter would — not by keyword count, but by genuine fit.*")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidates Scored", "100,000", "Full pool")
    c2.metric("Runtime (CPU)",      "~90 s",  "No GPU needed")
    c3.metric("Scoring Signals",    "8 components", "+ 23 behavioral")
    c4.metric("Tests Passing",      "40 / 40", "All green")

    st.divider()
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("## What makes this different")
        st.markdown("""
**Problem:** The sample submission ranks HR Managers #1 — because keyword filters score
skills *before* checking whether the person is even in the right domain.

**This system's 3 key innovations:**

**① Title gate first (22%)**
HR Manager → 0.04 regardless of skills listed.
A RecSys Engineer starts at 1.0 *before* skills are even checked.

**② Semantic similarity layer (14%)**
Every candidate's summary + career descriptions are embedded via TF-IDF cosine (or
`sentence-transformers` locally). A candidate who writes *"led the migration from keyword
search to embedding-based retrieval"* scores high even without listing FAISS explicitly.

**③ Career narrative scoring (5%)**
12 regex patterns detect production evidence:
`"shipped.*retrieval"` · `"migrat.*keyword.*embed"` · `"ndcg|mrr"` · `"embedding.*drift"`

This is what the JD means by *"people who understood retrieval before it became fashionable."*
""")

    with col2:
        fig = go.Figure(go.Pie(
            labels=list(COMP_LABELS.values()),
            values=[WEIGHTS[k] for k in COMP_LABELS],
            hole=0.45,
            marker_colors=list(COMP_COLORS.values()),
            textinfo="label+percent",
            textfont_size=10,
        ))
        fig.update_layout(
            title="8-component weight distribution",
            height=400, showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("## Why this beats keyword rankers")
    c1, c2, c3 = st.columns(3)
    c1.error("**❌ Sample submission**\nRanks HR Managers + Accountants in top 10. Pure keyword count — no role understanding.")
    c2.warning("**⚠️ Typical approach**\nWeighted keyword count. Better, but 30 skills at 0 months duration still scores high.")
    c3.success("**✅ This system**\nTitle-gated + semantically scored + narrative-verified.\nHR Manager → 0.04. 'Shipped embedding retrieval' → high semantic score.")

    st.divider()
    st.markdown("## JD semantic document (what we embed)")
    with st.expander("View JD embedding source"):
        st.code(JD_SEMANTIC_DOCUMENT, language="text")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Live Analyzer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Live Analyzer":
    st.title("🔍 Live Candidate Analyzer")
    st.markdown("Select a sample candidate or paste JSON to see the full 8-component score breakdown.")

    sem_scorer = get_sem_scorer(samples) if samples else None

    sample_map = {
        f"{c['candidate_id']} · {c['profile']['current_title']} · {c['profile']['years_of_experience']}y · {c['profile']['location']}": c
        for c in samples
    }

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("### Input")
        use_sample = st.toggle("Use sample candidate", value=bool(sample_map))
        candidate  = None

        if use_sample and sample_map:
            chosen    = st.selectbox("Pick a sample", list(sample_map.keys()))
            candidate = sample_map[chosen]
            with st.expander("📄 View raw JSON"):
                st.json(candidate)
        else:
            raw = st.text_area(
                "Paste candidate JSON", height=380,
                placeholder='{"candidate_id":"CAND_...","profile":{...},"skills":[...],"redrob_signals":{...},"career_history":[...]}',
            )
            if raw.strip():
                try:
                    candidate = json.loads(raw)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")

    with col_right:
        st.markdown("### Results")
        if candidate:
            with st.spinner("Scoring …"):
                score, comps, reasoning, matched, missing = score_and_explain(candidate, sem_scorer)

            color = "#10B981" if score >= 0.80 else ("#F59E0B" if score >= 0.65 else "#EF4444")
            st.markdown(f"""
<div style='text-align:center;padding:16px 12px;background:#F0F4FF;border-radius:12px;margin-bottom:12px'>
  <div style='font-size:44px;font-weight:700;color:{color}'>{score*100:.1f}%</div>
  <div style='font-size:16px;margin-top:2px'>{verdict_badge(score)}</div>
  <div style='font-size:12px;color:#64748B;margin-top:4px'>
    {candidate["profile"]["current_title"]} · {candidate["profile"]["years_of_experience"]}y · {candidate["profile"]["location"]}
  </div>
</div>""", unsafe_allow_html=True)

            if comps.get("honeypot"):
                st.error(f"🚨 HONEYPOT: {comps.get('honeypot_reason')}")
            elif comps.get("early_reject"):
                st.warning("⚠️ Early reject: disqualified title + weak skills")

            st.markdown("**Score breakdown**")
            comp_keys = ["title","skills","semantic","experience","behavioral","narrative","location","edu_asm"]
            vals      = [comps.get(k, 0) for k in comp_keys]
            labels    = [COMP_LABELS[k]  for k in comp_keys]
            colors    = [COMP_COLORS[k]  for k in comp_keys]

            fig = go.Figure(go.Bar(
                x=vals, y=labels, orientation="h",
                marker_color=colors,
                text=[pct(v) for v in vals], textposition="outside",
            ))
            fig.update_layout(
                xaxis=dict(range=[0, 1.15], showgrid=False, zeroline=False),
                yaxis=dict(autorange="reversed"),
                height=280, plot_bgcolor="white",
                margin=dict(l=0, r=50, t=5, b=5),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select or paste a candidate above.")

    # Detail tabs
    if candidate and not comps.get("honeypot") and not comps.get("early_reject"):
        st.divider()
        t1, t2, t3, t4, t5 = st.tabs(["🎯 Skills", "🧠 Semantic", "📡 Behavioral", "💬 Reasoning", "🏢 Career"])

        with t1:
            ca, cb, cc = st.columns(3)
            with ca:
                st.markdown("**✅ Matched JD must-haves**")
                direct = [m for m in matched if not m.startswith("~")]
                for s in direct: st.markdown(f"- `{s}`")
                if not direct: st.warning("No direct matches")
            with cb:
                st.markdown("**〜 Found in career text**")
                text_hits = [m[1:] for m in matched if m.startswith("~")]
                for s in text_hits: st.markdown(f"- `{s}`")
                if not text_hits: st.info("None")
            with cc:
                st.markdown("**❌ Key gaps**")
                for s in missing[:8]: st.markdown(f"- `{s}`")
                if not missing: st.success("Full coverage!")

            skill_data = candidate.get("skills", [])
            if skill_data:
                df_s = pd.DataFrame(skill_data).head(20)
                ft_  = full_text(candidate)
                df_s["is_core"] = df_s["name"].str.lower().apply(
                    lambda x: "JD Core" if any(k in x for k in list(CORE_SKILLS)[:15]) else "Other"
                )
                fig2 = px.bar(
                    df_s.sort_values("duration_months", ascending=False),
                    x="duration_months", y="name", orientation="h",
                    color="proficiency",
                    color_discrete_map={
                        "beginner":"#E5E7EB","intermediate":"#93C5FD",
                        "advanced":"#3B82F6","expert":"#1D4ED8",
                    },
                    title="Skills: experience depth (months)",
                )
                fig2.update_layout(height=420, yaxis=dict(autorange="reversed"),
                                   plot_bgcolor="white", margin=dict(l=0,r=20,t=40,b=10))
                st.plotly_chart(fig2, use_container_width=True)

        with t2:
            st.markdown("### Semantic alignment")
            sem_score = comps.get("semantic", 0)
            ca, cb = st.columns(2)
            with ca:
                st.metric("Semantic score",        pct(sem_score))
                st.metric("Career narrative score", pct(comps.get("narrative", 0)))
                if   sem_score >= 0.65: st.success("Career text strongly aligns with JD language")
                elif sem_score >= 0.45: st.warning("Moderate semantic alignment")
                else:                   st.error("Weak alignment — career text doesn't match JD vocabulary")
            with cb:
                st.markdown("**JD phrases we're matching against:**")
                jd_phrases = [
                    "embedding-based retrieval","hybrid search","vector database",
                    "ranking system","NDCG / MRR evaluation","A/B testing",
                    "learning-to-rank","LLM fine-tuning","RAG pipeline",
                    "production deployment","embedding drift",
                ]
                ft_ = full_text(candidate)
                for phrase in jd_phrases:
                    icon = "✅" if any(w in ft_ for w in phrase.lower().split()) else "❌"
                    st.markdown(f"{icon} `{phrase}`")

            st.markdown("**Career text used for semantic scoring:**")
            sem_text = build_candidate_semantic_text(candidate)
            st.text_area("", value=sem_text[:1500] + ("…" if len(sem_text) > 1500 else ""),
                         height=180, disabled=True)

        with t3:
            sig = candidate.get("redrob_signals", {})
            bd  = comps.get("beh_detail", {})

            ca, cb, cc, cd = st.columns(4)
            ca.metric("Last Active",    f"{bd.get('days_inactive',999)}d ago")
            cb.metric("Response Rate",  pct(bd.get("response_rate", 0)))
            cc.metric("Notice Period",  f"{sig.get('notice_period_days','?')}d")
            cd.metric("GitHub Score",   f"{sig.get('github_activity_score','N/A')}")

            ce, cf, cg, ch = st.columns(4)
            ce.metric("Open to Work",   "✅" if sig.get("open_to_work_flag") else "❌")
            cf.metric("Interview Rate", pct(sig.get("interview_completion_rate", 0)))
            cg.metric("Offer Accept.",  f"{sig.get('offer_acceptance_rate',0):.0%}"
                                        if sig.get("offer_acceptance_rate",-1) >= 0 else "No history")
            ch.metric("Profile %",      f"{sig.get('profile_completeness_score',0):.0f}%")

            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                sal = sig.get("expected_salary_range_inr_lpa", {})
                st.markdown(f"**Salary:** ₹{sal.get('min',0):.0f}–{sal.get('max',0):.0f} LPA")
                st.markdown(f"**Work mode:** {sig.get('preferred_work_mode','?')}")
                st.markdown(f"**Relocate:** {'✅' if sig.get('willing_to_relocate') else '❌'}")
                st.markdown(f"**Saved by recruiters (30d):** {sig.get('saved_by_recruiters_30d',0)}")
            with col_b:
                st.markdown(f"**Verified email:** {'✅' if sig.get('verified_email') else '❌'}")
                st.markdown(f"**Verified phone:** {'✅' if sig.get('verified_phone') else '❌'}")
                st.markdown(f"**LinkedIn:** {'✅' if sig.get('linkedin_connected') else '❌'}")
                st.markdown(f"**Profile views (30d):** {sig.get('profile_views_received_30d',0)}")

            beh_cats = ["Recency","Response","Notice","GitHub","Interview","Salary Fit"]
            beh_vals = [
                bd.get("recency", 0),
                bd.get("response_rate", 0),
                1.0 if sig.get("notice_period_days",60) <= 30 else
                0.7 if sig.get("notice_period_days",60) <= 60 else 0.4,
                bd.get("github", 0),
                sig.get("interview_completion_rate", 0.7),
                bd.get("salary_fit", 0.7),
            ]
            fig3 = go.Figure(go.Scatterpolar(
                r=beh_vals + [beh_vals[0]], theta=beh_cats + [beh_cats[0]],
                fill="toself", fillcolor="rgba(79,134,247,0.15)",
                line=dict(color="#4F86F7", width=2),
            ))
            fig3.update_layout(
                polar=dict(radialaxis=dict(range=[0,1], showticklabels=False)),
                title="Behavioral signal radar", height=320,
            )
            st.plotly_chart(fig3, use_container_width=True)

        with t4:
            st.markdown("### Recruiter reasoning")
            st.info(reasoning)
            st.markdown("### Profile summary")
            st.markdown(candidate["profile"].get("summary", "*No summary.*"))

        with t5:
            for job in candidate.get("career_history", []):
                with st.expander(
                    f"**{job['title']}** @ {job['company']} · "
                    f"{job.get('duration_months',0)}mo · {job.get('industry','')}"
                ):
                    st.markdown(job.get("description", ""))
            edu = candidate.get("education", [])
            if edu:
                st.markdown("### Education")
                for e in edu:
                    st.markdown(
                        f"**{e.get('degree','')}** in {e.get('field_of_study','')} "
                        f"— {e.get('institution','')} · Tier: {e.get('tier','?')}"
                    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Top 100 Results
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Top 100 Results":
    st.title("📊 Top 100 Ranked Candidates")
    df = load_submission()
    if df.empty:
        st.warning("submission.csv not found. Run `python rank.py` first.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    min_score = col1.slider("Min score filter", 0.0, 1.0, 0.0, 0.01)
    search    = col2.text_input("Search reasoning keywords")
    show_n    = col3.selectbox("Show rows", [10, 25, 50, 100], index=1)

    fdf = df[df["score"] >= min_score]
    if search:
        fdf = fdf[fdf["reasoning"].str.contains(search, case=False, na=False)]
    fdf = fdf.head(show_n)

    ca, cb, cc, cd = st.columns(4)
    ca.metric("Top score",    f"{df['score'].max():.4f}")
    cb.metric("P50 score",    f"{df['score'].median():.4f}")
    cc.metric("Bottom score", f"{df['score'].min():.4f}")
    cd.metric("Shown",        len(fdf))

    fig = px.histogram(df, x="score", nbins=25, title="Score distribution — top 100",
                       color_discrete_sequence=["#4F86F7"])
    fig.update_layout(height=200, margin=dict(l=0,r=0,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        fdf[["rank","candidate_id","score","reasoning"]].rename(columns={"candidate_id":"Candidate ID"}),
        use_container_width=True, height=500,
    )

    if SUBMISSION_FILE.exists():
        st.download_button(
            "⬇ Download submission.csv",
            data=open(SUBMISSION_FILE,"rb").read(),
            file_name="submission.csv", mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Batch Ranking
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Batch Ranking":
    st.title("🧪 Batch Ranking Demo")
    st.markdown("Scores all sample candidates and ranks them — exactly what `rank.py` does on 100 K.")

    upload = st.file_uploader("Upload your own candidates JSON (optional)", type=["json"])
    if upload:
        try:
            batch = json.loads(upload.read())
            if isinstance(batch, dict):
                batch = [batch]
            st.success(f"Loaded {len(batch)} candidates from upload")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            batch = samples
    else:
        batch = samples

    if not batch:
        st.info("No sample candidates found. Add data/sample_candidates.json or upload a batch.")
        st.stop()

    if st.button("▶ Run Batch Ranking", type="primary"):
        sem_sc = get_sem_scorer(batch)
        results, prog = [], st.progress(0, "Scoring …")

        for i, c in enumerate(batch):
            score, comps, reasoning, matched, _ = score_and_explain(c, sem_sc)
            results.append({
                "candidate_id": c["candidate_id"],
                "title":        c["profile"]["current_title"],
                "yoe":          c["profile"]["years_of_experience"],
                "location":     c["profile"]["location"],
                "company":      c["profile"]["current_company"],
                "score":        round(score, 4),
                "verdict":      verdict_badge(score),
                "role_fit":     round(comps.get("title",      0) * 100),
                "skills":       round(comps.get("skills",     0) * 100),
                "semantic":     round(comps.get("semantic",   0) * 100),
                "experience":   round(comps.get("experience", 0) * 100),
                "behavioral":   round(comps.get("behavioral", 0) * 100),
                "narrative":    round(comps.get("narrative",  0) * 100),
                "matched":      ", ".join(matched[:4]),
                "honeypot":     comps.get("honeypot", False),
                "reasoning":    reasoning,
            })
            prog.progress((i+1) / len(batch), f"Scored {i+1}/{len(batch)}")

        prog.empty()
        df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        st.session_state.batch_results = df

    if st.session_state.batch_results is not None:
        df = st.session_state.batch_results
        st.success(f"✅ Ranked {len(df)} candidates")

        top20 = df.head(20)
        fig = px.bar(
            top20, x="score", y="title", orientation="h",
            color="score", color_continuous_scale=["#EF4444","#F59E0B","#10B981"],
            range_color=[0.3, 1.0], hover_data=["yoe","location","company"],
            title="Top 20 candidates by score",
        )
        fig.update_layout(height=500, yaxis=dict(autorange="reversed"),
                          coloraxis_showscale=False, margin=dict(l=0,r=0,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

        heat_cols = ["role_fit","skills","semantic","experience","behavioral","narrative"]
        heat_df   = top20.set_index("title")[heat_cols]
        fig2 = px.imshow(heat_df, color_continuous_scale="Blues", aspect="auto",
                         text_auto=True, title="Component score heatmap (top 20)")
        fig2.update_layout(height=480)
        st.plotly_chart(fig2, use_container_width=True)

        hp = df[df["honeypot"]]
        if not hp.empty:
            st.error(f"🚨 {len(hp)} honeypot(s) detected")
            st.dataframe(hp[["rank","candidate_id","title","yoe","reasoning"]])

        st.dataframe(
            df[["rank","candidate_id","title","yoe","location","score","verdict","matched"]],
            use_container_width=True, height=400,
        )
        csv_bytes = df[["candidate_id","rank","score","reasoning"]].to_csv(index=False).encode()
        st.download_button("⬇ Download ranked CSV", data=csv_bytes,
                           file_name="batch_ranked.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Architecture
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📐 Architecture":
    st.title("📐 System Architecture")
    st.code("""
candidates.jsonl (100 K)
    │
    ▼ [Stage 1] Honeypot Detection  (5 rules)
    │   → Impossible YoE · expert/0-duration · too many skills/<2y
    │     all signals at ceiling · signup > last_active  →  score 0.0
    │
    ▼ [Stage 2] Title Gate  (22 % weight)
    │   → HR Manager / Accountant / Graphic Designer → 0.04
    │     RecSys Engineer / ML Engineer / NLP Engineer → 1.0
    │     Backend Engineer → 0.52  (adjacent; skills must compensate)
    │
    ▼ [Stage 3] Semantic Index  (14 % weight)
    │   → JD embedded as TF-IDF vector (or neural sentence-transformers)
    │     Each candidate: summary + career desc + headline → candidate vector
    │     Score = cosine(JD_vector, candidate_vector)
    │     + phrase-hit bonus: "shipped retrieval" · "embedding drift" · "NDCG"
    │
    ▼ [Stage 4] Multi-signal Scoring  (remaining 64 %)
    │   Skills (26%)     — proficiency × duration × endorsements
    │   Experience (16%) — YoE range · company type · consulting penalty · hop penalty
    │   Behavioral (12%) — 23 signals: recency · response rate · notice · GitHub · salary
    │   Narrative (5%)   — 14 regex patterns on career text
    │   Location (3%)    — Pune/Noida ideal · India tier-1 good · relocation considered
    │   Edu/Assess (2%)  — institution tier × CS field · platform assessment scores
    │
    ▼ [Stage 5] Sort + Select top 100
    │   Descending composite score.  Tie-break: candidate_id ascending (deterministic).
    │
    ▼ [Stage 6] Reasoning Generation
    │   1-2 sentences.  References title · company · YoE · real skills · behavioural signals.
    │   Tone calibrated to rank tier.  Concerns acknowledged for rank 7+.
    │
    → submission.csv ✅  (validated · monotonically non-increasing scores)
""", language="text")

    st.divider()
    st.markdown("## Design decisions — each grounded in the JD")
    decisions = {
        "Title gate before skills": (
            "JD explicitly says keyword stuffers are traps. HR Manager + 9 AI skills = NOT an AI engineer. "
            "We score title first, skills second."
        ),
        "Semantic layer (cosine similarity)": (
            "A candidate who writes 'I led the migration from keyword search to embedding-based retrieval' "
            "may not list FAISS explicitly. TF-IDF cosine captures this; keyword matching misses it. "
            "Upgrades to neural embeddings with zero code change."
        ),
        "Career narrative scoring": (
            "JD says 'people who understood retrieval and ranking before it became fashionable.' "
            "We detect this via 14 production-evidence patterns: 'shipped', 'NDCG', 'A/B test', 'embedding drift'."
        ),
        "Consulting firm penalty (−35%)": (
            "JD explicitly: 'career in IT services outsourcing is a soft disqualifier.' "
            "TCS/Infosys full career → exp_score × 0.65."
        ),
        "Behavioral availability (12%)": (
            "JD: 'A perfect-on-paper candidate who hasn't logged in for 6 months is, for hiring purposes, "
            "not actually available.' Inactive 180d → recency = 0.42."
        ),
        "All 23 redrob_signals used": (
            "Salary fit, offer acceptance rate, verified phone/email, profile views — "
            "these are real hiring signals, not noise."
        ),
    }
    for title, detail in decisions.items():
        with st.expander(f"**{title}**"):
            st.markdown(detail)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Submission Guide
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Submission Guide":
    st.title("📋 Complete Submission Guide")
    st.markdown("Step-by-step from code to portal submission.")

    st.markdown("""
## Step 1 — Run locally

```bash
pip install numpy
python rank.py --candidates candidates.jsonl --out submission.csv
python validate_submission.py submission.csv
# → Submission is valid.
```

## Step 2 — (Optional) Neural semantic scoring

```bash
pip install sentence-transformers
python rank.py --candidates candidates.jsonl --out submission.csv
# Automatically uses all-MiniLM-L6-v2 (384d) — no code change needed
```

Pre-compute for repeated runs:
```bash
python precompute/embed.py --candidates candidates.jsonl
python rank.py   # loads precomputed in ~2s
```

## Step 3 — Push to GitHub

```bash
git init
git add .
git commit -m "feat: multi-signal ranker with semantic layer, 40/40 tests"
git remote add origin https://github.com/YOUR_USERNAME/redrob-ranker
git push -u origin main
```

Make the repo **public**.

## Step 4 — Deploy sandbox (HuggingFace Spaces)

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → New Space → SDK: **Streamlit** → Public
2. Upload: `app.py`, `ranker/`, `data/sample_candidates.json`, `submission.csv`, `requirements.txt`
3. Copy the Space URL → paste into `submission_metadata.yaml`

**requirements.txt for HF Spaces:**
```
streamlit>=1.35.0
pandas>=2.0.0
plotly>=5.18.0
numpy>=1.24.0
```

## Step 5 — Fill submission_metadata.yaml

```yaml
team_name: "purvi-porwal"
github_repo: "https://github.com/YOUR_USERNAME/redrob-ranker"
sandbox_link: "https://huggingface.co/spaces/YOUR_USERNAME/redrob-ranker"
```

## Step 6 — Upload to hack2skill portal

Go to: **INDIA RUNS → The Data & AI Challenge → Submit**

1. Select challenge: **Data & AI Challenge : Intelligent Candidate Discovery**
2. GitHub URL → your public repo
3. PDF deck → `redrob_deck.pdf`
4. Ranked output → `submission.csv`
5. Click **Submit**

⚠️ Deadline: **July 2, 2026 11:59 PM IST**

## What judges evaluate

| Stage | Check | How to pass |
|-------|-------|-------------|
| Stage 3 | Reproduce CSV in Docker, 5 min CPU | `python rank.py` must work, no network |
| Stage 4 | 10 random reasoning rows | Specific facts, no templates, concerns acknowledged |
| Stage 5 | 30-min video call | Know every design decision in your code |
""")
