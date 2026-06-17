"""
ranker/reasoning.py  —  Stage-4-quality per-candidate reasoning.

Every claim references a real field from the candidate profile.
No hallucination: skills named must exist in candidate.skills,
signal values quoted come directly from redrob_signals.
Spec Stage-4 checks: specific facts, JD connection, honest concerns,
no hallucination, variation, rank-consistent tone.
"""
from __future__ import annotations
import re
from ranker.scorer import (
    norm, days_since, full_text,
    CONSULTING_FIRMS, PRODUCT_COMPANIES, CORE_SKILLS,
)

JD_MUST_HAVE = {
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "chromadb", "lancedb", "annoy",
    "sentence-transformers", "sentence transformers", "bge", "e5",
    "openai embeddings", "embeddings", "vector embeddings",
    "dense retrieval", "semantic search", "hybrid search", "bm25",
    "learning to rank", "ltr", "ndcg", "mrr",
    "rag", "retrieval augmented generation",
    "fine-tuning", "lora", "qlora", "peft",
    "information retrieval", "ranking", "reranking",
    "vector database", "vector db",
}

_PRODUCTION_PATTERNS: list[tuple[str, str]] = [
    (r"(\d[\d,]*\s*(?:M\+?|million|B\+?|billion|K\+?|thousand)?\s*(?:queries|requests|QPM|QPS|users|candidates|records))",
     "scale"),
    (r"(hybrid\s+(?:BM25|search|retrieval)[^.]{0,60})",              "hybrid-search"),
    (r"(BGE|E5|bge-|openai.*embed|sentence.transformer)[^.]{0,40}",   "embedding-model"),
    (r"(FAISS|Pinecone|Weaviate|Qdrant|Milvus|OpenSearch|pgvector)[^.]{0,40}", "vector-db"),
    (r"(NDCG|MRR|MAP|Recall@\d|Precision@\d|A/B\s+test)[^.]{0,60}",  "eval-metric"),
    (r"(learning.to.rank|LTR|XGBoost.*rank|LightGBM.*rank)[^.]{0,50}", "ltr"),
    (r"(LoRA|QLoRA|PEFT|fine.tun)[^.]{0,50}",                         "finetuning"),
    (r"(RAG|retrieval.augmented)[^.]{0,60}",                           "rag"),
    (r"(embedding.*drift|index.*refresh|retrieval.*regress)[^.]{0,60}", "embed-ops"),
]


def _extract_career_keyword(hist: list[dict]) -> str:
    """Extract the most specific technical keyword/phrase from career history.
    Returns a SHORT phrase (not a full sentence) to avoid template repetition."""
    for job in hist[:3]:
        desc = job.get("description", "")
        for pat, tag in _PRODUCTION_PATTERNS:
            m = re.search(pat, desc, re.IGNORECASE)
            if m:
                kw = m.group(1).strip().rstrip(".,;")
                if len(kw) > 80:
                    kw = kw[:80].rsplit(" ", 1)[0] + "..."
                if len(kw) > 5:
                    return kw
    return ""


def _jd_matched_skills(candidate: dict) -> list[str]:
    """Skills in the profile that are explicitly in the JD must-have list."""
    named = []
    for s in candidate.get("skills", []):
        name = s.get("name", "").lower()
        if any(jd in name or name in jd for jd in JD_MUST_HAVE):
            named.append(s["name"])
        if len(named) >= 5:
            break
    return named


def _top_skills_by_proficiency(candidate: dict, n: int = 3) -> list[str]:
    """Top N skills ranked by proficiency level, to show actual depth."""
    order = {"expert": 3, "advanced": 2, "intermediate": 1, "beginner": 0}
    skills = sorted(
        candidate.get("skills", []),
        key=lambda s: order.get(s.get("proficiency_level", "").lower(), 0),
        reverse=True,
    )
    return [s["name"] for s in skills[:n]]


def generate(candidate: dict, rank: int, score: float, components: dict) -> str:
    if components.get("honeypot"):
        return (
            f"Profile flagged as honeypot: "
            f"{components.get('honeypot_reason', 'impossible profile')}. "
            "Excluded from shortlist."
        )

    p    = candidate["profile"]
    sig  = candidate.get("redrob_signals", {})
    hist = candidate.get("career_history", [])

    title   = p.get("current_title", "")
    yoe     = p.get("years_of_experience", 0)
    loc     = p.get("location", "unknown")
    company = p.get("current_company", "")
    loc_str = loc.split(",")[0].strip()

    jd_skills  = _jd_matched_skills(candidate)
    top_skills = _top_skills_by_proficiency(candidate, 3)
    career_kw  = _extract_career_keyword(hist)

    notice   = sig.get("notice_period_days", 60)
    rr       = sig.get("recruiter_response_rate", None)
    gh       = sig.get("github_activity_score", -1)
    otw      = sig.get("open_to_work_flag", False)
    sal      = sig.get("expected_salary_range_inr_lpa", {})
    inactive = components.get("beh_detail", {}).get("days_inactive", 9999)
    exp_tier = components.get("exp_tier", "")
    t_tier   = components.get("title_tier", "")
    neg_hits = components.get("skills_neg_hits", 0)

    cos        = [norm(j.get("company", "")) for j in hist]
    is_product = any(any(p_ in co for p_ in PRODUCT_COMPANIES) for co in cos)

    # ── Sentence 1: identity + unique technical evidence ───────────────────
    # Lead with title, YoE, company (always unique per candidate)
    s1 = f"{title} ({yoe:.1f}yr) at {company}, {loc_str}."

    # Add the most specific unique technical signal we can find, in priority order:
    # 1. JD-aligned skills (always unique per candidate's actual skill list)
    # 2. Career keyword (short phrase, not full sentence — avoids template repeats)
    # 3. Top skills by proficiency

    if jd_skills and career_kw:
        # Best case: both skills and a specific career keyword
        skills_str = ", ".join(jd_skills[:3])
        s1 += f" JD-aligned: {skills_str}. Production evidence: {career_kw}."
    elif jd_skills:
        skills_str = ", ".join(jd_skills[:3])
        s1 += f" JD-aligned skills: {skills_str}."
        if len(jd_skills) >= 4:
            s1 += f" Also: {jd_skills[3]}."
    elif career_kw:
        # No JD skills but has a career keyword
        s1 += f" Career highlight: {career_kw}."
        if top_skills:
            s1 += f" Top skills: {', '.join(top_skills)}."
    elif top_skills:
        s1 += f" Strongest skills: {', '.join(top_skills)}."

    # ── Gather strengths and concerns (signal-value specific) ───────────
    strengths, concerns = [], []

    if inactive <= 7:
        strengths.append(f"active {inactive}d ago")
    elif inactive <= 30:
        strengths.append(f"active within {inactive}d")
    elif inactive > 180:
        concerns.append(f"inactive {inactive}d — reachability risk")

    if rr is not None:
        if rr >= 0.70:
            strengths.append(f"{rr:.0%} response rate")
        elif rr < 0.30:
            concerns.append(f"low response rate ({rr:.0%})")

    if gh >= 70:
        strengths.append(f"GitHub {gh:.0f}/100")
    elif gh >= 40:
        strengths.append(f"GitHub {gh:.0f}/100")
    elif gh == -1:
        concerns.append("no GitHub linked")

    if notice == 0:
        strengths.append("immediate joiner")
    elif notice <= 15:
        strengths.append(f"{notice}d notice")
    elif notice <= 30:
        strengths.append(f"≤30d notice")
    elif notice > 90:
        concerns.append(f"{notice}d notice period")

    if otw:
        strengths.append("open-to-work")

    if exp_tier == "consulting":
        concerns.append("consulting-only career (JD disqualifier)")
    elif is_product:
        strengths.append("product company pedigree")

    if t_tier in ("strong_current", "strong_history_2plus"):
        strengths.append("title directly matches JD scope")
    elif t_tier == "adjacent":
        concerns.append("adjacent title — not a direct AI/ML match")

    if neg_hits >= 2:
        concerns.append(f"{neg_hits} CV/robotics/speech skills — potential domain mismatch")

    if not jd_skills:
        concerns.append("no direct JD must-have skills listed")

    sal_max = sal.get("max", 0)
    if sal_max > 95:
        concerns.append(f"salary expectation {sal.get('min',0):.0f}–{sal_max:.0f} LPA may exceed band")

    # ── Sentence 2: tone calibrated to rank tier ─────────────────────────
    if rank <= 10:
        highlight = [x for x in strengths if any(k in x for k in
                     ["GitHub", "response", "notice", "active", "product", "open-to-work"])][:3]
        if not highlight:
            highlight = strengths[:3]
        if concerns:
            s2 = f"{'; '.join(highlight[:2])}. Worth noting: {concerns[0]}."
        elif highlight:
            s2 = f"{'; '.join(highlight[:3])}."
        else:
            s2 = "Ranks in top 10 on combined skill depth, semantic alignment, and availability signals."

    elif rank <= 30:
        if strengths and concerns:
            s2 = f"{'; '.join(strengths[:2])}. Concern: {concerns[0]}."
        elif strengths:
            s2 = f"{'; '.join(strengths[:2])}."
        else:
            s2 = f"{'; '.join(concerns[:2])}." if concerns else "Solid across most signals; marginal on some JD requirements."

    else:
        if concerns and strengths:
            if rank <= 50:
                s2 = f"{strengths[0]}. Held back by: {concerns[0]}."
            elif rank <= 75:
                s2 = f"Gap: {concerns[0]}. {strengths[0]} keeps them in scope."
            else:
                s2 = f"Outside the strong-fit band — {concerns[0]}."
        elif concerns:
            if rank <= 50:
                s2 = f"Main gap: {concerns[0]}."
            elif rank <= 75:
                s2 = f"Ranked lower because {concerns[0]}."
            else:
                s2 = f"Clear disqualifier: {concerns[0]}."
        elif strengths:
            if rank <= 50:
                s2 = f"{strengths[0]}, but JD core (retrieval + ranking + production eval) isn't well evidenced."
            else:
                s2 = f"Positive signal: {strengths[0]}. Weak on JD specifics — no retrieval/ranking production evidence."
        else:
            s2 = "Below threshold on title fit, skills coverage, and behavioral availability."

    return f"{s1} {s2}"
