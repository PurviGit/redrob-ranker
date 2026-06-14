"""
ranker/reasoning.py  —  Stage-4-quality per-candidate reasoning.

Every claim references a real field from the candidate profile.
No hallucination: skills named must exist in candidate.skills,
achievements quoted must appear in career_history descriptions.
Spec Stage-4 checks: specific facts, JD connection, honest concerns,
no hallucination, variation, rank-consistent tone.
"""
from __future__ import annotations
import re
from ranker.scorer import (
    norm, days_since, full_text,
    CONSULTING_FIRMS, PRODUCT_COMPANIES, CORE_SKILLS,
)

# JD "must-have" skills — naming these in reasoning = explicit JD connection
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

# Career text patterns that prove production experience (JD cares about this most)
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


def _extract_career_achievement(hist: list[dict]) -> str:
    """Pull one concrete, quantified achievement from career descriptions."""
    for job in hist[:3]:
        desc = job.get("description", "")
        for pat, _ in _PRODUCTION_PATTERNS:
            m = re.search(pat, desc, re.IGNORECASE)
            if m:
                # Find sentence start (last '. ' or start of string before match)
                before = desc[:m.start()]
                sent_start = max(before.rfind(". "), before.rfind(".\n"), before.rfind(": "))
                sent_start = sent_start + 2 if sent_start >= 0 else 0
                # Find sentence end
                after      = desc[m.end():]
                end_offset = min(
                    (after.find(". ") if after.find(". ") >= 0 else 200),
                    (after.find(".\n") if after.find(".\n") >= 0 else 200),
                    120,
                )
                snippet = desc[sent_start : m.end() + end_offset].strip().rstrip(".,;")
                if len(snippet) > 30:
                    # Cap at ~120 chars to keep reasoning readable
                    return snippet[:120].strip()
    return ""


def _jd_matched_skills(candidate: dict) -> list[str]:
    """Skills in the profile that are explicitly in the JD must-have list."""
    named = []
    for s in candidate.get("skills", []):
        name = s.get("name", "").lower()
        if any(jd in name or name in jd for jd in JD_MUST_HAVE):
            named.append(s["name"])
        if len(named) >= 4:
            break
    return named


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

    loc_str    = loc.split(",")[0].strip()
    ft         = full_text(candidate)
    jd_skills  = _jd_matched_skills(candidate)
    achievement = _extract_career_achievement(hist)

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

    # ── Sentence 1: identity + most specific technical evidence ─────────
    s1 = f"{title} ({yoe:.1f}yr) at {company}, {loc_str}."

    if achievement:
        s1 += f" Career: {achievement}."
    elif jd_skills:
        s1 += f" JD-aligned skills: {', '.join(jd_skills[:3])}."
    elif components.get("skills_matched"):
        matched = components["skills_matched"][:3]
        s1 += f" Relevant skills: {', '.join(matched)}."

    # ── Gather strengths and concerns (signal-value specific) ───────────
    strengths, concerns = [], []

    # Recency
    if inactive <= 7:
        strengths.append(f"active {inactive}d ago")
    elif inactive <= 30:
        strengths.append(f"active within {inactive}d")
    elif inactive > 180:
        concerns.append(f"inactive {inactive}d — reachability risk")

    # Response rate with value
    if rr is not None:
        if rr >= 0.70:
            strengths.append(f"{rr:.0%} response rate")
        elif rr < 0.30:
            concerns.append(f"low response rate ({rr:.0%})")

    # GitHub with actual score
    if gh >= 70:
        strengths.append(f"GitHub {gh:.0f}/100")
    elif gh >= 40:
        strengths.append(f"GitHub {gh:.0f}/100")
    elif gh == -1:
        concerns.append("no GitHub linked")

    # Notice
    if notice == 0:
        strengths.append("immediate joiner")
    elif notice <= 15:
        strengths.append(f"{notice}d notice")
    elif notice <= 30:
        strengths.append(f"≤30d notice")
    elif notice > 90:
        concerns.append(f"{notice}d notice period")

    # Open to work
    if otw:
        strengths.append("open-to-work")

    # Company type — JD explicitly penalises consulting-only
    if exp_tier == "consulting":
        concerns.append("consulting-only career (JD disqualifier)")
    elif is_product:
        strengths.append("product company pedigree")

    # Title match
    if t_tier in ("strong_current", "strong_history_2plus"):
        strengths.append("title directly matches JD scope")
    elif t_tier == "adjacent":
        concerns.append("adjacent title — not a direct AI/ML match")

    # Domain mismatch
    if neg_hits >= 2:
        concerns.append(f"{neg_hits} CV/robotics/speech skills — potential domain mismatch")

    # JD skill coverage
    if jd_skills and len(jd_skills) >= 3:
        strengths.append(f"JD skills covered: {', '.join(jd_skills[:3])}")
    elif not jd_skills:
        concerns.append("no direct JD must-have skills in profile")

    # Salary
    sal_max = sal.get("max", 0)
    if sal_max > 95:
        concerns.append(f"salary {sal.get('min',0):.0f}–{sal_max:.0f} LPA may exceed band")

    # ── Sentence 2: tone calibrated to rank tier ─────────────────────────
    if rank <= 10:
        top_s = [x for x in strengths if any(k in x for k in
                 ["GitHub", "response", "notice", "JD skills", "active", "product"])][:3]
        if not top_s:
            top_s = strengths[:3]
        if concerns:
            s2 = f"Strong match: {'; '.join(top_s[:2])}. Note: {concerns[0]}."
        elif top_s:
            s2 = f"Strong across all JD dimensions: {'; '.join(top_s[:3])}."
        else:
            s2 = "Ranks highest on combined JD skill alignment and behavioral signals."

    elif rank <= 30:
        if strengths and concerns:
            s2 = f"Strengths: {'; '.join(strengths[:2])}. Concern: {concerns[0]}."
        elif strengths:
            s2 = f"Good fit: {'; '.join(strengths[:2])}."
        else:
            s2 = f"Concern: {'; '.join(concerns[:2])}."

    else:
        if concerns:
            s2 = f"Below cutoff: {'; '.join(concerns[:2])}."
        elif strengths:
            s2 = f"Borderline — {strengths[0]}; marginal JD alignment overall."
        else:
            s2 = "Borderline fit — marginal alignment across all JD signals."

    return f"{s1} {s2}"
