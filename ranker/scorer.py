"""
ranker/scorer.py  —  8-component candidate scoring engine
Redrob Hackathon  —  Intelligent Candidate Discovery v2

Weights (sum = 1.0):
  title      0.22   title gate scored BEFORE skills
  skills     0.26   proficiency × duration × endorsements
  semantic   0.14   TF-IDF / neural cosine vs JD embedding
  exp        0.16   YoE band, company type, stability
  behavioral 0.12   all 23 redrob_signals
  narrative  0.05   production-evidence regex patterns
  location   0.03   India / tier-1 city preference
  edu_asm    0.02   institution tier + platform assessment scores
"""
from __future__ import annotations
import re
from datetime import date
from typing import Optional

TODAY = date(2026, 6, 11)

# ── JD taxonomy ───────────────────────────────────────────────────────────────
STRONG_TITLES = frozenset({
    "ml engineer","machine learning engineer","senior ml engineer",
    "senior machine learning engineer","staff machine learning engineer",
    "ai engineer","senior ai engineer","lead ai engineer","lead ml engineer",
    "nlp engineer","senior nlp engineer","search engineer","ranking engineer",
    "recommendation systems engineer","applied ml engineer","applied ai engineer",
    "applied scientist","senior applied scientist","research engineer",
    "information retrieval engineer","embedding engineer","retrieval engineer",
    "staff ml engineer","staff ai engineer","principal ml engineer",
    "principal ai engineer","recsys engineer","data scientist",
    "senior data scientist","staff data scientist",
})

ADJACENT_TITLES = frozenset({
    "backend engineer","senior backend engineer","software engineer",
    "senior software engineer","full stack developer","full stack engineer",
    "platform engineer","data engineer","tech lead","lead engineer",
    "python developer","senior python developer","sde","senior sde",
})

DISQUALIFYING_TITLES = frozenset({
    "hr manager","human resources","marketing manager","accountant",
    "finance manager","sales executive","content writer","graphic designer",
    "operations manager","civil engineer","mechanical engineer",
    "project manager","customer support","business analyst","teacher",
    "doctor","lawyer","legal counsel","recruiter","ui designer","ux designer",
    "product designer","qa engineer","quality assurance",
})

CORE_SKILLS = frozenset({
    "sentence-transformers","sentence transformers","openai embeddings",
    "bge","e5","embeddings","vector embeddings","text embeddings",
    "dense retrieval","semantic search","bi-encoder","cross-encoder",
    "pinecone","qdrant","weaviate","milvus","faiss","opensearch",
    "elasticsearch","vector database","vector db","hybrid search",
    "annoy","scann","chromadb","lancedb",
    "ranking","information retrieval","learning to rank","ltr","bm25",
    "hybrid retrieval","ndcg","mrr","map","reranking","recall@k",
    "two-stage retrieval","cross-encoder reranking",
    "nlp","natural language processing","transformers","bert","roberta",
    "fine-tuning","rag","retrieval augmented generation","lora","qlora",
    "peft","llm","large language model","instruction tuning",
    "sentence similarity","text classification","hugging face transformers",
    "python",
    "a/b testing","ab testing","evaluation","offline evaluation",
    "online evaluation","experiment","hypothesis testing",
})

SECONDARY_SKILLS = frozenset({
    "xgboost","lightgbm","catboost","pytorch","tensorflow","mlflow",
    "wandb","weights & biases","ray","triton","onnx","vllm","spark","kafka",
    "distributed systems","docker","kubernetes","redis","feature store",
    "dbt","airflow","data quality","mlops",
})

NEGATIVE_SKILLS = frozenset({
    "computer vision","image classification","object detection","yolo",
    "yolov8","yolov5","image segmentation","ocr","speech recognition",
    "tts","text to speech","asr","automatic speech recognition",
    "robotics","ros","slam","lidar","opengl","unity","unreal engine",
})

CONSULTING_FIRMS = frozenset({
    "tcs","tata consultancy","infosys","wipro","accenture","cognizant",
    "capgemini","hcl","hcl tech","tech mahindra","mphasis","mindtree",
    "hexaware","niit","l&t infotech","mastech","igate","kpit","cyient",
    "zensar","persistent systems","coforge","ltimindtree",
})

PRODUCT_COMPANIES = frozenset({
    "swiggy","zomato","flipkart","amazon","google","microsoft","meta",
    "uber","ola","paytm","phonepe","razorpay","freshworks","zoho",
    "meesho","cred","nykaa","myntra","dream11","groww","slice",
    "cleartax","zerodha","upstox","sarvam","haptik","unacademy","byju",
    "sharechat","mad street den","yellow.ai","rephrase.ai","glance",
    "inmobi","krutrim","netflix","apple","salesforce","upgrad","verloop",
    "vedantu","pharm","saarthi","walmart","adobe","atlassian","stripe",
})

TARGET_CITIES = frozenset({
    "pune","noida","delhi","delhi ncr","gurgaon","gurugram","hyderabad",
    "bangalore","bengaluru","mumbai","chennai","ahmedabad","kolkata",
    "vizag","visakhapatnam","indore","jaipur","chandigarh","bhubaneswar",
    "coimbatore","trivandrum","kochi",
})

NARRATIVE_PHRASES = [
    (r"shipped.*retrieval",              0.15),
    (r"embedding.*production",           0.15),
    (r"retrieval.*production",           0.15),
    (r"migrat.*keyword.*embed",          0.20),
    (r"ranking.*model",                  0.12),
    (r"ndcg|mrr|offline.*eval|a/b.*test",0.12),
    (r"embedding.*drift|index.*refresh", 0.18),
    (r"learning.to.rank|ltr",            0.12),
    (r"hybrid.*search|bm25.*embed",      0.15),
    (r"recall.*precision|relevance.*label",0.12),
    (r"production.*ml|ml.*production",   0.10),
    (r"vector.*search|vector.*db",       0.12),
    (r"fine.tun.*llm|rag.*pipeline",     0.10),
    (r"product.*company|user.*facing",   0.08),
]

EDU_TIER_BONUS = {"tier_1": 0.08, "tier_2": 0.04, "tier_3": 0.01, "tier_4": 0.0}
PROF_WEIGHT    = {"beginner": 0.40, "intermediate": 0.70, "advanced": 0.90, "expert": 1.00}

WEIGHTS = {
    "title":      0.22,
    "skills":     0.26,
    "semantic":   0.14,
    "exp":        0.16,
    "behavioral": 0.12,
    "narrative":  0.05,
    "location":   0.03,
    "edu_asm":    0.02,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


# ── Utilities ─────────────────────────────────────────────────────────────────
def norm(t: str) -> str:
    return t.lower().strip() if t else ""

def days_since(ds: str) -> int:
    try:
        d = date(int(ds[:4]), int(ds[5:7]), int(ds[8:10]))
        return (TODAY - d).days
    except Exception:
        return 9999

def skill_map(c: dict) -> dict[str, dict]:
    return {norm(s["name"]): s for s in c.get("skills", [])}

def full_text(c: dict) -> str:
    parts = [
        c["profile"].get("summary", ""),
        c["profile"].get("headline", ""),
        c["profile"].get("current_title", ""),
    ]
    for j in c.get("career_history", []):
        parts += [j.get("description", ""), j.get("title", ""), j.get("company", "")]
    for s in c.get("skills", []):
        parts.append(s.get("name", ""))
    for cert in c.get("certifications", []):
        parts.append(cert.get("name", ""))
    return " ".join(parts).lower()


# ── Honeypot Detection ────────────────────────────────────────────────────────
def is_honeypot(c: dict) -> tuple[bool, str]:
    hist   = c.get("career_history", [])
    yoe    = c["profile"].get("years_of_experience", 0)
    skills = c.get("skills", [])
    sig    = c.get("redrob_signals", {})

    total_months = sum(j.get("duration_months", 0) for j in hist)
    if total_months > 0 and yoe > (total_months / 12) * 1.6 + 3:
        return True, f"yoe={yoe:.1f} but career={total_months}mo"

    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    if expert_zero >= 5:
        return True, f"{expert_zero} expert/0-duration skills"

    if len(skills) > 28 and yoe < 2:
        return True, f"{len(skills)} skills at {yoe:.1f}y"

    rr  = sig.get("recruiter_response_rate", 0)
    icr = sig.get("interview_completion_rate", 0)
    gh  = sig.get("github_activity_score", -1)
    pc  = sig.get("profile_completeness_score", 0)
    if rr == 1.0 and icr == 1.0 and gh == 100 and pc == 100:
        return True, "all signals at ceiling — synthetic artifact"

    signup      = sig.get("signup_date", "2000-01-01")
    last_active = sig.get("last_active_date", "2000-01-01")
    if signup > last_active and last_active != "2000-01-01":
        return True, f"signup={signup} after last_active={last_active}"

    return False, ""


# ── Component Scorers ─────────────────────────────────────────────────────────
def score_title(c: dict) -> tuple[float, str]:
    title      = norm(c["profile"].get("current_title", ""))
    hist_titles = [norm(j.get("title", "")) for j in c.get("career_history", [])]

    for dq in DISQUALIFYING_TITLES:
        if dq in title:
            return 0.04, "disqualified"

    if any(kw in title for kw in STRONG_TITLES):
        return 1.0, "strong_current"

    strong_hist = sum(1 for t in hist_titles if any(kw in t for kw in STRONG_TITLES))
    if strong_hist >= 2: return 0.90, "strong_history_2plus"
    if strong_hist == 1: return 0.75, "strong_history_once"

    if any(kw in title for kw in ADJACENT_TITLES):
        return 0.52, "adjacent"

    if "engineer" in title or "developer" in title or "architect" in title:
        return 0.28, "generic_tech"

    return 0.08, "weak"


def score_skills(c: dict, ft: str) -> tuple[float, dict]:
    smap       = skill_map(c)
    core_score = 0.0
    matched    = []

    for kw in CORE_SKILLS:
        if kw in smap:
            s   = smap[kw]
            pw  = PROF_WEIGHT.get(s.get("proficiency", "intermediate"), 0.7)
            dw  = min(s.get("duration_months", 0) / 36.0, 1.0)
            ew  = min(s.get("endorsements", 0) / 30.0, 1.0) * 0.08
            core_score += (0.55 * pw + 0.37 * dw + 0.08) + ew
            matched.append(kw)
        elif kw in ft:
            core_score += 0.20
            matched.append(f"~{kw}")

    core_norm = min(core_score / 11.0, 1.0)
    sec       = sum(1 for kw in SECONDARY_SKILLS if kw in smap or kw in ft)
    sec_score = min(sec / 6.0, 1.0) * 0.12
    py_bonus  = 0.10 if "python" in smap else (0.04 if "python" in ft else 0.0)
    neg       = sum(1 for kw in NEGATIVE_SKILLS if kw in smap)
    neg_pen   = min(neg * 0.04, 0.16)

    rel_end   = sum(smap[k].get("endorsements", 0) for k in matched if k in smap)
    end_bonus = min(rel_end / 400.0, 0.05)

    final = min(core_norm + sec_score + py_bonus + end_bonus - neg_pen, 1.0)
    return max(final, 0.0), {
        "matched":       matched,
        "sec_hits":      sec,
        "neg_hits":      neg,
        "matched_clean": [m for m in matched if not m.startswith("~")],
    }


def score_experience(c: dict) -> tuple[float, str]:
    yoe  = c["profile"].get("years_of_experience", 0)
    hist = c.get("career_history", [])
    n    = max(len(hist), 1)

    if   5 <= yoe <= 9:   exp_base = 1.00
    elif 4 <= yoe <  5:   exp_base = 0.85
    elif 9 <  yoe <= 12:  exp_base = 0.78
    elif 3 <= yoe <  4:   exp_base = 0.65
    elif yoe > 12:        exp_base = 0.62
    else:                 exp_base = max(yoe / 5.0 * 0.60, 0.0)

    companies = [norm(j.get("company", "")) for j in hist]

    consulting_ratio = sum(
        1 for co in companies if any(f in co for f in CONSULTING_FIRMS)
    ) / n
    product_bonus = 0.10 if any(
        any(p in co for p in PRODUCT_COMPANIES) for co in companies
    ) else 0.0

    if   consulting_ratio >= 0.90: company_pen = 0.35
    elif consulting_ratio >= 0.50: company_pen = 0.12
    else:                          company_pen = 0.0

    short_stints = sum(
        1 for j in hist
        if j.get("duration_months", 24) < 14 and not j.get("is_current", False)
    )
    hop_pen = min(short_stints * 0.06, 0.20)

    research_pen = 0.0
    research_sigs = {"research","academic","phd","postdoc","professor","lab","institute"}
    if hist and all(
        any(rs in norm(j.get("industry", "")) for rs in research_sigs)
        for j in hist
    ):
        research_pen = 0.25

    edu_bonus = max(
        (EDU_TIER_BONUS.get(e.get("tier", ""), 0.0) for e in c.get("education", [])),
        default=0.0,
    )

    tier  = "consulting" if consulting_ratio >= 0.9 else ("product" if product_bonus > 0 else "mixed")
    final = max(exp_base + product_bonus + edu_bonus - company_pen - hop_pen - research_pen, 0.0)
    return min(final, 1.0), tier


def score_behavioral(c: dict) -> tuple[float, dict]:
    sig = c.get("redrob_signals", {})

    inactive = days_since(sig.get("last_active_date", "2020-01-01"))
    if   inactive <= 14:  rec = 1.00
    elif inactive <= 30:  rec = 0.92
    elif inactive <= 60:  rec = 0.78
    elif inactive <= 90:  rec = 0.62
    elif inactive <= 180: rec = 0.42
    elif inactive <= 365: rec = 0.22
    else:                 rec = 0.08

    rr  = max(0.0, min(1.0, float(sig.get("recruiter_response_rate", 0.5))))

    rt  = sig.get("avg_response_time_hours", 72)
    if   rt <= 6:   rt_s = 1.00
    elif rt <= 24:  rt_s = 0.90
    elif rt <= 72:  rt_s = 0.72
    elif rt <= 168: rt_s = 0.50
    else:           rt_s = 0.28

    notice = sig.get("notice_period_days", 60)
    if   notice == 0:   not_s = 1.00
    elif notice <= 15:  not_s = 0.98
    elif notice <= 30:  not_s = 0.90
    elif notice <= 45:  not_s = 0.82
    elif notice <= 60:  not_s = 0.72
    elif notice <= 90:  not_s = 0.55
    else:               not_s = 0.38

    otw = 0.12 if sig.get("open_to_work_flag", False) else 0.0
    icr = max(0.0, min(1.0, float(sig.get("interview_completion_rate", 0.7))))

    gh = sig.get("github_activity_score", -1)
    if   gh == -1: gh_s = 0.38
    elif gh >= 75: gh_s = 1.00
    elif gh >= 50: gh_s = 0.82
    elif gh >= 25: gh_s = 0.62
    elif gh >= 10: gh_s = 0.45
    else:          gh_s = 0.30

    sal     = sig.get("expected_salary_range_inr_lpa", {})
    sal_mid = (sal.get("min", 0) + sal.get("max", 0)) / 2 if sal.get("max", 0) > 0 else 0
    if   25 <= sal_mid <= 65:  sal_s = 1.00
    elif 20 <= sal_mid <= 80:  sal_s = 0.85
    elif sal_mid > 80:         sal_s = 0.62
    elif sal_mid > 0:          sal_s = 0.75
    else:                      sal_s = 0.70

    trust  = (
        (0.04 if sig.get("verified_email")    else 0) +
        (0.03 if sig.get("verified_phone")    else 0) +
        (0.03 if sig.get("linkedin_connected") else 0)
    )
    saves  = min(sig.get("saved_by_recruiters_30d", 0), 20)
    views  = min(sig.get("profile_views_received_30d", 0), 50)
    demand = min(saves * 0.012 + views * 0.002, 0.07)

    oar   = sig.get("offer_acceptance_rate", -1)
    oar_s = 0.04 if oar >= 0.7 else (0.02 if oar >= 0.3 else (0.01 if oar == -1 else 0.005))

    score = (
        rec   * 0.23 +
        rr    * 0.20 +
        rt_s  * 0.07 +
        not_s * 0.13 +
        icr   * 0.10 +
        gh_s  * 0.08 +
        sal_s * 0.07 +
        trust + demand + oar_s + otw
    )
    return min(score, 1.0), {
        "recency":       round(rec,   3),
        "response_rate": round(rr,    3),
        "notice_days":   notice,
        "github":        round(gh_s,  3),
        "salary_fit":    round(sal_s, 3),
        "days_inactive": inactive,
        "trust":         round(trust, 3),
    }


def score_narrative(c: dict) -> float:
    ft    = full_text(c)
    total = 0.0
    for pattern, weight in NARRATIVE_PHRASES:
        if re.search(pattern, ft):
            total += weight

    research_words   = ["publication","paper","arxiv","theorem","proof","lab","phd thesis"]
    production_words = ["shipped","production","users","deployed","scale","traffic","latency"]
    rc = sum(1 for w in research_words   if w in ft)
    pc = sum(1 for w in production_words if w in ft)
    if rc > 3 and pc == 0:
        total -= 0.20

    return max(0.0, min(1.0, total))


def score_location(c: dict) -> float:
    sig     = c.get("redrob_signals", {})
    loc     = norm(c["profile"].get("location", ""))
    country = norm(c["profile"].get("country", ""))
    reloc   = sig.get("willing_to_relocate", False)
    mode    = norm(sig.get("preferred_work_mode", ""))

    if any(city in loc for city in ["pune", "noida", "delhi"]):
        return 1.0
    if any(city in loc for city in TARGET_CITIES):
        return 0.90
    if "india" in country or "india" in loc:
        return 0.78 if reloc else 0.65
    if reloc:
        return 0.48
    if "remote" in mode or "flexible" in mode:
        return 0.38
    return 0.20


def score_edu_assessment(c: dict) -> float:
    edu = c.get("education", [])
    sig = c.get("redrob_signals", {})
    cs_fields = {
        "computer","machine learning","data science","artificial intelligence",
        "statistics","mathematics","information technology",
    }

    edu_score = 0.40
    for e in edu:
        field = norm(e.get("field_of_study", ""))
        tier  = e.get("tier", "unknown")
        fm    = any(f in field for f in cs_fields)
        tv    = {"tier_1":1.0,"tier_2":0.80,"tier_3":0.60,"tier_4":0.40,"unknown":0.50}.get(tier, 0.50)
        edu_score = max(edu_score, 0.7 * tv + 0.3 * (1.0 if fm else 0.4))

    assessments = sig.get("skill_assessment_scores", {})
    rel_keys    = {
        "nlp","machine learning","python","deep learning","pytorch","tensorflow",
        "transformers","fine-tuning llms","rag","information retrieval",
        "vector search","embeddings","ranking","recommendation systems",
    }
    rel_scores = [v for k, v in assessments.items() if any(r in norm(k) for r in rel_keys)]
    all_scores = list(assessments.values())
    scores     = rel_scores if rel_scores else all_scores
    asm_score  = (sum(scores) / len(scores) / 100.0) if scores else 0.50

    return 0.5 * edu_score + 0.5 * asm_score


# ── Composite ─────────────────────────────────────────────────────────────────
def score_candidate(
    c: dict,
    semantic_score: Optional[float] = None,
    phrase_score:   Optional[float] = None,
) -> tuple[float, dict]:
    hp, hp_reason = is_honeypot(c)
    if hp:
        return 0.0, {"honeypot": True, "honeypot_reason": hp_reason}

    ft = full_text(c)

    t_score,  t_tier    = score_title(c)
    sk_score, sk_detail = score_skills(c, ft)
    exp_score, exp_tier = score_experience(c)
    beh_score, beh_det  = score_behavioral(c)
    nar_score           = score_narrative(c)
    loc_score           = score_location(c)
    edu_score           = score_edu_assessment(c)

    sem = 0.50 if semantic_score is None else semantic_score
    phr = 0.0  if phrase_score   is None else phrase_score
    sem_combined = 0.75 * sem + 0.25 * phr

    if t_tier == "disqualified" and sk_score < 0.30:
        return max(t_score * sk_score, 0.01), {
            "honeypot": False, "early_reject": True,
            "title": t_score, "skills": sk_score, "title_tier": t_tier,
        }

    composite = (
        t_score      * WEIGHTS["title"]     +
        sk_score     * WEIGHTS["skills"]    +
        sem_combined * WEIGHTS["semantic"]  +
        exp_score    * WEIGHTS["exp"]       +
        beh_score    * WEIGHTS["behavioral"]+
        nar_score    * WEIGHTS["narrative"] +
        loc_score    * WEIGHTS["location"]  +
        edu_score    * WEIGHTS["edu_asm"]
    )

    return round(composite, 5), {
        "honeypot":        False,
        "early_reject":    False,
        "title":           round(t_score,   4),
        "title_tier":      t_tier,
        "skills":          round(sk_score,  4),
        "skills_matched":  sk_detail["matched_clean"][:8],
        "skills_neg_hits": sk_detail["neg_hits"],
        "semantic":        round(sem_combined, 4),
        "experience":      round(exp_score, 4),
        "exp_tier":        exp_tier,
        "behavioral":      round(beh_score, 4),
        "beh_detail":      beh_det,
        "narrative":       round(nar_score, 4),
        "location":        round(loc_score, 4),
        "edu_asm":         round(edu_score, 4),
    }
