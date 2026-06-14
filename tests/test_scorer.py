"""
tests/test_scorer.py  —  Full test suite (40 tests)
Run: pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from ranker.scorer import (
    score_candidate, score_title, score_skills, score_experience,
    score_behavioral, score_narrative, score_location, is_honeypot,
    norm, skill_map, full_text, WEIGHTS,
)
from ranker.reasoning import generate
from ranker.semantic  import SemanticScorer, build_candidate_semantic_text, JD_SEMANTIC_DOCUMENT

# ── fixture helpers ───────────────────────────────────────────────────────────
_SENTINEL = object()

def make_candidate(
    title="Machine Learning Engineer",
    yoe=6.5,
    location="Pune, Maharashtra",
    country="India",
    company="Swiggy",
    company_size="5001-10000",
    industry="Food Delivery",
    skills=_SENTINEL,
    history=_SENTINEL,
    signals=_SENTINEL,
    education=_SENTINEL,
    summary="ML engineer with production experience in NLP, embeddings, and ranking systems.",
):
    _skills = [
        {"name":"Python",                "proficiency":"expert",       "endorsements":50,"duration_months":60},
        {"name":"NLP",                   "proficiency":"advanced",     "endorsements":30,"duration_months":36},
        {"name":"FAISS",                 "proficiency":"advanced",     "endorsements":20,"duration_months":24},
        {"name":"Pinecone",              "proficiency":"expert",       "endorsements":25,"duration_months":30},
        {"name":"Semantic Search",       "proficiency":"advanced",     "endorsements":15,"duration_months":20},
        {"name":"PyTorch",               "proficiency":"advanced",     "endorsements":18,"duration_months":30},
        {"name":"Fine-tuning",           "proficiency":"intermediate", "endorsements":10,"duration_months":12},
        {"name":"RAG",                   "proficiency":"intermediate", "endorsements": 8,"duration_months":10},
        {"name":"Embeddings",            "proficiency":"expert",       "endorsements":35,"duration_months":48},
        {"name":"Information Retrieval", "proficiency":"expert",       "endorsements":20,"duration_months":60},
    ]
    _signals = {
        "profile_completeness_score":85, "signup_date":"2022-01-01",
        "last_active_date":"2026-05-20", "open_to_work_flag":True,
        "profile_views_received_30d":15, "applications_submitted_30d":3,
        "recruiter_response_rate":0.75,  "avg_response_time_hours":18,
        "skill_assessment_scores":{"nlp":82,"python":90},
        "connection_count":200,          "endorsements_received":120,
        "notice_period_days":30,
        "expected_salary_range_inr_lpa":{"min":30,"max":50},
        "preferred_work_mode":"hybrid",  "willing_to_relocate":True,
        "github_activity_score":65,      "search_appearance_30d":20,
        "saved_by_recruiters_30d":5,     "interview_completion_rate":0.85,
        "offer_acceptance_rate":0.8,     "verified_email":True,
        "verified_phone":True,           "linkedin_connected":True,
    }
    _history = [{
        "company":company, "title":title,
        "start_date":"2021-01-01", "end_date":None,
        "duration_months":42, "is_current":True,
        "industry":industry, "company_size":company_size,
        "description": (
            "Shipped embedding-based retrieval to production. "
            "Built hybrid BM25 + dense retrieval system. "
            "Evaluated with NDCG and MRR. Led A/B testing framework."
        ),
    }]
    _edu = [{
        "institution":"IIT Bombay","degree":"B.Tech",
        "field_of_study":"Computer Science",
        "start_year":2014,"end_year":2018,"grade":"8.5","tier":"tier_1",
    }]

    return {
        "candidate_id": "CAND_TEST001",
        "profile": {
            "anonymized_name":"Test Candidate",
            "headline":"ML Engineer | Search | Embeddings",
            "summary":summary, "location":location, "country":country,
            "years_of_experience":yoe, "current_title":title,
            "current_company":company, "current_company_size":company_size,
            "current_industry":industry,
        },
        "career_history": history    if history    is not _SENTINEL else _history,
        "education":      education  if education  is not _SENTINEL else _edu,
        "skills":         skills     if skills     is not _SENTINEL else _skills,
        "certifications": [],
        "languages":      [{"language":"English","proficiency":"professional"}],
        "redrob_signals": signals    if signals    is not _SENTINEL else _signals,
    }


_WEAK_SIG = {
    "profile_completeness_score":60, "signup_date":"2021-01-01",
    "last_active_date":"2026-05-01", "open_to_work_flag":False,
    "profile_views_received_30d":3,  "applications_submitted_30d":1,
    "recruiter_response_rate":0.40,  "avg_response_time_hours":72,
    "skill_assessment_scores":{},    "connection_count":80,
    "endorsements_received":30,      "notice_period_days":60,
    "expected_salary_range_inr_lpa":{"min":30,"max":50},
    "preferred_work_mode":"hybrid",  "willing_to_relocate":True,
    "github_activity_score":20,      "search_appearance_30d":5,
    "saved_by_recruiters_30d":1,     "interview_completion_rate":0.55,
    "offer_acceptance_rate":0.40,    "verified_email":True,
    "verified_phone":False,          "linkedin_connected":False,
}


# ── title ─────────────────────────────────────────────────────────────────────
def test_ml_engineer_strong():
    s, t = score_title(make_candidate("Machine Learning Engineer"))
    assert s == 1.0 and t == "strong_current"

def test_recsys_engineer_strong():
    s, _ = score_title(make_candidate("Recommendation Systems Engineer"))
    assert s == 1.0

def test_nlp_engineer_strong():
    s, _ = score_title(make_candidate("NLP Engineer"))
    assert s == 1.0

def test_hr_manager_disqualified():
    s, t = score_title(make_candidate("HR Manager"))
    assert s < 0.10 and t == "disqualified"

def test_accountant_disqualified():
    s, _ = score_title(make_candidate("Accountant"))
    assert s < 0.10

def test_backend_engineer_adjacent():
    s, t = score_title(make_candidate("Backend Engineer"))
    assert 0.40 < s < 0.65 and t == "adjacent"

def test_strong_history_trajectory():
    hist = [
        {"company":"X","title":"ML Engineer","start_date":"2019-01-01","end_date":"2022-01-01",
         "duration_months":36,"is_current":False,"industry":"AI","company_size":"51-200",
         "description":"Built ranking systems."},
        {"company":"Y","title":"Senior NLP Engineer","start_date":"2022-01-01","end_date":None,
         "duration_months":30,"is_current":True,"industry":"AI","company_size":"201-500",
         "description":"Production NLP."},
    ]
    c = make_candidate("Data Analyst", history=hist)
    s, _ = score_title(c)
    assert s >= 0.85


# ── skills ────────────────────────────────────────────────────────────────────
def test_rich_ml_skills_score_high():
    c = make_candidate()
    s, _ = score_skills(c, full_text(c))
    assert s > 0.55

def test_python_bonus_applied():
    c_py   = make_candidate(skills=[{"name":"Python","proficiency":"expert","endorsements":20,"duration_months":36}])
    c_nopy = make_candidate(skills=[{"name":"Java",  "proficiency":"expert","endorsements":20,"duration_months":36}])
    s_py,  _ = score_skills(c_py,   full_text(c_py))
    s_no,  _ = score_skills(c_nopy, full_text(c_nopy))
    assert s_py > s_no

def test_expert_long_beats_beginner_short():
    c_exp = make_candidate(skills=[{"name":"FAISS","proficiency":"expert",  "endorsements":30,"duration_months":48}])
    c_beg = make_candidate(skills=[{"name":"FAISS","proficiency":"beginner","endorsements": 2,"duration_months": 2}])
    s_e, _ = score_skills(c_exp, full_text(c_exp))
    s_b, _ = score_skills(c_beg, full_text(c_beg))
    assert s_e > s_b

def test_cv_robotics_penalised():
    c = make_candidate(skills=[
        {"name":"YOLO",             "proficiency":"expert","endorsements":20,"duration_months":36},
        {"name":"Computer Vision",  "proficiency":"expert","endorsements":20,"duration_months":36},
        {"name":"Object Detection", "proficiency":"expert","endorsements":20,"duration_months":36},
    ])
    _, d = score_skills(c, full_text(c))
    assert d["neg_hits"] >= 3

def test_endorsement_bonus_applies():
    c_hi = make_candidate(skills=[{"name":"Embeddings","proficiency":"advanced","endorsements":100,"duration_months":36}])
    c_lo = make_candidate(skills=[{"name":"Embeddings","proficiency":"advanced","endorsements":  2,"duration_months":36}])
    s_hi, _ = score_skills(c_hi, full_text(c_hi))
    s_lo, _ = score_skills(c_lo, full_text(c_lo))
    assert s_hi > s_lo


# ── experience ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("yoe", [5.0, 6.5, 8.0, 9.0])
def test_ideal_yoe_scores_high(yoe):
    s, _ = score_experience(make_candidate(yoe=yoe))
    assert s >= 0.85

def test_under_2_years_scores_low():
    s, _ = score_experience(make_candidate(yoe=1.5))
    assert s < 0.45

def test_consulting_only_penalised():
    hist = [
        {"company":"TCS",    "title":"ML Engineer",   "start_date":"2018-01-01","end_date":"2021-01-01",
         "duration_months":36,"is_current":False,"industry":"IT Services","company_size":"10001+","description":"…"},
        {"company":"Infosys","title":"Data Scientist","start_date":"2021-01-01","end_date":None,
         "duration_months":30,"is_current":True, "industry":"IT Services","company_size":"10001+","description":"…"},
    ]
    s, tier = score_experience(make_candidate(yoe=5.5, history=hist))
    assert tier == "consulting" and s < 0.75

def test_product_company_bonus():
    h_prod = [{"company":"Swiggy","title":"ML Engineer","start_date":"2020-01-01","end_date":None,
               "duration_months":54,"is_current":True,"industry":"Food Delivery","company_size":"5001-10000","description":"…"}]
    h_cons = [{"company":"Wipro", "title":"ML Engineer","start_date":"2020-01-01","end_date":None,
               "duration_months":54,"is_current":True,"industry":"IT Services", "company_size":"10001+","description":"…"}]
    s_p, _ = score_experience(make_candidate(yoe=6.0, history=h_prod))
    s_c, _ = score_experience(make_candidate(yoe=6.0, history=h_cons))
    assert s_p > s_c


# ── behavioral ────────────────────────────────────────────────────────────────
def test_recently_active_scores_higher():
    sig_new = {**_WEAK_SIG, "last_active_date":"2026-06-08"}
    sig_old = {**_WEAK_SIG, "last_active_date":"2025-06-08"}
    s_new, _ = score_behavioral(make_candidate(signals=sig_new))
    s_old, _ = score_behavioral(make_candidate(signals=sig_old))
    assert s_new > s_old

def test_short_notice_beats_long():
    s_s, _ = score_behavioral(make_candidate(signals={**_WEAK_SIG, "notice_period_days":15}))
    s_l, _ = score_behavioral(make_candidate(signals={**_WEAK_SIG, "notice_period_days":150}))
    assert s_s > s_l

def test_high_response_rate_scores_better():
    s_h, _ = score_behavioral(make_candidate(signals={**_WEAK_SIG, "recruiter_response_rate":0.95}))
    s_l, _ = score_behavioral(make_candidate(signals={**_WEAK_SIG, "recruiter_response_rate":0.05}))
    assert s_h > s_l

def test_expensive_salary_penalised():
    s_n, _ = score_behavioral(make_candidate(signals={**_WEAK_SIG, "expected_salary_range_inr_lpa":{"min":30, "max":50}}))
    s_e, _ = score_behavioral(make_candidate(signals={**_WEAK_SIG, "expected_salary_range_inr_lpa":{"min":100,"max":150}}))
    assert s_n > s_e

def test_github_ordering():
    _, beh_h = score_behavioral(make_candidate(signals={**_WEAK_SIG, "github_activity_score":80}))
    _, beh_l = score_behavioral(make_candidate(signals={**_WEAK_SIG, "github_activity_score": 5}))
    _, beh_n = score_behavioral(make_candidate(signals={**_WEAK_SIG, "github_activity_score":-1}))
    assert beh_h["github"] > beh_n["github"] > beh_l["github"]


# ── narrative ─────────────────────────────────────────────────────────────────
def test_production_phrases_score_high():
    c = make_candidate(
        summary="I shipped embedding-based retrieval to production. "
                "Led A/B testing with NDCG. Migrated keyword search to embedding-based system."
    )
    assert score_narrative(c) > 0.25

def test_pure_research_penalised():
    c = make_candidate(
        summary="Published 5 arxiv papers on transformer probing. Theorem on attention complexity. Lab-based.",
        history=[{
            "company":"IIT Lab","title":"Research Scholar",
            "start_date":"2019-01-01","end_date":None,
            "duration_months":60,"is_current":True,
            "industry":"Research","company_size":"1-50",
            "description":"Proved theorems, wrote papers.",
        }],
    )
    assert score_narrative(c) < 0.20

def test_narrative_bounded():
    s = score_narrative(make_candidate())
    assert 0.0 <= s <= 1.0


# ── location ──────────────────────────────────────────────────────────────────
def test_pune_max():
    assert score_location(make_candidate(location="Pune, Maharashtra", country="India")) == 1.0

def test_noida_max():
    assert score_location(make_candidate(location="Noida, UP", country="India")) == 1.0

def test_bangalore_high():
    assert score_location(make_candidate(location="Bangalore, Karnataka", country="India")) >= 0.85

def test_outside_india_no_reloc_low():
    sig = {**_WEAK_SIG, "willing_to_relocate":False}
    assert score_location(make_candidate(location="Toronto, ON", country="Canada", signals=sig)) < 0.35

def test_outside_india_reloc_medium():
    sig = {**_WEAK_SIG, "willing_to_relocate":True}
    s   = score_location(make_candidate(location="London, UK", country="UK", signals=sig))
    assert 0.40 <= s <= 0.60


# ── honeypot ──────────────────────────────────────────────────────────────────
def test_normal_not_honeypot():
    hp, _ = is_honeypot(make_candidate())
    assert not hp

def test_impossible_yoe():
    c = make_candidate(yoe=15.0, history=[{
        "company":"X","title":"T","start_date":"2023-01-01","end_date":None,
        "duration_months":18,"is_current":True,"industry":"AI","company_size":"51-200","description":"…",
    }])
    hp, reason = is_honeypot(c)
    assert hp and "yoe" in reason

def test_expert_zero_duration():
    skills = [{"name":f"S{i}","proficiency":"expert","endorsements":10,"duration_months":0} for i in range(7)]
    hp, _  = is_honeypot(make_candidate(skills=skills))
    assert hp

def test_too_many_skills_low_yoe():
    skills = [{"name":f"S{i}","proficiency":"intermediate","endorsements":5,"duration_months":6} for i in range(30)]
    hp, _  = is_honeypot(make_candidate(yoe=1.5, skills=skills))
    assert hp

def test_all_signals_ceiling():
    sig = {**_WEAK_SIG, "recruiter_response_rate":1.0,"interview_completion_rate":1.0,
           "github_activity_score":100,"profile_completeness_score":100}
    hp, _ = is_honeypot(make_candidate(signals=sig))
    assert hp

def test_signup_after_last_active():
    sig = {**_WEAK_SIG, "signup_date":"2026-05-01","last_active_date":"2025-01-01"}
    hp, _ = is_honeypot(make_candidate(signals=sig))
    assert hp


# ── composite ─────────────────────────────────────────────────────────────────
def test_strong_candidate_above_75():
    s, _ = score_candidate(make_candidate())
    assert s > 0.75

def test_hr_manager_very_low():
    c = make_candidate(
        title="HR Manager",
        summary="Experienced HR professional managing talent acquisition.",
        skills=[{"name":"Python","proficiency":"expert","endorsements":10,"duration_months":12}],
        history=[{
            "company":"Infosys","title":"HR Manager","start_date":"2018-01-01","end_date":None,
            "duration_months":60,"is_current":True,"industry":"Human Resources",
            "company_size":"10001+","description":"Managed employee onboarding and payroll.",
        }],
    )
    s, _ = score_candidate(c)
    assert s < 0.25

def test_honeypot_scores_zero():
    skills = [{"name":f"S{i}","proficiency":"expert","endorsements":5,"duration_months":0} for i in range(7)]
    s, comps = score_candidate(make_candidate(skills=skills))
    assert s == 0.0 and comps["honeypot"]

def test_score_in_valid_range():
    s, _ = score_candidate(make_candidate())
    assert 0.0 <= s <= 1.0

def test_semantic_score_improves_result():
    c = make_candidate()
    s_lo, _ = score_candidate(c, semantic_score=0.10, phrase_score=0.0)
    s_hi, _ = score_candidate(c, semantic_score=0.90, phrase_score=1.0)
    assert s_hi > s_lo

def test_monotonic_across_quality():
    s_e, _ = score_candidate(make_candidate(title="Senior ML Engineer", yoe=7.0))
    s_d, _ = score_candidate(make_candidate(title="Backend Engineer",   yoe=5.0))
    s_p, _ = score_candidate(make_candidate(title="Accountant",         yoe=10.0))
    assert s_e > s_d > s_p

def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


# ── reasoning ─────────────────────────────────────────────────────────────────
def test_reasoning_not_empty():
    c = make_candidate()
    s, comps = score_candidate(c)
    assert len(generate(c, 1, s, comps)) > 30

def test_reasoning_mentions_title():
    c = make_candidate(title="Recommendation Systems Engineer")
    s, comps = score_candidate(c)
    r = generate(c, 1, s, comps)
    assert "Recommendation" in r or "Systems" in r

def test_reasoning_references_company():
    c = make_candidate(company="Swiggy")
    s, comps = score_candidate(c)
    assert "Swiggy" in generate(c, 1, s, comps)

def test_honeypot_reasoning_explicit():
    skills = [{"name":f"S{i}","proficiency":"expert","endorsements":5,"duration_months":0} for i in range(7)]
    c = make_candidate(skills=skills)
    s, comps = score_candidate(c)
    r = generate(c, 1, s, comps).lower()
    assert "honeypot" in r or "expert" in r or "impossible" in r

def test_different_candidates_different_reasoning():
    c1 = make_candidate(title="ML Engineer",  company="Swiggy")
    c2 = make_candidate(title="NLP Engineer", company="Ola", yoe=8.0)
    s1, comps1 = score_candidate(c1)
    s2, comps2 = score_candidate(c2)
    assert generate(c1, 1, s1, comps1) != generate(c2, 2, s2, comps2)


# ── semantic ──────────────────────────────────────────────────────────────────
def test_tfidf_mode_returns_scores():
    sc = SemanticScorer(use_neural=False)
    candidates = [make_candidate("ML Engineer"), make_candidate("HR Manager", summary="Managing HR policies")]
    sc.fit(candidates)
    scores = sc.score_all()
    assert scores.shape[0] == 2 and 0.0 <= float(scores[0]) <= 1.0

def test_ml_scores_higher_than_hr():
    sc   = SemanticScorer(use_neural=False)
    c_ml = make_candidate("ML Engineer",  summary="Shipped embedding-based retrieval to production. Vector search with FAISS.")
    c_hr = make_candidate("HR Manager",   summary="Managing HR operations, employee relations, recruitment pipeline.",
                          skills=[{"name":"HR Management","proficiency":"expert","endorsements":5,"duration_months":60}])
    sc.fit([c_ml, c_hr])
    scores = sc.score_all()
    assert float(scores[0]) > float(scores[1])

def test_phrase_hits_nonzero():
    sc = SemanticScorer(use_neural=False)
    c  = make_candidate(summary="I shipped embedding-based retrieval to production. Used NDCG and MRR. Migrated keyword to vector search.")
    assert sc.score_phrase_hits(c) > 0.0

def test_build_semantic_text():
    text = build_candidate_semantic_text(make_candidate())
    assert len(text) > 50 and ("python" in text.lower() or "ml" in text.lower())


# ── integration ───────────────────────────────────────────────────────────────
def test_end_to_end_pipeline():
    candidates = [
        make_candidate(title="ML Engineer",  yoe=6.5, company="Swiggy"),
        make_candidate(title="HR Manager",   yoe=8.0, company="Some Corp",
                       summary="HR operations and policy management.",
                       history=[{"company":"Some Corp","title":"HR Manager",
                                 "start_date":"2018-01-01","end_date":None,
                                 "duration_months":72,"is_current":True,
                                 "industry":"Human Resources","company_size":"1001-5000",
                                 "description":"Managed talent acquisition and HR operations."}]),
        make_candidate(title="NLP Engineer", yoe=5.0, company="Ola"),
    ]
    sc = SemanticScorer(use_neural=False)
    sc.fit(candidates)
    sem_map = sc.get_id_to_score_map()

    scored = []
    for c in candidates:
        cid = c["candidate_id"]
        sem = sem_map.get(cid, 0.5)
        phr = sc.score_phrase_hits(c)
        s, comps = score_candidate(c, semantic_score=sem, phrase_score=phr)
        scored.append((s, c, comps))

    scored.sort(key=lambda x: -x[0])
    assert "HR" not in scored[0][1]["profile"]["current_title"]
    assert all(0.0 <= s <= 1.0 for s, _, _ in scored)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
