#!/usr/bin/env python3
"""
OFFLINE PRE-COMPUTATION SCRIPT
================================
Run this ONCE before ranking, or whenever the Job Description / candidate pool changes.

What it does:
  1. Loads all candidates from candidates.jsonl (or .jsonl.gz)
  2. Parses the Job Description dynamically (via --jd option) to extract requirements
  3. Detects and flags honeypot candidates using structural timeline validations
  4. Engineers 40+ features per candidate relative to the dynamic JD constraints
  5. Generates sentence-transformer embeddings for candidate summaries and JD text
  6. Builds a FAISS index for fast similarity search
  7. Saves all artifacts (including JD metadata and index) to ./artifacts/ directory

Usage:
  python precompute.py --candidates ./candidates.jsonl --artifacts ./artifacts --jd ./job_description.docx

Output artifacts:
  artifacts/features.npz         - Engineered feature matrix (100k x ~45)
  artifacts/candidate_ids.json   - Ordered list of candidate IDs
  artifacts/honeypot_flags.json  - Set of honeypot candidate IDs
  artifacts/embeddings.npy       - Dense embeddings (100k x 384)
  artifacts/faiss_index.bin      - FAISS index for cosine similarity
  artifacts/meta.json            - JD embedding + config
"""

import argparse
import gzip
import json
import logging
import os
import re
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JD CONSTANTS & GLOBAL CRITERIA (Default fallback)
# ---------------------------------------------------------------------------
JD_TEXT = """
Senior AI Engineer at Redrob AI, Series A company.
5-9 years experience. Production experience with embeddings-based retrieval systems 
(sentence-transformers, BGE, E5, OpenAI embeddings). Vector databases hybrid search 
infrastructure (Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch, OpenSearch).
Strong Python. Ranking evaluation frameworks NDCG MRR MAP A/B testing.
LLM fine-tuning LoRA QLoRA PEFT. Learning-to-rank XGBoost LightGBM.
HR-tech recruiting marketplace products. RAG retrieval augmented generation.
NLP information retrieval recommendation systems.
Product company experience. Not pure research. Not consulting firms.
Pune Noida Hyderabad Mumbai Delhi NCR India. Hybrid work mode.
Notice period under 30 days preferred, up to 90 days acceptable.
"""

MUST_HAVE_SKILLS = {
    "embeddings", "sentence-transformers", "vector search", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch", "retrieval",
    "ranking", "nlp", "python", "rag", "information retrieval",
    "semantic search", "dense retrieval", "hybrid search",
    "recommendation systems", "learning to rank", "xgboost", "lightgbm",
    "transformers", "bert", "llm", "fine-tuning", "lora", "qlora",
}

NICE_SKILLS = {
    "pytorch", "tensorflow", "huggingface", "langchain", "openai",
    "distributed systems", "kafka", "redis", "kubernetes", "docker",
    "a/b testing", "ndcg", "mrr", "map", "evaluation", "mlflow",
    "weights & biases", "wandb", "airflow", "spark", "data pipelines",
}

STRONG_TITLE_KEYWORDS = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "nlp engineer", "search engineer", "ranking engineer",
    "applied scientist", "research engineer", "data scientist",
    "ai researcher", "retrieval engineer",
}

WEAK_TITLE_KEYWORDS = {
    "marketing", "sales", "finance", "hr", "recruiter", "product manager",
    "project manager", "scrum master", "business analyst", "consultant",
    "frontend", "android", "ios", "mobile", "devops", "sre",
}

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree",
}

PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "bengaluru", "bangalore",
    "delhi", "gurgaon", "gurugram", "ncr", "india",
}

TODAY = date(2026, 6, 9)
YOE_MIN = 5.0
YOE_MAX = 9.0
WORK_MODE = "hybrid"
NOTICE_PREFERRED = 30


# ---------------------------------------------------------------------------
# DOCX & DYNAMIC JD PARSING
# ---------------------------------------------------------------------------
def read_docx_from_zip(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as z:
            xml_content = z.read('word/document.xml')
            root = ET.fromstring(xml_content)
            texts = []
            for elem in root.iter():
                if elem.tag.endswith('t') and elem.text is not None:
                    texts.append(elem.text)
            return "".join(texts)
    except Exception as e:
        logger.error(f"Error reading docx {path}: {e}")
        return ""


def load_jd(jd_path: Optional[str]) -> str:
    if jd_path:
        path = Path(jd_path)
        if not path.exists():
            logger.warning(f"JD path {jd_path} not found. Falling back to default search in workspace.")
        else:
            if path.suffix == ".docx":
                return read_docx_from_zip(path)
            else:
                return path.read_text(encoding="utf-8", errors="ignore")
            
    # Try default filenames in workspace
    for default_fn in ("job_description.docx", "job_description.txt"):
        p = Path(default_fn)
        if p.exists():
            logger.info(f"Found default JD file: {default_fn}")
            if p.suffix == ".docx":
                return read_docx_from_zip(p)
            else:
                return p.read_text(encoding="utf-8", errors="ignore")
                
    logger.info("No external JD file specified or found. Using default hardcoded JD.")
    return JD_TEXT


def parse_job_description(jd_text: str, candidates: list) -> dict:
    jd_lower = jd_text.lower()
    
    # 1. Compile skill vocabulary from candidates
    vocab = {}
    for c in candidates:
        for s in c.get("skills", []) or []:
            name = s.get("name", "").strip()
            if name:
                name_lower = name.lower()
                vocab[name_lower] = vocab.get(name_lower, 0) + 1
    
    # Match candidate skills inside the JD text
    matched_skills = {}
    for sk, count in vocab.items():
        if len(sk) < 3 and sk not in ("ml", "ai", "go", "c", "r", "ip", "js"):
            continue
        if sk in ("and", "the", "for", "with", "software", "development", "engineer", "engineering", "years", "experience", "role", "team", "product", "company"):
            continue
        # Use word boundaries to match whole terms
        pattern = r'\b' + re.escape(sk) + r'\b'
        if re.search(pattern, jd_lower):
            matched_skills[sk] = count
            
    # Sort matched skills to find the most relevant ones
    must_have = set()
    nice_have = set()
    
    sentences = re.split(r'[.!?\n]', jd_lower)
    for sk in matched_skills:
        is_nice = False
        for sentence in sentences:
            if sk in sentence:
                if any(w in sentence for w in ("preferred", "nice to", "plus", "like to have", "desired", "optional", "bonus", "should have")):
                    is_nice = True
                    break
        if is_nice:
            nice_have.add(sk)
        else:
            must_have.add(sk)
            
    if not must_have:
        must_have = set(matched_skills.keys())
        
    # 2. Extract preferred locations
    loc_vocab = {}
    for c in candidates:
        profile = c.get("profile", {}) or {}
        loc = profile.get("location", "").strip()
        country = profile.get("country", "").strip()
        if loc:
            loc_vocab[loc.lower()] = loc_vocab.get(loc.lower(), 0) + 1
        if country:
            loc_vocab[country.lower()] = loc_vocab.get(country.lower(), 0) + 1
            
    preferred_locations = set()
    for loc in loc_vocab:
        if len(loc) < 3:
            continue
        if loc in ("and", "the", "new", "city", "state", "north", "south", "east", "west"):
            continue
        pattern = r'\b' + re.escape(loc) + r'\b'
        if re.search(pattern, jd_lower):
            preferred_locations.add(loc)
            
    if not preferred_locations:
        preferred_locations = {"pune", "noida", "hyderabad", "mumbai", "bengaluru", "delhi", "gurgaon", "india"}
        
    # 3. Extract target Years of Experience range
    yoe_min = 5.0
    yoe_max = 9.0
    # Search for patterns like "5-9 years" or "5 to 9 years"
    yoe_range_match = re.search(r'(\d+)\s*-\s*(\d+)\s*(?:years|yrs|yoe)', jd_lower)
    if not yoe_range_match:
        yoe_range_match = re.search(r'(\d+)\s*to\s*(\d+)\s*(?:years|yrs|yoe)', jd_lower)
    if yoe_range_match:
        yoe_min = float(yoe_range_match.group(1))
        yoe_max = float(yoe_range_match.group(2))
    else:
        # Search for "X+ years"
        yoe_plus_match = re.search(r'(\d+)\+?\s*(?:years|yrs|yoe)', jd_lower)
        if yoe_plus_match:
            yoe_min = float(yoe_plus_match.group(1))
            yoe_max = yoe_min + 5.0
        else:
            yoe_atleast_match = re.search(r'(?:at least|minimum of)\s*(\d+)\s*(?:years|yrs|yoe)', jd_lower)
            if yoe_atleast_match:
                yoe_min = float(yoe_atleast_match.group(1))
                yoe_max = yoe_min + 5.0

    # 4. Extract strong title keywords from first line
    strong_title_keywords = set()
    first_line = jd_text.splitlines()[0].lower() if jd_text.splitlines() else ""
    title_words = re.findall(r'[a-zA-Z]+', first_line)
    for w in title_words:
        if len(w) > 3 and w not in ("senior", "junior", "lead", "staff", "principal", "manager", "director", "role", "position", "recruiting", "talent"):
            strong_title_keywords.add(w)
            
    # Add strong title matches based on candidates
    title_vocab = {}
    for c in candidates:
        profile = c.get("profile", {}) or {}
        title = profile.get("current_title", "").strip()
        if title:
            title_vocab[title.lower()] = title_vocab.get(title.lower(), 0) + 1
            
    for title in title_vocab:
        if title in jd_lower and len(title) > 5:
            for word in title.split():
                if len(word) > 2 and word not in ("and", "the", "for", "with", "senior", "lead", "junior", "staff", "principal"):
                    strong_title_keywords.add(word)

    if not strong_title_keywords:
        strong_title_keywords = {"ai", "ml", "engineer", "machine", "learning", "data", "scientist", "nlp", "search"}

    # 5. Extract preferred work mode
    work_mode = "hybrid"
    for mode in ("hybrid", "remote", "onsite", "flexible"):
        if mode in jd_lower:
            work_mode = mode
            break
            
    # 6. Extract notice period preference
    notice_preferred = 30
    notice_match = re.search(r'(?:notice|availability)[^\d]*(\d+)\s*days?', jd_lower)
    if notice_match:
        notice_preferred = int(notice_match.group(1))
        
    return {
        "must_have_skills": list(must_have) if must_have else list(MUST_HAVE_SKILLS),
        "nice_skills": list(nice_have) if nice_have else list(NICE_SKILLS),
        "preferred_locations": list(preferred_locations),
        "strong_title_keywords": list(strong_title_keywords),
        "yoe_min": yoe_min,
        "yoe_max": yoe_max,
        "work_mode": work_mode,
        "notice_preferred": notice_preferred,
        "consulting_firms": list(CONSULTING_FIRMS)
    }


# ---------------------------------------------------------------------------
# HONEYPOT DETECTION
# ---------------------------------------------------------------------------
def detect_honeypot(candidate: dict) -> tuple[bool, list[str]]:
    """
    Returns (is_honeypot, reasons).
    Checks for logically impossible profile combinations.
    """
    reasons = []
    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []

    yoe = profile.get("years_of_experience") or 0.0
    try:
        yoe = float(yoe)
    except (TypeError, ValueError):
        yoe = 0.0
    yoe_months = yoe * 12

    # Check 1: skill duration_months > total years_of_experience
    # (can't have used a skill for longer than you've been working)
    impossible_skill_count = 0
    for sk in skills:
        if sk and isinstance(sk, dict):
            dur = sk.get("duration_months") or 0
            if dur > yoe_months + 12:  # +12 months grace
                impossible_skill_count += 1
    if impossible_skill_count >= 3:
        reasons.append(f"{impossible_skill_count} skills with duration > total YoE")

    # Check 2: career history duration inconsistency
    # Sum of all role durations significantly exceeds stated YoE
    total_career_months = sum((j.get("duration_months") or 0) for j in career if j and isinstance(j, dict))
    if total_career_months > yoe_months + 24 and total_career_months > 0:
        reasons.append(
            f"Career months {total_career_months} >> YoE months {yoe_months:.0f}"
        )

    # Check 3: Expert proficiency in 10+ skills with 0 endorsements each
    expert_zero_endorse = sum(
        1 for sk in skills
        if sk and isinstance(sk, dict) and sk.get("proficiency") == "expert" and (sk.get("endorsements") or 0) == 0
    )
    if expert_zero_endorse >= 8:
        reasons.append(f"{expert_zero_endorse} 'expert' skills with 0 endorsements")

    # Check 4: Start date at a company before plausible founding
    # (e.g., 8 years at a 3-year-old company)
    for job in career:
        if job and isinstance(job, dict):
            start_str = job.get("start_date") or ""
            dur = job.get("duration_months") or 0
            if start_str and dur > 0:
                try:
                    start = date.fromisoformat(start_str)
                    company = (job.get("company") or "").lower()
                    if dur > 120:  # 10+ years at one place is rare
                        reasons.append(
                            f"Suspicious: {dur}m tenure at {job.get('company')}"
                        )
                except ValueError:
                    pass

    # Check 5: YoE = 0 or 1 but has "expert" in core ML skills
    if yoe < 2:
        expert_ml = sum(
            1 for sk in skills
            if sk and isinstance(sk, dict) and sk.get("proficiency") in ("expert", "advanced")
            and any(kw in (sk.get("name") or "").lower()
                    for kw in ("ml", "ai", "nlp", "deep learning", "llm"))
        )
        if expert_ml >= 3:
            reasons.append(f"YoE {yoe} but {expert_ml} expert ML skills")

    # Threshold: 1+ definitive reasons = honeypot
    return len(reasons) > 0, reasons


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_features(candidate: dict) -> dict:
    """
    Returns a flat dict of 45 features for a single candidate.
    All features are floats in a normalized-ish range.
    """
    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    education = candidate.get("education", []) or []
    sigs = candidate.get("redrob_signals", {}) or {}
    certs = candidate.get("certifications", []) or []

    yoe = float(profile.get("years_of_experience") or 0)

    # --- SKILL FEATURES ---
    skill_names_lower = {sk.get("name", "").lower() for sk in skills if sk and sk.get("name")}
    
    must_have_hits = sum(
        1 for kw in MUST_HAVE_SKILLS
        if any(kw in sn for sn in skill_names_lower)
    )
    nice_hits = sum(
        1 for kw in NICE_SKILLS
        if any(kw in sn for sn in skill_names_lower)
    )

    # Skill trust score: endorsements × proficiency weight × duration
    proficiency_weight = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}
    skill_trust = 0.0
    for sk in skills:
        if not sk:
            continue
        name_lower = sk.get("name", "").lower()
        is_relevant = any(kw in name_lower for kw in MUST_HAVE_SKILLS | NICE_SKILLS)
        if is_relevant:
            pw = proficiency_weight.get(sk.get("proficiency", "beginner"), 0.25)
            endorse = min(sk.get("endorsements") or 0, 100) / 100.0
            dur = min(sk.get("duration_months") or 0, 60) / 60.0
            skill_trust += pw * (0.4 + 0.3 * endorse + 0.3 * dur)

    # Core ML skill depth (dynamic based on must-have skills keywords)
    core_ml_depth = 0.0
    core_keywords = [w.lower() for s in MUST_HAVE_SKILLS for w in s.split() if len(w) > 3]
    if not core_keywords:
        core_keywords = ["embed", "retriev", "rank", "vector", "faiss", "nlp", "transformer", "llm", "rag", "semantic"]
    for sk in skills:
        if not sk:
            continue
        name_lower = sk.get("name", "").lower()
        if any(ck in name_lower for ck in core_keywords):
            pw = proficiency_weight.get(sk.get("proficiency", "beginner"), 0.25)
            dur = min(sk.get("duration_months") or 0, 60) / 60.0
            core_ml_depth += pw * (0.5 + 0.5 * dur)

    # --- TITLE / CAREER FEATURES ---
    current_title = (profile.get("current_title") or "").lower()
    title_score = 0.0
    for kw in STRONG_TITLE_KEYWORDS:
        if kw in current_title:
            title_score = 1.0
            break
    for kw in WEAK_TITLE_KEYWORDS:
        if kw in current_title:
            title_score = max(0.0, title_score - 0.5)

    # Product company vs consulting
    consulting_penalty = 0.0
    all_companies = [
        ((j.get("company") or "").lower(), j.get("duration_months") or 0)
        for j in career if j
    ]
    consulting_months = sum(
        dur for company, dur in all_companies
        if any(cf in company for cf in CONSULTING_FIRMS)
    )
    total_career_months = sum(d for _, d in all_companies) or 1
    consulting_ratio = consulting_months / total_career_months
    if consulting_ratio > 0.8:
        consulting_penalty = 1.0  # Career entirely in consulting
    elif consulting_ratio > 0.5:
        consulting_penalty = 0.5

    # Current company consulting penalty
    current_company_lower = (profile.get("current_company") or "").lower()
    current_is_consulting = float(
        any(cf in current_company_lower for cf in CONSULTING_FIRMS)
    )

    # YoE score: ideal is YOE_MIN to YOE_MAX
    if YOE_MIN <= yoe <= YOE_MAX:
        yoe_score = 1.0
    elif (YOE_MIN - 1.0) <= yoe < YOE_MIN or YOE_MAX < yoe <= (YOE_MAX + 2.0):
        yoe_score = 0.8
    elif (YOE_MIN - 2.0) <= yoe < (YOE_MIN - 1.0) or (YOE_MAX + 2.0) < yoe <= (YOE_MAX + 4.0):
        yoe_score = 0.6
    elif yoe < (YOE_MIN - 2.0):
        yoe_score = 0.3
    else:
        yoe_score = 0.5  # overqualified risk

    # Career trajectory: progression in roles aligned with title/skills
    ml_role_count = 0
    title_keywords = [kw.lower() for kw in STRONG_TITLE_KEYWORDS]
    desc_keywords = [kw.lower() for kw in MUST_HAVE_SKILLS]
    for job in career:
        if not job:
            continue
        job_title = (job.get("title") or "").lower()
        job_desc = (job.get("description") or "").lower()
        if any(kw in job_title for kw in title_keywords):
            ml_role_count += 1
        elif any(kw in job_desc for kw in desc_keywords):
            ml_role_count += 0.5
    ml_career_ratio = min(ml_role_count / max(len(career), 1), 1.0)

    # Longest tenure (shows commitment, not job-hopper)
    max_tenure = max((j.get("duration_months") or 0 for j in career if j), default=0)
    tenure_score = min(max_tenure / 36.0, 1.0)  # 3 years = ideal

    # --- LOCATION FEATURES ---
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    location_score = 0.0
    if any(loc in location or loc in country for loc in PREFERRED_LOCATIONS):
        location_score = 1.0
    elif country == "india" and "india" in PREFERRED_LOCATIONS:
        location_score = 0.5
    willing_to_relocate = float(sigs.get("willing_to_relocate") or False)

    # --- EDUCATION FEATURES ---
    edu_tier_score = 0.0
    tier_map = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.25, "unknown": 0.3}
    if education:
        best_tier = max(
            tier_map.get(e.get("tier", "unknown"), 0.3) for e in education if e
        )
        edu_tier_score = best_tier
    has_relevant_degree = float(
        any(
            any(kw in e.get("field_of_study", "").lower()
                for kw in ["computer", "machine learn", "data", "information", "engineer"])
            for e in education if e and e.get("field_of_study")
        )
    )

    # --- BEHAVIORAL SIGNALS ---
    last_active_str = sigs.get("last_active_date") or ""
    days_inactive = 365  # default to very inactive
    if last_active_str:
        try:
            last_active = date.fromisoformat(last_active_str)
            days_inactive = (TODAY - last_active).days
        except ValueError:
            pass
    # Recency score: 0-30d = 1.0, 30-90d = 0.7, 90-180d = 0.4, 180+ = 0.1
    if days_inactive <= 30:
        recency_score = 1.0
    elif days_inactive <= 90:
        recency_score = 0.7
    elif days_inactive <= 180:
        recency_score = 0.4
    else:
        recency_score = 0.1

    open_to_work = float(sigs.get("open_to_work_flag") or False)
    recruiter_response = float(sigs.get("recruiter_response_rate") or 0.0)
    avg_response_hrs = sigs.get("avg_response_time_hours") or 999
    response_speed = max(0.0, 1.0 - min(avg_response_hrs / 72.0, 1.0))
    profile_completeness = (sigs.get("profile_completeness_score") or 0) / 100.0
    github_score = sigs.get("github_activity_score") or -1
    github_norm = 0.0 if github_score < 0 else github_score / 100.0
    interview_completion = float(sigs.get("interview_completion_rate") or 0.0)
    offer_acceptance = sigs.get("offer_acceptance_rate") or -1
    offer_acc_norm = 0.5 if offer_acceptance < 0 else float(offer_acceptance)
    profile_views = min(sigs.get("profile_views_received_30d") or 0, 200) / 200.0
    saved_by = min(sigs.get("saved_by_recruiters_30d") or 0, 20) / 20.0
    apps_30d = min(sigs.get("applications_submitted_30d") or 0, 10) / 10.0
    connection_count = min(sigs.get("connection_count") or 0, 1000) / 1000.0
    endorsements_recv = min(sigs.get("endorsements_received") or 0, 200) / 200.0
    verified = float(sigs.get("verified_email") or False) * 0.5 + \
               float(sigs.get("verified_phone") or False) * 0.3 + \
               float(sigs.get("linkedin_connected") or False) * 0.2

    # Notice period based on notice preference
    notice = sigs.get("notice_period_days") or 60
    if notice <= NOTICE_PREFERRED:
        notice_score = 1.0
    elif notice <= NOTICE_PREFERRED + 30:
        notice_score = 0.7
    elif notice <= NOTICE_PREFERRED + 60:
        notice_score = 0.4
    else:
        notice_score = 0.1

    # Skill assessment scores from Redrob platform
    assessment_scores = sigs.get("skill_assessment_scores") or {}
    relevant_assessments = [
        v for k, v in assessment_scores.items()
        if any(kw in k.lower() for kw in MUST_HAVE_SKILLS)
    ]
    avg_assessment = np.mean(relevant_assessments) / 100.0 if relevant_assessments else 0.3

    # Preferred work mode alignment
    candidate_work_mode = sigs.get("preferred_work_mode") or ""
    if WORK_MODE == "hybrid":
        work_mode_score = {"hybrid": 1.0, "flexible": 0.9, "onsite": 0.7, "remote": 0.5}.get(candidate_work_mode, 0.6)
    elif WORK_MODE == "remote":
        work_mode_score = {"remote": 1.0, "flexible": 0.9, "hybrid": 0.7, "onsite": 0.3}.get(candidate_work_mode, 0.6)
    elif WORK_MODE == "onsite":
        work_mode_score = {"onsite": 1.0, "hybrid": 0.7, "flexible": 0.6, "remote": 0.2}.get(candidate_work_mode, 0.5)
    else:
        work_mode_score = {"flexible": 1.0, "hybrid": 0.9, "remote": 0.8, "onsite": 0.7}.get(candidate_work_mode, 0.8)

    # Salary range (flag outliers)
    sal = sigs.get("expected_salary_range_inr_lpa") or {}
    sal_mid = ((sal.get("min") or 0) + (sal.get("max") or 0)) / 2.0
    # Dynamic or general senior range
    if 35 <= sal_mid <= 90:
        salary_fit = 1.0
    elif 25 <= sal_mid < 35 or 90 < sal_mid <= 120:
        salary_fit = 0.7
    elif sal_mid > 120:
        salary_fit = 0.4
    else:
        salary_fit = 0.5

    # Certifications
    relevant_cert_count = sum(
        1 for c in certs if c
        if any(kw in c.get("name", "").lower()
               for kw in ["ml", "ai", "nlp", "aws", "gcp", "azure", "deep", "data"])
    )
    cert_score = min(relevant_cert_count / 3.0, 1.0)

    # --- COMPOSITE BEHAVIORAL MULTIPLIER ---
    behavioral_multiplier = (
        0.25 * recency_score +
        0.20 * recruiter_response +
        0.15 * open_to_work +
        0.10 * interview_completion +
        0.10 * response_speed +
        0.10 * profile_completeness +
        0.10 * offer_acc_norm
    )
    behavioral_multiplier = 0.3 + behavioral_multiplier * 0.9

    n_must_haves = float(len(MUST_HAVE_SKILLS) or 1)

    return {
        # Skill features
        "must_have_hits": float(must_have_hits),
        "must_have_ratio": min(must_have_hits / n_must_haves, 1.0),
        "nice_hits": float(nice_hits),
        "skill_trust": min(skill_trust / 5.0, 1.0),
        "core_ml_depth": min(core_ml_depth / 4.0, 1.0),
        "total_skills": min(len(skills) / 20.0, 1.0),
        # Title/career features
        "title_score": title_score,
        "consulting_penalty": consulting_penalty,
        "current_is_consulting": current_is_consulting,
        "yoe_score": yoe_score,
        "yoe_raw": min(yoe / 15.0, 1.0),
        "ml_career_ratio": ml_career_ratio,
        "tenure_score": tenure_score,
        "career_length": min(len(career) / 5.0, 1.0),
        # Location features
        "location_score": location_score,
        "willing_to_relocate": willing_to_relocate,
        # Education features
        "edu_tier_score": edu_tier_score,
        "has_relevant_degree": has_relevant_degree,
        # Behavioral signals
        "recency_score": recency_score,
        "days_inactive_norm": min(days_inactive / 365.0, 1.0),
        "open_to_work": open_to_work,
        "recruiter_response": recruiter_response,
        "response_speed": response_speed,
        "profile_completeness": profile_completeness,
        "github_score": github_norm,
        "interview_completion": interview_completion,
        "offer_acceptance": offer_acc_norm,
        "profile_views": profile_views,
        "saved_by_recruiters": saved_by,
        "apps_30d": apps_30d,
        "connection_count": connection_count,
        "endorsements_recv": endorsements_recv,
        "verified": verified,
        "notice_score": notice_score,
        "avg_assessment": avg_assessment,
        "work_mode_score": work_mode_score,
        "salary_fit": salary_fit,
        "cert_score": cert_score,
        "behavioral_multiplier": behavioral_multiplier,
        # Raw values for reasoning generation
        "_yoe": yoe,
        "_days_inactive": float(days_inactive),
        "_notice_period": float(notice),
        "_must_have_hits": float(must_have_hits),
        "_recruiter_response": recruiter_response,
        "_open_to_work": float(sigs.get("open_to_work_flag") or False),
    }


# ---------------------------------------------------------------------------
# TEXT PREPARATION FOR EMBEDDINGS
# ---------------------------------------------------------------------------
def build_candidate_text(candidate: dict) -> str:
    """
    Builds a rich text representation of a candidate for embedding.
    Emphasizes signals that matter for this JD.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # Title + summary
    text_parts = [
        profile.get("current_title", ""),
        profile.get("headline", ""),
        profile.get("summary", ""),
    ]

    # Top skills (by endorsements + duration)
    sorted_skills = sorted(
        skills,
        key=lambda s: s.get("endorsements", 0) + s.get("duration_months", 0),
        reverse=True
    )
    skill_text = ", ".join(
        f"{s['name']} ({s.get('proficiency', '')})"
        for s in sorted_skills[:15]
    )
    text_parts.append(f"Skills: {skill_text}")

    # Career descriptions (last 3 roles)
    for job in career[:3]:
        text_parts.append(
            f"{job.get('title', '')} at {job.get('company', '')}: "
            f"{job.get('description', '')[:300]}"
        )

    return " ".join(filter(None, text_parts))[:2048]  # Cap at 2048 chars


# ---------------------------------------------------------------------------
# MAIN PRECOMPUTE
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Offline pre-computation for Redrob ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--artifacts", default="./artifacts", help="Output directory for artifacts")
    parser.add_argument("--jd", default=None, help="Path to job description text/docx file")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip embedding generation (faster testing)")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # --- LOAD CANDIDATES ---
    logger.info("Loading candidates...")
    t0 = time.time()
    candidates_path = Path(args.candidates)
    opener = gzip.open if candidates_path.suffix == ".gz" else open
    candidates = []
    with opener(candidates_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    logger.info(f"  Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    global MUST_HAVE_SKILLS, NICE_SKILLS, STRONG_TITLE_KEYWORDS, WEAK_TITLE_KEYWORDS, CONSULTING_FIRMS, PREFERRED_LOCATIONS, TODAY, YOE_MIN, YOE_MAX, WORK_MODE, NOTICE_PREFERRED, JD_TEXT
    
    # --- LOAD AND PARSE JD ---
    JD_TEXT = load_jd(args.jd)
    logger.info(f"JD Loaded. Length: {len(JD_TEXT)} characters.")
    
    # --- DYNAMIC REFERENCE DATE (TODAY) ---
    latest_date = date(2026, 6, 9)
    for c in candidates:
        sigs = c.get("redrob_signals", {}) or {}
        for date_field in ("last_active_date", "signup_date"):
            d_str = sigs.get(date_field)
            if d_str:
                try:
                    d = date.fromisoformat(d_str)
                    if d > latest_date:
                        latest_date = d
                except ValueError:
                    pass
    TODAY = latest_date
    logger.info(f"Reference date (TODAY) set dynamically to: {TODAY.isoformat()}")

    # --- DYNAMIC JD CRITERIA EXTRACTION ---
    logger.info("Parsing Job Description dynamically...")
    jd_meta = parse_job_description(JD_TEXT, candidates)
    
    MUST_HAVE_SKILLS = set(jd_meta["must_have_skills"])
    NICE_SKILLS = set(jd_meta["nice_skills"])
    PREFERRED_LOCATIONS = set(jd_meta["preferred_locations"])
    STRONG_TITLE_KEYWORDS = set(jd_meta["strong_title_keywords"])
    YOE_MIN = jd_meta["yoe_min"]
    YOE_MAX = jd_meta["yoe_max"]
    WORK_MODE = jd_meta["work_mode"]
    NOTICE_PREFERRED = jd_meta["notice_preferred"]
    
    logger.info("--- Extracted JD Criteria ---")
    logger.info(f"  YoE Range: {YOE_MIN} - {YOE_MAX} years")
    logger.info(f"  Work Mode: {WORK_MODE}")
    logger.info(f"  Notice Period Preferred: {NOTICE_PREFERRED} days")
    logger.info(f"  Must-Have Skills ({len(MUST_HAVE_SKILLS)}): {list(MUST_HAVE_SKILLS)[:10]}...")
    logger.info(f"  Preferred Locations ({len(PREFERRED_LOCATIONS)}): {list(PREFERRED_LOCATIONS)[:5]}...")
    logger.info(f"  Strong Title Keywords: {list(STRONG_TITLE_KEYWORDS)}")
    logger.info("-----------------------------")

    # --- HONEYPOT DETECTION ---
    logger.info("Detecting honeypots...")
    honeypot_ids = set()
    honeypot_reasons = {}
    for c in candidates:
        is_hp, reasons = detect_honeypot(c)
        if is_hp:
            cid = c["candidate_id"]
            honeypot_ids.add(cid)
            honeypot_reasons[cid] = reasons

    logger.info(f"  Flagged {len(honeypot_ids)} honeypots")
    with open(artifacts_dir / "honeypot_flags.json", "w") as f:
        json.dump({"honeypot_ids": list(honeypot_ids), "reasons": honeypot_reasons}, f)

    # --- FEATURE ENGINEERING ---
    logger.info("Engineering features...")
    t1 = time.time()
    all_features = []
    candidate_ids = []
    feature_names = None

    for c in candidates:
        feat_dict = engineer_features(c)
        if feature_names is None:
            feature_names = list(feat_dict.keys())
        all_features.append([feat_dict[k] for k in feature_names])
        candidate_ids.append(c["candidate_id"])

    feature_matrix = np.array(all_features, dtype=np.float32)
    logger.info(f"  Feature matrix shape: {feature_matrix.shape} in {time.time()-t1:.1f}s")

    np.savez_compressed(
        artifacts_dir / "features.npz",
        features=feature_matrix,
        feature_names=np.array(feature_names)
    )
    with open(artifacts_dir / "candidate_ids.json", "w") as f:
        json.dump(candidate_ids, f)

    # --- EMBEDDINGS + FAISS ---
    jd_meta_dict = {
        "must_have_skills": list(MUST_HAVE_SKILLS),
        "nice_skills": list(NICE_SKILLS),
        "strong_title_keywords": list(STRONG_TITLE_KEYWORDS),
        "weak_title_keywords": list(WEAK_TITLE_KEYWORDS),
        "consulting_firms": list(CONSULTING_FIRMS),
        "preferred_locations": list(PREFERRED_LOCATIONS),
        "yoe_min": YOE_MIN,
        "yoe_max": YOE_MAX,
        "work_mode": WORK_MODE,
        "notice_preferred": NOTICE_PREFERRED,
        "today": TODAY.isoformat(),
        "jd_text": JD_TEXT
    }

    if not args.no_embeddings:
        logger.info("Generating embeddings (this is the slow part ~20-40min)...")
        try:
            from sentence_transformers import SentenceTransformer
            import faiss

            model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, fast, ~80MB

            # Build candidate texts
            logger.info("  Building candidate texts...")
            texts = [build_candidate_text(c) for c in candidates]

            # JD embedding
            jd_embedding = model.encode([JD_TEXT], normalize_embeddings=True)[0]

            # Batch encode (batch_size=256 to stay in RAM)
            logger.info("  Encoding candidates in batches...")
            t2 = time.time()
            embeddings = model.encode(
                texts,
                batch_size=256,
                show_progress_bar=True,
                normalize_embeddings=True,  # L2 normalize for cosine similarity
                convert_to_numpy=True
            )
            logger.info(f"  Encoded {len(embeddings):,} candidates in {time.time()-t2:.1f}s")

            # Save embeddings
            np.save(artifacts_dir / "embeddings.npy", embeddings.astype(np.float32))

            # Build FAISS index (Inner Product = cosine after L2 normalize)
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings.astype(np.float32))
            faiss.write_index(index, str(artifacts_dir / "faiss_index.bin"))
            logger.info(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")

            # Save meta
            meta = {
                "jd_embedding": jd_embedding.tolist(),
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dim": dim,
                "n_candidates": len(candidates),
                "feature_names": feature_names,
                "jd_meta": jd_meta_dict
            }

        except ImportError as e:
            logger.warning(f"  WARNING: Could not generate embeddings: {e}")
            logger.info("  Install with: pip install sentence-transformers faiss-cpu")
            meta = {
                "jd_embedding": None,
                "embedding_model": None,
                "embedding_dim": None,
                "n_candidates": len(candidates),
                "feature_names": feature_names,
                "jd_meta": jd_meta_dict
            }
    else:
        logger.info("Skipping embeddings (--no-embeddings flag set)")
        meta = {
            "jd_embedding": None,
            "embedding_model": None,
            "n_candidates": len(candidates),
            "feature_names": feature_names,
            "jd_meta": jd_meta_dict
        }

    with open(artifacts_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    logger.info(f"\nPre-computation complete. Artifacts saved to {artifacts_dir}/")
    logger.info(f"Total time: {time.time()-t0:.1f}s")
    logger.info("Artifact sizes:")
    for p in sorted(artifacts_dir.iterdir()):
        logger.info(f"  {p.name}: {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
