#!/usr/bin/env python3
"""
OFFLINE PRE-COMPUTATION SCRIPT
================================
Run this ONCE before ranking. No time limit applies here.

What it does:
  1. Loads all 100k candidates from candidates.jsonl (or .jsonl.gz)
  2. Detects and flags honeypot candidates
  3. Engineers 40+ features per candidate
  4. Generates sentence-transformer embeddings for candidate summaries
  5. Builds a FAISS index for fast similarity search
  6. Saves all artifacts to ./artifacts/ directory

Usage:
  python offline/precompute.py --candidates ./candidates.jsonl --artifacts ./artifacts

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
# JD CONSTANTS — extracted from the Senior AI Engineer JD
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

# Must-have skills from JD (hard requirements)
MUST_HAVE_SKILLS = {
    "embeddings", "sentence-transformers", "vector search", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch", "retrieval",
    "ranking", "nlp", "python", "rag", "information retrieval",
    "semantic search", "dense retrieval", "hybrid search",
    "recommendation systems", "learning to rank", "xgboost", "lightgbm",
    "transformers", "bert", "llm", "fine-tuning", "lora", "qlora",
}

# Nice-to-have skills
NICE_SKILLS = {
    "pytorch", "tensorflow", "huggingface", "langchain", "openai",
    "distributed systems", "kafka", "redis", "kubernetes", "docker",
    "a/b testing", "ndcg", "mrr", "map", "evaluation", "mlflow",
    "weights & biases", "wandb", "airflow", "spark", "data pipelines",
}

# Title signals that indicate strong fit
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

# Consulting firms — explicit JD disqualifier
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree",  # mindtree is now LTI Mindtree (Wipro adjacent)
}

# Preferred locations for this JD
PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "bengaluru", "bangalore",
    "delhi", "gurgaon", "gurugram", "ncr", "india",
}

# Reference date for recency calculations
TODAY = date(2026, 6, 9)


# ---------------------------------------------------------------------------
# HONEYPOT DETECTION
# ---------------------------------------------------------------------------
def detect_honeypot(candidate: dict) -> tuple[bool, list[str]]:
    """
    Returns (is_honeypot, reasons).
    Checks for logically impossible profile combinations.
    """
    reasons = []
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    yoe = profile.get("years_of_experience", 0)
    yoe_months = yoe * 12

    # Check 1: skill duration_months > total years_of_experience
    # (can't have used a skill for longer than you've been working)
    impossible_skill_count = 0
    for sk in skills:
        dur = sk.get("duration_months", 0)
        if dur > yoe_months + 12:  # +12 months grace
            impossible_skill_count += 1
    if impossible_skill_count >= 3:
        reasons.append(f"{impossible_skill_count} skills with duration > total YoE")

    # Check 2: career history duration inconsistency
    # Sum of all role durations significantly exceeds stated YoE
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    if total_career_months > yoe_months + 24 and total_career_months > 0:
        reasons.append(
            f"Career months {total_career_months} >> YoE months {yoe_months:.0f}"
        )

    # Check 3: Expert proficiency in 10+ skills with 0 endorsements each
    expert_zero_endorse = sum(
        1 for sk in skills
        if sk.get("proficiency") == "expert" and sk.get("endorsements", 0) == 0
    )
    if expert_zero_endorse >= 8:
        reasons.append(f"{expert_zero_endorse} 'expert' skills with 0 endorsements")

    # Check 4: Start date at a company before plausible founding
    # (e.g., 8 years at a 3-year-old company)
    for job in career:
        start_str = job.get("start_date", "")
        dur = job.get("duration_months", 0)
        if start_str and dur > 0:
            try:
                start = date.fromisoformat(start_str)
                # If job started before 2010 but company is clearly modern
                company = job.get("company", "").lower()
                # Heuristic: duration claimed > actual possible tenure
                # (company founded after start date)
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
            if sk.get("proficiency") in ("expert", "advanced")
            and any(kw in sk.get("name", "").lower()
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
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    sigs = candidate.get("redrob_signals", {})
    certs = candidate.get("certifications", [])

    yoe = float(profile.get("years_of_experience", 0))

    # --- SKILL FEATURES ---
    skill_names_lower = {sk.get("name", "").lower() for sk in skills}
    skill_map = {sk.get("name", "").lower(): sk for sk in skills}

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
        name_lower = sk.get("name", "").lower()
        is_relevant = any(kw in name_lower for kw in MUST_HAVE_SKILLS | NICE_SKILLS)
        if is_relevant:
            pw = proficiency_weight.get(sk.get("proficiency", "beginner"), 0.25)
            endorse = min(sk.get("endorsements", 0), 100) / 100.0
            dur = min(sk.get("duration_months", 0), 60) / 60.0
            skill_trust += pw * (0.4 + 0.3 * endorse + 0.3 * dur)

    # Core ML skill depth (embeddings + retrieval specifically)
    core_ml_depth = 0.0
    core_keywords = [
        "embed", "retriev", "rank", "vector", "faiss", "nlp",
        "transformer", "llm", "rag", "semantic"
    ]
    for sk in skills:
        name_lower = sk.get("name", "").lower()
        if any(ck in name_lower for ck in core_keywords):
            pw = proficiency_weight.get(sk.get("proficiency", "beginner"), 0.25)
            dur = min(sk.get("duration_months", 0), 60) / 60.0
            core_ml_depth += pw * (0.5 + 0.5 * dur)

    # --- TITLE / CAREER FEATURES ---
    current_title = profile.get("current_title", "").lower()
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
        (j.get("company", "").lower(), j.get("duration_months", 0))
        for j in career
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
    current_company_lower = profile.get("current_company", "").lower()
    current_is_consulting = float(
        any(cf in current_company_lower for cf in CONSULTING_FIRMS)
    )

    # YoE score: ideal is 5-9 years, penalize < 3 or > 12
    if 5 <= yoe <= 9:
        yoe_score = 1.0
    elif 4 <= yoe < 5 or 9 < yoe <= 11:
        yoe_score = 0.8
    elif 3 <= yoe < 4 or 11 < yoe <= 13:
        yoe_score = 0.6
    elif yoe < 3:
        yoe_score = 0.3
    else:
        yoe_score = 0.5  # 13+ years, overqualified risk

    # Career trajectory: are they progressing in ML/AI roles?
    ml_role_count = 0
    for job in career:
        job_title = job.get("title", "").lower()
        job_desc = job.get("description", "").lower()
        if any(kw in job_title for kw in ["ml", "ai", "nlp", "data scien", "research"]):
            ml_role_count += 1
        elif any(kw in job_desc for kw in ["embedding", "retrieval", "ranking", "llm", "vector"]):
            ml_role_count += 0.5
    ml_career_ratio = min(ml_role_count / max(len(career), 1), 1.0)

    # Longest tenure (shows commitment, not job-hopper)
    max_tenure = max((j.get("duration_months", 0) for j in career), default=0)
    tenure_score = min(max_tenure / 36.0, 1.0)  # 3 years = ideal

    # --- LOCATION FEATURES ---
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    location_score = 0.0
    if country == "india":
        location_score = 0.5
        if any(loc in location for loc in PREFERRED_LOCATIONS):
            location_score = 1.0
    willing_to_relocate = float(sigs.get("willing_to_relocate", False))

    # --- EDUCATION FEATURES ---
    edu_tier_score = 0.0
    tier_map = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.25, "unknown": 0.3}
    if education:
        best_tier = max(
            tier_map.get(e.get("tier", "unknown"), 0.3) for e in education
        )
        edu_tier_score = best_tier
    has_relevant_degree = float(
        any(
            any(kw in e.get("field_of_study", "").lower()
                for kw in ["computer", "machine learn", "data", "information", "engineer"])
            for e in education
        )
    )

    # --- BEHAVIORAL SIGNALS ---
    # Recency: days since last active
    last_active_str = sigs.get("last_active_date", "")
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

    open_to_work = float(sigs.get("open_to_work_flag", False))
    recruiter_response = float(sigs.get("recruiter_response_rate", 0.0))
    avg_response_hrs = sigs.get("avg_response_time_hours", 999)
    response_speed = max(0.0, 1.0 - min(avg_response_hrs / 72.0, 1.0))
    profile_completeness = sigs.get("profile_completeness_score", 0) / 100.0
    github_score = sigs.get("github_activity_score", -1)
    github_norm = 0.0 if github_score < 0 else github_score / 100.0
    interview_completion = float(sigs.get("interview_completion_rate", 0.0))
    offer_acceptance = sigs.get("offer_acceptance_rate", -1)
    offer_acc_norm = 0.5 if offer_acceptance < 0 else float(offer_acceptance)
    profile_views = min(sigs.get("profile_views_received_30d", 0), 200) / 200.0
    saved_by = min(sigs.get("saved_by_recruiters_30d", 0), 20) / 20.0
    apps_30d = min(sigs.get("applications_submitted_30d", 0), 10) / 10.0
    connection_count = min(sigs.get("connection_count", 0), 1000) / 1000.0
    endorsements_recv = min(sigs.get("endorsements_received", 0), 200) / 200.0
    verified = float(sigs.get("verified_email", False)) * 0.5 + \
               float(sigs.get("verified_phone", False)) * 0.3 + \
               float(sigs.get("linkedin_connected", False)) * 0.2

    # Notice period: <30d ideal, 30-60 ok, 60-90 marginal, 90+ bad
    notice = sigs.get("notice_period_days", 60)
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.7
    elif notice <= 90:
        notice_score = 0.4
    else:
        notice_score = 0.1

    # Skill assessment scores from Redrob platform
    assessment_scores = sigs.get("skill_assessment_scores", {})
    relevant_assessments = [
        v for k, v in assessment_scores.items()
        if any(kw in k.lower() for kw in
               ["nlp", "ml", "ai", "llm", "retriev", "rank", "python",
                "deep learn", "transformer", "recommendation"])
    ]
    avg_assessment = np.mean(relevant_assessments) / 100.0 if relevant_assessments else 0.3

    # Preferred work mode alignment (JD says hybrid)
    work_mode = sigs.get("preferred_work_mode", "")
    work_mode_score = {
        "hybrid": 1.0, "flexible": 0.9, "onsite": 0.7, "remote": 0.5
    }.get(work_mode, 0.6)

    # Salary range (JD doesn't specify, but we flag extreme outliers)
    sal = sigs.get("expected_salary_range_inr_lpa", {})
    sal_mid = (sal.get("min", 0) + sal.get("max", 0)) / 2.0
    # Senior AI Engineer at Series A: ~40-80 LPA is typical
    if 35 <= sal_mid <= 90:
        salary_fit = 1.0
    elif 25 <= sal_mid < 35 or 90 < sal_mid <= 120:
        salary_fit = 0.7
    elif sal_mid > 120:
        salary_fit = 0.4  # Overpriced for Series A
    else:
        salary_fit = 0.5  # Unknown / 0

    # Certifications
    relevant_cert_count = sum(
        1 for c in certs
        if any(kw in c.get("name", "").lower()
               for kw in ["ml", "ai", "nlp", "aws", "gcp", "azure", "deep", "data"])
    )
    cert_score = min(relevant_cert_count / 3.0, 1.0)

    # --- COMPOSITE BEHAVIORAL MULTIPLIER ---
    # This is a multiplicative modifier (0.3 - 1.2) applied to skill score
    behavioral_multiplier = (
        0.25 * recency_score +
        0.20 * recruiter_response +
        0.15 * open_to_work +
        0.10 * interview_completion +
        0.10 * response_speed +
        0.10 * profile_completeness +
        0.10 * offer_acc_norm
    )
    # Scale to 0.3-1.2 range
    behavioral_multiplier = 0.3 + behavioral_multiplier * 0.9

    return {
        # Skill features
        "must_have_hits": float(must_have_hits),
        "must_have_ratio": min(must_have_hits / 8.0, 1.0),
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
        "_open_to_work": float(sigs.get("open_to_work_flag", False)),
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
            }
    else:
        logger.info("Skipping embeddings (--no-embeddings flag set)")
        meta = {
            "jd_embedding": None,
            "embedding_model": None,
            "n_candidates": len(candidates),
            "feature_names": feature_names,
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
