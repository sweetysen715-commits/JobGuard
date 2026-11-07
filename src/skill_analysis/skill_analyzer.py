from learning_recommender import recommend_learning
from skill_extractor import extract_skills, normalize_skill
from skill_matcher import compare_skills
from semantic_skill_matcher import semantic_match
from resume_parser import extract_resume_text
from career_readiness import calculate_readiness
from skill_dependency import get_learning_path

def analyze_skill_gap(resume_text, job_description):

    # Step 1: Extract skills
    resume_skills = extract_skills(resume_text)
    required_skills = extract_skills(job_description)

    # Step 2: Normalize skills
    resume_skills = sorted({
        normalize_skill(skill) for skill in resume_skills
    })

    required_skills = sorted({
        normalize_skill(skill) for skill in required_skills
    })

    # Step 3: Perform exact matching
    exact_result = compare_skills(
        resume_skills,
        required_skills
    )

    exact_matches = exact_result["matched_skills"]

    # Step 4: Find requirements not matched exactly
    unmatched_requirements = exact_result["missing_skills"]

    # Step 5: Perform semantic matching on remaining requirements
    semantic_matches = semantic_match(
        resume_skills,
        unmatched_requirements
    )

    # Step 6: Combine exact and semantic matches
    matched_skills = []

    for skill in exact_matches:
        matched_skills.append({
            "required_skill": skill,
            "resume_skill": skill,
            "similarity": 1.0,
            "match_type": "Exact"
        })

    for match in semantic_matches:
        matched_skills.append({
            "required_skill": match["required_skill"],
            "resume_skill": match["resume_skill"],
            "similarity": match["similarity"],
            "match_type": "Semantic"
        })

    # Step 7: Identify missing skills
    matched_requirements = {
        match["required_skill"] for match in matched_skills
    }

    missing_skills = sorted(
        set(required_skills) - matched_requirements
    )

    # Step 8: Identify extra skills
    used_resume_skills = {
        match["resume_skill"] for match in matched_skills
    }

    extra_skills = sorted(
        set(resume_skills) - used_resume_skills
    )

    # Step 9: Calculate match percentage
    if required_skills:
        match_percentage = (
            len(matched_skills) / len(required_skills)
        ) * 100
    else:
        match_percentage = 0
    readiness = calculate_readiness({
        "required_skills": required_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    })
    return {
        "resume_skills": resume_skills,
        "required_skills": required_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "learning_recommendations": recommend_learning(missing_skills),
        "learning_path": get_learning_path(missing_skills),
        "extra_skills": extra_skills,
        "career_readiness": readiness,
        "match_percentage": round(match_percentage, 2)
    }


def analyze_resume_file(resume_path, job_description):

    resume_text = extract_resume_text(resume_path)

    return analyze_skill_gap(
        resume_text,
        job_description
    )


if __name__ == "__main__":

    resume = """
    I have knowledge of Python, Java, SQL,
    Pandas, NumPy and Git.
    """

    job_description = """
    We are looking for a candidate with Python,
    SQL, Pandas, Machine Learning, Flask and Docker.
    """

    report = analyze_skill_gap(resume, job_description)

    print("\nJOBGUARD COMPLETE SKILL ANALYSIS")
    print("--------------------------------")

    for key, value in report.items():
        print(f"{key}: {value}")
