
def calculate_readiness(report):

    required_skills = report["required_skills"]
    matched_skills = report["matched_skills"]
    missing_skills = report["missing_skills"]

    total_required = len(required_skills)

    if total_required == 0:
        return {
            "readiness_score": 0,
            "readiness_level": "Cannot calculate"
        }

    # Give exact and semantic matches credit
    total_score = sum(
        match["similarity"] for match in matched_skills
    )

    readiness_score = round(
        (total_score / total_required) * 100, 2
    )

    # Classify readiness
    if readiness_score >= 85:
        level = "Highly Prepared"

    elif readiness_score >= 70:
        level = "Well Prepared"

    elif readiness_score >= 50:
        level = "Moderately Prepared"

    else:
        level = "Needs Improvement"

    return {
        "readiness_score": readiness_score,
        "readiness_level": level,
        "matched_count": len(matched_skills),
        "missing_count": len(missing_skills),
        "total_required": total_required
    }


if __name__ == "__main__":

    sample_report = {
        "required_skills": [
            "Python", "SQL", "Machine Learning",
            "Flask", "Docker"
        ],
        "matched_skills": [
            {"required_skill": "Python", "similarity": 1.0},
            {"required_skill": "SQL", "similarity": 1.0},
            {"required_skill": "Machine Learning", "similarity": 0.9}
        ],
        "missing_skills": ["Flask", "Docker"]
    }

    result = calculate_readiness(sample_report)

    print("\nJOBGUARD CAREER READINESS")
    print("-------------------------")

    for key, value in result.items():
        print(f"{key}: {value}")
