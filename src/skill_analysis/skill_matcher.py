
def compare_skills(resume_skills, required_skills):

    resume_set = set(resume_skills)
    required_set = set(required_skills)

    # Skills present in both
    matched_skills = resume_set.intersection(required_set)

    # Skills missing from the resume
    missing_skills = required_set - resume_set

    # Extra skills in the resume
    extra_skills = resume_set - required_set

    # Calculate match percentage
    if len(required_set) == 0:
        match_percentage = 0
    else:
        match_percentage = (
            len(matched_skills) / len(required_set)
        ) * 100

    return {
        "matched_skills": sorted(matched_skills),
        "missing_skills": sorted(missing_skills),
        "extra_skills": sorted(extra_skills),
        "match_percentage": round(match_percentage, 2)
    }


# Test the skill matching engine
if __name__ == "__main__":

    resume_skills = [
        "Python",
        "Java",
        "SQL",
        "Pandas",
        "Git"
    ]

    required_skills = [
        "Python",
        "SQL",
        "Pandas",
        "Machine Learning",
        "Flask",
        "Docker"
    ]

    result = compare_skills(resume_skills, required_skills)

    print("\nJOBGUARD SKILL GAP REPORT")
    print("------------------------")

    print("Matched Skills:", result["matched_skills"])
    print("Missing Skills:", result["missing_skills"])
    print("Extra Skills:", result["extra_skills"])
    print("Match Percentage:", result["match_percentage"], "%")
