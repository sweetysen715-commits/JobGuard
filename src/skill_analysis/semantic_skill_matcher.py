
from skill_extractor import normalize_skill
from sentence_transformers import SentenceTransformer, util

# Load a lightweight sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_match(resume_skills, required_skills, threshold=0.80):

    # Normalize skill names
    resume_skills = sorted({
        normalize_skill(skill) for skill in resume_skills
    })

    required_skills = sorted({
        normalize_skill(skill) for skill in required_skills
    })

    if not resume_skills or not required_skills:
        return []

    # Convert skills into numerical embeddings
    resume_embeddings = model.encode(
        resume_skills,
        convert_to_tensor=True
    )

    required_embeddings = model.encode(
        required_skills,
        convert_to_tensor=True
    )

    # Calculate similarity between every pair
    similarity_matrix = util.cos_sim(
        required_embeddings,
        resume_embeddings
    )

    # One-to-one matching
    matched_skills = []
    used_resume_skills = set()

    for i, required_skill in enumerate(required_skills):

        best_index = -1
        best_score = 0

        for j, resume_skill in enumerate(resume_skills):

            if j in used_resume_skills:
                continue

            score = similarity_matrix[i][j].item()

            if score > best_score:
                best_score = score
                best_index = j

        if best_index != -1 and best_score >= threshold:

            matched_skills.append({
                "required_skill": required_skill,
                "resume_skill": resume_skills[best_index],
                "similarity": round(best_score, 3)
            })

            used_resume_skills.add(best_index)

    return matched_skills


if __name__ == "__main__":

    resume_skills = [
        "Python",
        "Machine Learning",
        "SQL",
        "Data Analysis",
        "Power BI"
    ]

    required_skills = [
        "Python programming",
        "ML",
        "Database querying",
        "Data visualization",
        "Deep Learning"
    ]

    results = semantic_match(
        resume_skills,
        required_skills
    )

    print("\nJOBGUARD SEMANTIC SKILL MATCHING")
    print("--------------------------------")

    for result in results:
        print(
            f"Required: {result['required_skill']} | "
            f"Resume: {result['resume_skill']} | "
            f"Similarity: {result['similarity']}"
        )

