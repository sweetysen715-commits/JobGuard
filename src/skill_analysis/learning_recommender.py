import pandas as pd
from pathlib import Path
# Learning recommendations for missing skills

LEARNING_RESOURCES = {

    "Python": {
        "topics": ["Python fundamentals", "Functions", "OOP", "File handling"],
        "level": "Beginner"
    },

    "Machine Learning": {
        "topics": ["Regression", "Classification", "Clustering", "Model evaluation"],
        "level": "Intermediate"
    },

    "Flask": {
        "topics": ["Flask basics", "Routing", "REST APIs", "Database integration"],
        "level": "Intermediate"
    },

    "SQL": {
        "topics": ["Joins", "Subqueries", "Window functions", "Query optimization"],
        "level": "Beginner"
    },

    "Docker": {
        "topics": ["Images", "Containers", "Dockerfiles", "Docker Compose"],
        "level": "Intermediate"
    },

    "Pandas": {
        "topics": ["DataFrames", "Data cleaning", "GroupBy", "Data analysis"],
        "level": "Beginner"
    },

    "NumPy": {
        "topics": ["Arrays", "Indexing", "Broadcasting", "Matrix operations"],
        "level": "Beginner"
    },

    "Java": {
        "topics": ["OOP", "Collections", "Exception handling", "Multithreading"],
        "level": "Beginner"
    },

    "Scikit-learn": {
        "topics": ["Preprocessing", "Model training", "Pipelines", "Evaluation"],
        "level": "Intermediate"
    }
}

def load_market_insights():

    file_path = Path(
        "datasets/processed/market_skill_insights.csv"
    )

    if not file_path.exists():
        return {}

    data = pd.read_csv(file_path)

    return {
        row["skill"]: row["market_percentage"]
        for _, row in data.iterrows()
    }


def recommend_learning(missing_skills):
    market_insights = load_market_insights()
    recommendations = []

    for skill in missing_skills:

        if skill in LEARNING_RESOURCES:

            recommendations.append({
                "skill": skill,
                "level": LEARNING_RESOURCES[skill]["level"],
                "topics": LEARNING_RESOURCES[skill]["topics"],
                "market_percentage": market_insights.get(skill, 0)
            })

        else:

            recommendations.append({
                "skill": skill,
                "level": "Not classified",
                "topics": ["Explore fundamentals and practical projects"]
            })

    return recommendations


if __name__ == "__main__":

    missing_skills = [
        "Flask",
        "Machine Learning",
        "Docker"
    ]

    results = recommend_learning(missing_skills)

    print("\nJOBGUARD LEARNING RECOMMENDATIONS")
    print("--------------------------------")

    for item in results:

        print("\nSkill:", item["skill"])
        print("Level:", item["level"])
        print("Topics:")

        for topic in item["topics"]:
            print("-", topic)
