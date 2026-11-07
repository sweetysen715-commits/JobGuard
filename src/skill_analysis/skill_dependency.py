
# Skill prerequisites and learning dependencies

SKILL_DEPENDENCIES = {

    "Machine Learning": [
        "Python",
        "NumPy",
        "Pandas",
        "Statistics"
    ],

    "Deep Learning": [
        "Python",
        "Machine Learning",
        "Linear Algebra"
    ],

    "Data Science": [
        "Python",
        "Statistics",
        "NumPy",
        "Pandas"
    ],

    "Flask": [
        "Python",
        "HTML",
        "CSS"
    ],

    "Django": [
        "Python",
        "HTML",
        "CSS",
        "SQL"
    ],

    "Scikit-learn": [
        "Python",
        "NumPy",
        "Pandas",
        "Machine Learning"
    ],

    "Power BI": [
        "Excel",
        "SQL",
        "Data Analysis"
    ]
}


def get_learning_path(missing_skills):

    learning_path = []
    visited = set()

    def visit(skill):

        if skill in visited:
            return

        visited.add(skill)

        prerequisites = SKILL_DEPENDENCIES.get(skill, [])

        for prerequisite in prerequisites:
            visit(prerequisite)

        learning_path.append(skill)

    for skill in missing_skills:
        visit(skill)

    return learning_path


if __name__ == "__main__":

    missing_skills = [
        "Machine Learning",
        "Flask"
    ]

    path = get_learning_path(missing_skills)

    print("\nJOBGUARD SKILL LEARNING PATH")
    print("----------------------------")

    for index, skill in enumerate(path, start=1):
        print(f"{index}. {skill}")
