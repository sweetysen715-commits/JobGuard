
import re


# Skills and their alternative names
SKILL_DICTIONARY = {
    "Python": ["python"],
    "Java": ["java"],
    "C++": ["c++"],
    "C#": ["c#"],
    "JavaScript": ["javascript", "js"],
    "SQL": ["sql", "structured query language"],
    "HTML": ["html"],
    "CSS": ["css"],
    "React": ["react", "react.js"],
    "Node.js": ["node.js", "nodejs"],
    "Flask": ["flask"],
    "Django": ["django"],
    "Spring Boot": ["spring boot"],
    "Machine Learning": ["machine learning", "ml"],
    "Artificial Intelligence": ["artificial intelligence", "ai"],
"Deep Learning": ["deep learning", "dl"],
"Natural Language Processing": ["natural language processing", "nlp"],
"Computer Vision": ["computer vision", "cv"],
"Data Visualization": ["data visualization"],
"Statistics": ["statistics", "statistical analysis"],
"Probability": ["probability"],
"Power BI": ["power bi", "powerbi"],
"Scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
"Microsoft Azure": ["azure", "microsoft azure"],
"Google Cloud": ["google cloud", "gcp"],
"REST API": ["rest api", "restful api"],
"Microservices": ["microservices", "micro-services"],
"CI/CD": ["ci/cd", "continuous integration", "continuous deployment"],
    "Deep Learning": ["deep learning"],
    "Data Science": ["data science"],
    "Data Analysis": ["data analysis"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Matplotlib": ["matplotlib"],
    "Scikit-learn": ["scikit-learn", "sklearn"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Excel": ["excel", "microsoft excel"],
    "Power BI": ["power bi"],
    "Tableau": ["tableau"],
    "Git": ["git"],
    "GitHub": ["github"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes"],
    "AWS": ["aws", "amazon web services"],
    "Linux": ["linux"],
    "MongoDB": ["mongodb"],
    "MySQL": ["mysql"],
    "PostgreSQL": ["postgresql"],
    "Communication": ["communication skills"],
    "Problem Solving": ["problem solving", "problem-solving"],
}
def normalize_skill(skill):

    skill = skill.strip().lower()

    for standard_name, aliases in SKILL_DICTIONARY.items():

        if skill == standard_name.lower():
            return standard_name

        if skill in [alias.lower() for alias in aliases]:
            return standard_name

    return skill.title()
print("\nSKILL NORMALIZATION TEST")
print("------------------------")

test_skills = ["ML", "sklearn", "JS", "NLP", "GCP"]

for skill in test_skills:
    print(skill, "->", normalize_skill(skill))

def extract_skills(text):

    if not isinstance(text, str):
        return []

    text = text.lower()

    found_skills = []

    for skill, keywords in SKILL_DICTIONARY.items():

        for keyword in keywords:

            pattern = (
                r"(?<![a-z0-9+#.])"
                + re.escape(keyword)
                + r"(?![a-z0-9+#.])"
            )

            if re.search(pattern, text):
                found_skills.append(skill)
                break

    return sorted(found_skills)


# Test the skill extractor
if __name__ == "__main__":

    sample_resume = """
    I know Python, Java, SQL, Pandas, NumPy,
    Machine Learning, Flask, Git and Docker.
    """

    skills = extract_skills(sample_resume)

    print("EXTRACTED SKILLS")
    print("----------------")

    for skill in skills:
        print(skill)
