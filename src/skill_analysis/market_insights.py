
import pandas as pd
from collections import Counter
from pathlib import Path

from skill_extractor import extract_skills


def analyze_market_skills(csv_path, top_n=15):

    data = pd.read_csv(csv_path)

    # Use legitimate job postings for market analysis
    data = data[data["fraudulent"] == 0].copy()

    skill_counts = Counter()

    total_jobs = len(data)

    for _, row in data.iterrows():

        job_text = " ".join([
            str(row.get("title", "")),
            str(row.get("description", "")),
            str(row.get("requirements", ""))
        ])

        # Count each skill only once per job posting
        skills = set(extract_skills(job_text))

        skill_counts.update(skills)

    results = []

    for skill, count in skill_counts.most_common():
        percentage = (count / total_jobs) * 100

        results.append({
            "skill": skill,
            "job_count": count,
            "market_percentage": round(percentage, 2)
        })

    insights = pd.DataFrame(results)

    output_path = Path("datasets/processed/market_skill_insights.csv")
    insights.to_csv(output_path, index=False)

    print("\nJOBGUARD JOB MARKET SKILL INSIGHTS")
    print("---------------------------------")
    print(f"Legitimate job postings analyzed: {total_jobs}")
    print("\nMost demanded skills:")
    print(insights.head(top_n).to_string(index=False))
    print(f"\nSaved insights to: {output_path}")

    return insights


if __name__ == "__main__":

    analyze_market_skills(
        "datasets/raw/fake_job_postings.csv",
        top_n=15
    )
