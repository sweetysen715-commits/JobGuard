def make_readable_name(feature):

    if feature.startswith("numeric__"):
        name = feature.replace("numeric__", "")

        descriptions = {
            "has_company_logo": "Company logo information",
            "has_questions": "Screening questions",
            "company_profile_length": "Company profile length",
            "company_profile_word_count": "Company profile word count",
            "salary_missing": "Salary information",
            "telecommuting": "Remote work information",
            "title_length": "Job title length",
            "description_length": "Job description length",
            "requirements_length": "Requirements length",
        }

        return descriptions.get(
            name,
            name.replace("_", " ").title()
        )

    elif feature.startswith("categorical__"):
        name = feature.replace("categorical__", "")
        return name.replace("_", " ").title()

    elif feature.startswith("text__"):
        word = feature.replace("text__", "")
        return f"Text signal: '{word}'"

    return feature


def generate_explanations(feature_names, shap_values, top_n=10):

    if len(feature_names) != len(shap_values):
        raise ValueError("Feature names and SHAP values do not match.")

    ranked_features = sorted(
        zip(feature_names, shap_values),
        key=lambda item: abs(item[1]),
        reverse=True
    )

    explanations = []

    for feature, value in ranked_features[:top_n]:

        readable_name = make_readable_name(feature)

        if value > 0:
            direction = "increased"
        else:
            direction = "decreased"

        explanations.append({
            "feature": readable_name,
            "impact": round(float(value), 5),
            "direction": direction
        })

    return explanations