
import pandas as pd

data = pd.read_csv(
    "datasets/processed/jobs_feature_engineered.csv"
)

categorical_columns = [
    "location",
    "department",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]

summary = []

for column in categorical_columns:
    summary.append({
        "Feature": column,
        "Unique Categories": data[column].nunique(),
        "Unknown Values": (data[column] == "Unknown").sum(),
        "Missing Values": data[column].isnull().sum()
    })

summary_df = pd.DataFrame(summary)

print("\nCategorical Feature Summary:")
print(summary_df.to_string(index=False))

