
import pandas as pd

# Load cleaned dataset
data = pd.read_csv("datasets/processed/jobs_cleaned.csv")

# Create a copy
df = data.copy()

# Text columns
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

# Convert all text columns to strings
for column in text_columns:
    df[column] = df[column].fillna("").astype(str)

# 1. Calculate character length
for column in text_columns:
    df[column + "_length"] = df[column].apply(len)

# 2. Calculate word count
for column in text_columns:
    df[column + "_word_count"] = (
        df[column].apply(lambda x: len(x.split()))
    )

# 3. Calculate total word count
word_count_columns = [
    column + "_word_count" for column in text_columns
]

df["total_word_count"] = df[word_count_columns].sum(axis=1)

# Save dataset
df.to_csv(
    "datasets/processed/jobs_feature_engineered.csv",
    index=False
)

print("Feature engineering completed!")
print("New dataset shape:", df.shape)
print("\nNew features:")
print(df.columns.tolist())
print("\nFeature statistics:")

print(
    df[
        [
            "title_length",
            "description_length",
            "requirements_length",
            "total_word_count"
        ]
    ].describe()
)
