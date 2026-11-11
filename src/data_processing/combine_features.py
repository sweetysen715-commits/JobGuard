
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
train_data = pd.read_csv(
    "datasets/processed/train_data.csv"
)

test_data = pd.read_csv(
    "datasets/processed/test_data.csv"
)

numerical_columns = [
    "title_length",
    "company_profile_length",
    "description_length",
    "requirements_length",
    "benefits_length",
    "title_word_count",
    "company_profile_word_count",
    "description_word_count",
    "requirements_word_count",
    "benefits_word_count",
    "total_word_count",
    "salary_missing",
    "telecommuting",
    "has_company_logo",
    "has_questions"
]
categorical_columns = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "location_grouped"
]

location_counts = train_data["location"].fillna("Unknown").value_counts()

frequent_locations = location_counts[
    location_counts >= 10
].index

train_data["location_grouped"] = (
    train_data["location"]
    .fillna("Unknown")
    .where(
        train_data["location"].fillna("Unknown").isin(frequent_locations),
        "Other"
    )
)

test_data["location_grouped"] = (
    test_data["location"]
    .fillna("Unknown")
    .where(
        test_data["location"].fillna("Unknown").isin(frequent_locations),
        "Other"
    )
)
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

for column in text_columns:
    train_data[column] = train_data[column].fillna("").astype(str)
    test_data[column] = test_data[column].fillna("").astype(str)

train_data["combined_text"] = train_data[text_columns].agg(" ".join, axis=1)
test_data["combined_text"] = test_data[text_columns].agg(" ".join, axis=1)

y_train = train_data["fraudulent"]
y_test = test_data["fraudulent"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(with_mean=False),
            numerical_columns
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "text",
            TfidfVectorizer(
                max_features=5000,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            ),
            "combined_text"
        )
    ]
)

X_train = preprocessor.fit_transform(train_data)

X_test = preprocessor.transform(test_data)

save_npz(
    "datasets/processed/X_train.npz",
    X_train
)

save_npz(
    "datasets/processed/X_test.npz",
    X_test
)

np.save(
    "datasets/processed/y_train.npy",
    y_train.to_numpy()
)

np.save(
    "datasets/processed/y_test.npy",
    y_test.to_numpy()
)

print("Feature combination completed!")

print("\nTraining features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)

print("\nTraining labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)

print("\nAll processed files saved successfully!")
