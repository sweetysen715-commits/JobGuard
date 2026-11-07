
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load training and testing datasets
train_data = pd.read_csv(
    "datasets/processed/train_data.csv"
)

test_data = pd.read_csv(
    "datasets/processed/test_data.csv"
)

# Fill missing text values
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

for column in text_columns:
    train_data[column] = train_data[column].fillna("")
    test_data[column] = test_data[column].fillna("")

# Combine job posting text
train_data["combined_text"] = (
    train_data[text_columns].agg(" ".join, axis=1)
)

test_data["combined_text"] = (
    test_data[text_columns].agg(" ".join, axis=1)
)

# Initialize TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

# Fit only on training data
X_train_text = tfidf.fit_transform(
    train_data["combined_text"]
)

# Transform testing data using the same vocabulary
X_test_text = tfidf.transform(
    test_data["combined_text"]
)

print("TF-IDF processing completed!")

print("\nTraining text shape:", X_train_text.shape)
print("Testing text shape:", X_test_text.shape)

print("\nSample important vocabulary:")
print(tfidf.get_feature_names_out()[:30])
