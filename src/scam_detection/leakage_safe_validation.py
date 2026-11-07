
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve
)

# 1. Load original training data
data = pd.read_csv("datasets/processed/train_data.csv")

X = data.drop(columns=["fraudulent"])
y = data["fraudulent"]

# 2. Define feature groups
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
    "location"
]

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

# 3. Combine text columns into one document per job
def combine_text(data):
    return data.fillna("").astype(str).agg(" ".join, axis=1)

# 4. Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(with_mean=False),
            numerical_columns
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=10
            ),
            categorical_columns
        ),
        (
            "text",
            Pipeline([
                (
                    "combine",
                    FunctionTransformer(
                        combine_text,
                        validate=False
                    )
                ),
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=5000,
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95
                    )
                )
            ]),
            text_columns
        )
    ]
)

# 5. Create complete ML pipeline
model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    (
        "model",
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear",
            random_state=42
        )
    )
])

# 6. Create groups to prevent duplicate-content leakage
group_columns = [
    "title",
    "company_profile",
    "description"
]

groups = pd.MultiIndex.from_frame(
    data[group_columns].fillna("").astype(str)
).factorize()[0]

# 7. Create grouped cross-validation
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 8. Generate out-of-fold fraud probabilities
print("Starting leakage-safe cross-validation...")

probabilities = cross_val_predict(
    model_pipeline,
    X,
    y,
    cv=cv,
    groups=groups,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

print("Cross-validation completed!")

# 9. Find the best threshold using validation predictions
precision, recall, thresholds = precision_recall_curve(
    y,
    probabilities
)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

predictions = (
    probabilities >= best_threshold
).astype(int)

# 10. Display results
print("\nLEAKAGE-SAFE VALIDATION RESULTS")
print("--------------------------------")

print("Best threshold:", best_threshold)
print("Best F1-score:", f1_scores[best_index])

print("\nClassification Report:")
print(classification_report(
    y,
    predictions,
    target_names=["Legitimate", "Fraudulent"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y, predictions))

# 11. Save the new threshold separately
with open(
    "models/leakage_safe_threshold.txt",
    "w"
) as file:
    file.write(str(best_threshold))

print("\nLeakage-safe threshold saved!")
