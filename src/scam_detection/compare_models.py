
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from xgboost import XGBClassifier

# 1. Load training data
data = pd.read_csv("datasets/processed/train_data.csv")

X = data.drop(columns=["fraudulent"])
y = data["fraudulent"]

# 2. Define feature columns
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

# 3. Combine text columns
def combine_text(data):
    return data.fillna("").astype(str).agg(" ".join, axis=1)

# 4. Create preprocessing pipeline
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

class_weight_ratio = (
    (y == 0).sum() / (y == 1).sum()
)


models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=2
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=class_weight_ratio,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=2
    )
}

group_columns = [
    "title",
    "company_profile",
    "description"
]

groups = pd.MultiIndex.from_frame(
    data[group_columns].fillna("").astype(str)
).factorize()[0]

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 7. Define evaluation metrics
scoring = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "average_precision": "average_precision",
    "roc_auc": "roc_auc"
}

results = []

# 8. Train and evaluate each algorithm
for name, algorithm in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", algorithm)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        groups=groups,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False
    )

    results.append({
        "Model": name,
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1-score": scores["test_f1"].mean(),
        "PR-AUC": scores["test_average_precision"].mean(),
        "ROC-AUC": scores["test_roc_auc"].mean()
    })

    print(f"{name} completed!")

# 9. Display comparison
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="PR-AUC",
    ascending=False
)

print("\nFINAL MODEL COMPARISON")
print("----------------------------")
print(results_df.to_string(index=False))

# 10. Save comparison
results_df.to_csv(
    "datasets/processed/model_comparison.csv",
    index=False
)

print("\nComparison saved successfully!")
