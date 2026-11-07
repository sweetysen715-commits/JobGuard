
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

# 1. Load training and testing datasets
train_data = pd.read_csv("datasets/processed/train_data.csv")
test_data = pd.read_csv("datasets/processed/test_data.csv")

X_train = train_data.drop(columns=["fraudulent"])
y_train = train_data["fraudulent"]

X_test = test_data.drop(columns=["fraudulent"])
y_test = test_data["fraudulent"]

# 2. Define feature columns
numerical_columns = [
    "title_length", "company_profile_length",
    "description_length", "requirements_length",
    "benefits_length", "title_word_count",
    "company_profile_word_count", "description_word_count",
    "requirements_word_count", "benefits_word_count",
    "total_word_count", "salary_missing",
    "telecommuting", "has_company_logo", "has_questions"
]

categorical_columns = [
    "employment_type", "required_experience",
    "required_education", "industry", "function", "location"
]

text_columns = [
    "title", "company_profile", "description",
    "requirements", "benefits"
]

from text_utils import combine_text

# 4. Build preprocessing pipeline
preprocessor = ColumnTransformer([
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
                FunctionTransformer(combine_text, validate=False)
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
])

# 5. Build final Random Forest pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=2
        )
    )
])

# 6. Train using complete training data
print("Training final Random Forest...")

model.fit(X_train, y_train)

print("Training completed!")

# 7. Predict probabilities on untouched test data
probabilities = model.predict_proba(X_test)[:, 1]

# 8. Apply optimized threshold
best_threshold = 0.295

predictions = (
    probabilities >= best_threshold
).astype(int)

# 9. Evaluate
print("\nFINAL RANDOM FOREST TEST RESULTS")
print("--------------------------------")

print("Threshold:", best_threshold)
print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Legitimate", "Fraudulent"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print(
    "\nPR-AUC:",
    average_precision_score(y_test, probabilities)
)

print(
    "ROC-AUC:",
    roc_auc_score(y_test, probabilities)
)

# 10. Save the complete pipeline
joblib.dump(
    model,
    "models/final_random_forest_pipeline.pkl"
)

# Save threshold
with open("models/final_random_forest_threshold.txt", "w") as file:
    file.write(str(best_threshold))

print("\nFinal Random Forest pipeline and threshold saved!")
