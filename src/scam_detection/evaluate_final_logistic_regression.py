
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

# 1. Load datasets
train_data = pd.read_csv("datasets/processed/train_data.csv")
test_data = pd.read_csv("datasets/processed/test_data.csv")

# 2. Define feature columns
numeric_columns = [
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
    return (
        data.fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

# 4. Separate features and target
X_train = train_data.drop(columns=["fraudulent"])
y_train = train_data["fraudulent"]

X_test = test_data.drop(columns=["fraudulent"])
y_test = test_data["fraudulent"]

# 5. Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(with_mean=False), numeric_columns),

        ("categorical",
         OneHotEncoder(
             handle_unknown="infrequent_if_exist",
             min_frequency=10
         ),
         categorical_columns),

        ("text",
         Pipeline([
             ("combine", FunctionTransformer(
                 combine_text,
                 validate=False
             )),
             ("tfidf", TfidfVectorizer(
                 max_features=5000,
                 stop_words="english",
                 ngram_range=(1, 2),
                 min_df=2,
                 max_df=0.95
             ))
         ]),
         text_columns)
    ]
)

# 6. Build final Logistic Regression pipeline
model = Pipeline([
    ("preprocessor", preprocessor),

    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=42
    ))
])

# 7. Train the model
print("Training final Logistic Regression...")

model.fit(X_train, y_train)

print("Training completed!")

# 8. Predict probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# 9. Apply leakage-safe validation threshold
threshold = 0.7577527427874831

y_pred = (y_prob >= threshold).astype(int)

# 10. Evaluate model
print("\nFINAL LOGISTIC REGRESSION TEST RESULTS")
print("--------------------------------------")

print("Threshold:", threshold)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Fraudulent"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nPR-AUC:", average_precision_score(y_test, y_prob))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# 11. Save final model and threshold
joblib.dump(
    model,
    "models/final_logistic_regression_pipeline.pkl"
)

with open("models/final_logistic_regression_threshold.txt", "w") as file:
    file.write(str(threshold))

print("\nFinal Logistic Regression pipeline and threshold saved!")
