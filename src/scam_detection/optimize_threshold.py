
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    confusion_matrix
)

# 1. Load training features and labels
X_train = load_npz("datasets/processed/X_train.npz")
y_train = np.load("datasets/processed/y_train.npy")

# Load original training data for duplicate grouping
data = pd.read_csv("datasets/processed/train_data.csv")

# 2. Create groups to keep repeated postings together
group_columns = [
    "title",
    "company_profile",
    "description"
]

groups = pd.MultiIndex.from_frame(
    data[group_columns].fillna("").astype(str)
).factorize()[0]

# 3. Create cross-validation splitter
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Store predictions for validation folds
oof_probabilities = np.zeros(len(y_train))

# 4. Train and validate across 5 folds
for fold, (train_idx, val_idx) in enumerate(
    cv.split(X_train, y_train, groups), start=1
):

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=42
    )

    model.fit(X_train[train_idx], y_train[train_idx])

    oof_probabilities[val_idx] = model.predict_proba(
        X_train[val_idx]
    )[:, 1]

    print(f"Completed fold {fold}")

# 5. Find the best threshold using F1-score
precision, recall, thresholds = precision_recall_curve(
    y_train,
    oof_probabilities
)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

# 6. Generate validation predictions
oof_predictions = (
    oof_probabilities >= best_threshold
).astype(int)

print("\nThreshold Optimization Results")
print("--------------------------------")

print("Best threshold:", best_threshold)
print("Best validation F1-score:", f1_scores[best_index])

print("\nClassification Report:")
print(classification_report(
    y_train,
    oof_predictions,
    target_names=["Legitimate", "Fraudulent"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_train, oof_predictions))

# 7. Save selected threshold
with open("models/best_threshold.txt", "w") as file:
    file.write(str(best_threshold))

print("\nBest threshold saved successfully!")
