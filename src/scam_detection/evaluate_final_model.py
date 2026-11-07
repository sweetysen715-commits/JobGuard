
import numpy as np
from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

import joblib

# 1. Load training and testing data
X_train = load_npz("datasets/processed/X_train.npz")
X_test = load_npz("datasets/processed/X_test.npz")

y_train = np.load("datasets/processed/y_train.npy")
y_test = np.load("datasets/processed/y_test.npy")

# 2. Load the selected threshold
with open("models/best_threshold.txt", "r") as file:
    best_threshold = float(file.read())

print("Selected threshold:", best_threshold)

# 3. Train the final Logistic Regression model
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    solver="liblinear",
    random_state=42
)

model.fit(X_train, y_train)

# 4. Get fraud probabilities on the test set
y_prob = model.predict_proba(X_test)[:, 1]

# 5. Apply the optimized threshold
y_pred = (y_prob >= best_threshold).astype(int)

# 6. Evaluate the final model
print("\nFINAL TEST RESULTS")
print("--------------------------")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Fraudulent"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(
    "\nPR-AUC:",
    average_precision_score(y_test, y_prob)
)

print(
    "ROC-AUC:",
    roc_auc_score(y_test, y_prob)
)

# 7. Save final model
joblib.dump(model, "models/final_logistic_regression.pkl")

print("\nFinal model saved successfully!")
