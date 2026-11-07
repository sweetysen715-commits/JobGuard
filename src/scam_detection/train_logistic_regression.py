
import numpy as np
import joblib

from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    average_precision_score,
    roc_auc_score
)

# 1. Load training and testing features
X_train = load_npz("datasets/processed/X_train.npz")
X_test = load_npz("datasets/processed/X_test.npz")

y_train = np.load("datasets/processed/y_train.npy")
y_test = np.load("datasets/processed/y_test.npy")

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# 2. Create Logistic Regression model
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    solver="liblinear",
    random_state=42
)

# 3. Train the model
print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

# 4. Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 5. Evaluate the model
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Fraudulent"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nPR-AUC (Average Precision):")
print(average_precision_score(y_test, y_prob))

print("\nROC-AUC:")
print(roc_auc_score(y_test, y_prob))

# 6. Save the trained model
joblib.dump(
    model,
    "models/logistic_regression.pkl"
)

print("\nModel saved successfully!")
