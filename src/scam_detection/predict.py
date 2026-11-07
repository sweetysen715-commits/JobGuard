from pathlib import Path
import sys
import joblib
import pandas as pd


# Locate the project and model files
BASE_DIR = Path(__file__).resolve().parents[2]

SCAM_DIR = BASE_DIR / "src" / "scam_detection"
sys.path.insert(0, str(SCAM_DIR))

# Required when loading the saved model
import text_utils


MODEL_PATH = BASE_DIR / "models" / "final_random_forest_pipeline.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "final_random_forest_threshold.txt"


# Load the trained model and threshold
model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as file:
    THRESHOLD = float(file.read().strip())


TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

NUMERICAL_COLUMNS = [
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

CATEGORICAL_COLUMNS = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "location"
]


def prepare_job_data(job_data):

    row = {}

    # Prepare text columns
    for column in TEXT_COLUMNS:
        row[column] = str(job_data.get(column) or "")

    # Calculate text lengths and word counts
    total_word_count = 0

    for column in TEXT_COLUMNS:

        text = row[column]

        row[column + "_length"] = len(text)

        word_count = len(text.split())

        row[column + "_word_count"] = word_count

        total_word_count += word_count

    row["total_word_count"] = total_word_count

    # Prepare salary information
    salary = str(job_data.get("salary_range") or "").strip()

    row["salary_missing"] = int(
        not salary or salary.lower() in ["unknown", "not disclosed"]
    )

    # Prepare numerical flags
    row["telecommuting"] = int(job_data.get("telecommuting", 0))
    row["has_company_logo"] = int(job_data.get("has_company_logo", 0))
    row["has_questions"] = int(job_data.get("has_questions", 0))

    # Prepare categorical columns
    for column in CATEGORICAL_COLUMNS:
        value = job_data.get(column)
        row[column] = str(value).strip() if value else "Unknown"

    # Return features in the training column order
    feature_columns = (
        TEXT_COLUMNS +
        NUMERICAL_COLUMNS +
        CATEGORICAL_COLUMNS
    )

    return pd.DataFrame([row])[feature_columns]


def predict_job_scam(job_data):

    input_data = prepare_job_data(job_data)

    # Predict probability of fraud
    probability = model.predict_proba(input_data)[0][1]

    # Apply the saved threshold
    prediction = int(probability >= THRESHOLD)

    if prediction == 1:
        label = "Potentially Fraudulent"
    else:
        label = "Likely Legitimate"

    return {
        "prediction": prediction,
        "label": label,
        "fraud_probability": round(float(probability) * 100, 2),
        "threshold": THRESHOLD
    }