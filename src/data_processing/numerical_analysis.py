
import pandas as pd

# Load dataset
data = pd.read_csv(
    "datasets/processed/jobs_feature_engineered.csv"
)

# Select numerical features
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

# Extract numerical data
X_numerical = data[numerical_columns]

print("Numerical feature analysis completed!")

print("\nNumerical dataset shape:")
print(X_numerical.shape)

print("\nMissing values:")
print(X_numerical.isnull().sum())

print("\nStatistical summary:")
print(X_numerical.describe())
