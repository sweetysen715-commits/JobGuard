
import pandas as pd
import joblib
import shap
from text_utils import combine_text
from explainability import generate_explanations
# 1. Load test dataset
test_data = pd.read_csv(
    "datasets/processed/test_data.csv"
)
train_data = pd.read_csv(
    "datasets/processed/train_data.csv"
)
# 2. Select one job posting
sample = test_data.drop(
    columns=["fraudulent"]
).iloc[[0]]

actual_label = test_data["fraudulent"].iloc[0]

# 3. Load our trained Random Forest pipeline
model = joblib.load(
    "models/final_random_forest_pipeline.pkl"
)
print("\nPipeline steps:")
print(model.named_steps.keys())

# 4. Separate preprocessing and classifier
preprocessor = model.named_steps["preprocessing"]
classifier = model.named_steps["classifier"]

# 5. Transform the sample using trained preprocessing
sample_transformed = preprocessor.transform(sample)
# Convert sparse matrix to a dense numerical array
if hasattr(sample_transformed, "toarray"):
    sample_transformed = sample_transformed.toarray()

sample_transformed = sample_transformed.astype("float64")

print("Transformed data type:", sample_transformed.dtype)
print("Transformed shape:", sample_transformed.shape)


# Use a small background sample from training data
background_data = train_data.drop(
    columns=["fraudulent"]
).sample(n=30, random_state=42)

background_transformed = preprocessor.transform(background_data)

if hasattr(background_transformed, "toarray"):
    background_transformed = background_transformed.toarray()

background_transformed = background_transformed.astype("float64")

# Create probability-based SHAP explainer
explainer = shap.TreeExplainer(
    classifier,
    data=background_transformed,
    feature_perturbation="interventional",
    model_output="probability"
)

# Calculate SHAP values
shap_values = explainer(sample_transformed)

print("\nSHAP explanation generated successfully!")
print("SHAP values shape:", shap_values.values.shape)


# 8. Display prediction
probability = model.predict_proba(sample)[0][1]

with open(
    "models/final_random_forest_threshold.txt",
    "r"
) as file:
    threshold = float(file.read())

prediction = int(probability >= threshold)

print("\nJOBGUARD EXPLANATION")
print("--------------------")

print("Actual label:", actual_label)
print("Fraud probability:", round(probability * 100, 2), "%")
print("Threshold:", threshold)
print("Predicted class:", prediction)

# 9. Display SHAP output shape
print("\nSHAP explanation generated!")
print("SHAP values shape:", shap_values.values.shape)

# 10. Display top contributing features
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
# Generate feature names manually

# Numeric feature names
numeric_names = [
    f"numeric__{name}" for name in numeric_columns
]

# Categorical feature names
categorical_encoder = (
    preprocessor.named_transformers_["categorical"]
)

categorical_names = [
    f"categorical__{name}"
    for name in categorical_encoder.get_feature_names_out(
        categorical_columns
    )
]

# Text feature names
text_pipeline = preprocessor.named_transformers_["text"]

tfidf = text_pipeline.named_steps["tfidf"]

text_names = [
    f"text__{name}"
    for name in tfidf.get_feature_names_out()
]

# Combine all feature names in the same order as preprocessing
feature_names = (
    numeric_names +
    categorical_names +
    text_names
)

print("Total feature names:", len(feature_names))

# Select SHAP values for the fraudulent class (class 1)
values = shap_values.values[0, :, 1]

importance = pd.DataFrame({
    "Feature": feature_names,
    "SHAP_Value": values
})

importance["Absolute_Impact"] = importance["SHAP_Value"].abs()

importance = importance.sort_values(
    by="Absolute_Impact",
    ascending=False
)

print("\nTOP 15 IMPORTANT FEATURES FOR THIS JOB")
print(importance.head(15).to_string(index=False))
# Generate human-readable explanations

readable_explanations = generate_explanations(
    feature_names,
    values,
    top_n=10
)

print("\nJOBGUARD: HUMAN-READABLE EXPLANATION")
print("------------------------------------")

for item in readable_explanations:

    if item["direction"] == "increased":
        message = "Increased estimated fraud risk"
    else:
        message = "Decreased estimated fraud risk"

    print(f"{item['feature']}: {message}")