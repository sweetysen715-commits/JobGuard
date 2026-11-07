
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load dataset
data = pd.read_csv(
    "datasets/processed/jobs_feature_engineered.csv"
)

# Select categorical columns
categorical_columns = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]

# Create a copy
df = data.copy()

# Group rare locations
location_counts = df["location"].value_counts()

df["location_grouped"] = df["location"].where(
    df["location"].map(location_counts) >= 10,
    "Other"
)

# Select features for encoding
X_categorical = df[
    categorical_columns + ["location_grouped"]
].fillna("Unknown")

# Initialize encoder
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

# Transform categorical data
encoded_data = encoder.fit_transform(X_categorical)

# Convert into DataFrame
encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out()
)

print("Categorical encoding completed!")
print("Original categorical shape:", X_categorical.shape)
print("Encoded categorical shape:", encoded_df.shape)

print("\nSample encoded data:")
print(encoded_df.head())
