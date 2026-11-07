
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Load feature-engineered dataset
data = pd.read_csv(
    "datasets/processed/jobs_feature_engineered.csv"
)

# Target variable
y = data["fraudulent"]

# Group similar job postings together
group_columns = [
    "title",
    "company_profile",
    "description"
]

groups = pd.MultiIndex.from_frame(
    data[group_columns].fillna("").astype(str)
).factorize()[0]

# Create stratified group splitter
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Generate train-test indices
train_index, test_index = next(
    splitter.split(data, y, groups)
)

# Create separate datasets
train_data = data.iloc[train_index].copy()
test_data = data.iloc[test_index].copy()

# Save datasets
train_data.to_csv(
    "datasets/processed/train_data.csv",
    index=False
)

test_data.to_csv(
    "datasets/processed/test_data.csv",
    index=False
)

# Display results
print("Train-test splitting completed!")

print("\nTraining dataset shape:", train_data.shape)
print("Testing dataset shape:", test_data.shape)

print("\nTraining fraud distribution:")
print(train_data["fraudulent"].value_counts(normalize=True) * 100)

print("\nTesting fraud distribution:")
print(test_data["fraudulent"].value_counts(normalize=True) * 100)

print(
    "\nCommon groups between train and test:",
    len(set(groups[train_index]) & set(groups[test_index]))
)
