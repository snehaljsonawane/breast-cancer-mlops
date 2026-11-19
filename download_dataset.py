
import pandas as pd
from sklearn.datasets import load_breast_cancer
from pathlib import Path

# Load Wisconsin Breast Cancer dataset
dataset = load_breast_cancer()

# Convert features into a DataFrame
df = pd.DataFrame(
    dataset.data,
    columns=dataset.feature_names
)

# Add target column
df["target"] = dataset.target

# Create raw data directory
Path("data/raw").mkdir(parents=True, exist_ok=True)

# Save dataset
df.to_csv("data/raw/breast_cancer.csv", index=False)

print("Dataset downloaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nTarget distribution:")
print(df["target"].value_counts())
